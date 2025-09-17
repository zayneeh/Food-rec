import os
from traceback import format_exc
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from starlette.requests import Request
import json
from google.oauth2 import service_account

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ────────────────────────────────────────────────────────────────────────────────
# Environment & Config
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()  # load .env when running locally

PROJECT_ID = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GCP_PROJECT")
LOCATION   = os.environ.get("GCP_LOCATION", "us-central1")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")

# Handle Google Cloud credentials properly
credentials = None
creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if creds_json:
    try:
        # If it's a file path
        if os.path.isfile(creds_json):
            credentials = service_account.Credentials.from_service_account_file(creds_json)
        else:
            # If it's JSON content
            creds_info = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(creds_info)
    except Exception as e:
        print(f"Failed to load credentials: {e}")
        credentials = None

app = FastAPI(title="Nigerian Food Recommender API")

# *** TEMPORARY DEBUG - More permissive CORS ***
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],               # Temporarily allow all origins
    allow_credentials=False,           # Keep False for wildcard origins
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Log every request (helps confirm OPTIONS preflight + POST are hitting)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    resp = await call_next(request)
    print(f"{request.method} {request.url.path} -> {resp.status_code}")
    return resp

# ────────────────────────────────────────────────────────────────────────────────
# LLM + Retrieval Setup
# ────────────────────────────────────────────────────────────────────────────────
qa_chain = None

def build_llm():
    try:
        # Try different regions and models
        regions_to_try = ["us-central1", "us-east1", "us-west1", "europe-west1"]
        models_to_try = [
            "gemini-1.5-flash-001",
            "gemini-1.0-pro-001", 
            "gemini-1.0-pro",
            "chat-bison-001",
        ]
        
        for region in regions_to_try:
            for model_name in models_to_try:
                try:
                    print(f"Trying model: {model_name} in region: {region}")
                    llm = ChatVertexAI(
                        model=model_name,
                        project=PROJECT_ID,
                        location=region,
                        temperature=0.2,
                        credentials=credentials,
                    )
                    # Test with a simple message
                    test_response = llm.invoke("Hello")
                    print(f"✓ Successfully initialized {model_name} in {region}")
                    return llm
                except Exception as e:
                    print(f"✗ {model_name} in {region} failed: {str(e)[:100]}...")
                    continue
        
        print("All model/region combinations failed")
        return None
    except Exception as e:
        print(f"LLM init failed completely: {e}")
        return None

LLM = build_llm()

def chroma_present(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    if not files:
        return False
    if any(f.endswith(".sqlite3") or f == "chroma.sqlite3" for f in files):
        return True
    return any(os.path.isdir(os.path.join(path, d)) for d in files)

def get_chain():
    """Return RetrievalQA if embeddings/Chroma are ready; else None."""
    global qa_chain
    if qa_chain:
        return qa_chain
    try:
        if not LLM:
            print("No LLM; retrieval disabled.")
            return None
        if not chroma_present(CHROMA_DIR):
            print(f"No valid Chroma store in '{CHROMA_DIR}'. Using LLM-only.")
            return None

        try:
            # Try different embedding models
            embedding_models = [
                "text-embedding-004",
                "textembedding-gecko@003",
                "textembedding-gecko@001",
            ]
            
            for emb_model in embedding_models:
                try:
                    print(f"Trying embedding model: {emb_model}")
                    emb = VertexAIEmbeddings(
                        project=PROJECT_ID,
                        location=LOCATION,
                        model_name=emb_model,
                        credentials=credentials,
                    )
                    print(f"Successfully initialized embedding model: {emb_model}")
                    break
                except Exception as e:
                    print(f"Embedding model {emb_model} failed: {e}")
                    continue
            else:
                raise Exception("All embedding models failed")
        except Exception as e:
            print(f"Embeddings init failed: {e}\n{format_exc()}")
            return None
        except Exception as e:
            print(f"Embeddings init failed: {e}\n{format_exc()}")
            return None

        vectordb = Chroma(
            collection_name="knowledge_base",
            embedding_function=emb,
            persist_directory=CHROMA_DIR,
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
        )
        print("RetrievalQA ready.")
        return qa_chain
    except Exception as e:
        print(f"Retrieval init failed: {e}\n{format_exc()}")
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Helpers & Mock
# ────────────────────────────────────────────────────────────────────────────────
def get_mock_response(question: str) -> Dict[str, Any]:
    q = question.lower()
    if "jollof" in q or "rice" in q:
        return {"result": "Jollof Rice is Nigeria's most famous dish—made with tomato-pepper base, onions, spices, and rice. Cook rice with tomato sauce, peppers, onions, and stock until fluffy and flavorful.", "source_documents": []}
    if "egusi" in q:
        return {"result": "Egusi soup uses ground melon seeds, palm oil, peppers and leafy greens like spinach or bitter leaf. Great with pounded yam, eba, or rice.", "source_documents": []}
    if "pepper soup" in q:
        return {"result": "Nigerian pepper soup is a spicy broth made with meat/fish, pepper soup spice blend, scotch bonnet, ginger, and garlic. Perfect for cold days.", "source_documents": []}
    if "suya" in q:
        return {"result": "Suya is grilled spiced meat (usually beef) coated in groundnut-based spice mix called yaji, served with onions and tomatoes.", "source_documents": []}
    if any(word in q for word in ["ingredient", "cook", "recipe", "make", "how"]):
        return {"result": "I can help with Nigerian recipes! Popular dishes include Jollof Rice, Egusi soup, Pepper soup, Suya, Efo Riro, Amala, Pounded Yam, and many more. What would you like to cook?", "source_documents": []}
    return {"result": "I specialize in Nigerian cuisine! Ask me about popular dishes like Jollof Rice, Egusi, Pepper Soup, Suya, or share ingredients you have and I'll suggest recipes.", "source_documents": []}

def llm_only_answer(question: str) -> str:
    if not LLM:
        return get_mock_response(question)["result"]
    try:
        prompt = (
            "You are a friendly Nigerian food assistant. Be concise and practical.\n\n"
            f"Question: {question}\nAnswer:"
        )
        msg = LLM.invoke(prompt)
        return getattr(msg, "content", msg) or get_mock_response(question)["result"]
    except Exception as e:
        print(f"[llm_only] Error: {e}\n{format_exc()}")
        return get_mock_response(question)["result"]

class AskRequest(BaseModel):
    question: str

# ────────────────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Nigerian Food Recommender API is running!"}

@app.get("/health")
async def health_check():
    llm_status = "not_initialized"
    if LLM:
        llm_status = "initialized"
        
    return {
        "status": "ok",
        "project": PROJECT_ID,
        "location": LOCATION,
        "chroma_present": chroma_present(CHROMA_DIR),
        "allowed_origins": ["*"],  
        "google_creds_set": bool(credentials),
        "credentials_type": type(credentials).__name__ if credentials else "None",
        "llm_status": llm_status,
        "apis_to_enable": [
            "Vertex AI API",
            "AI Platform API", 
            "Cloud Resource Manager API"
        ]
    }

# Add explicit OPTIONS handler for debugging
@app.options("/ask")
async def ask_options():
    return {"message": "CORS preflight OK"}

@app.get("/ask")
async def ask_get_hint():
    return {"hint": "Use POST /ask with JSON body: {\"question\": \"...\"}"}

@app.post("/ask")
async def ask(request: AskRequest) -> Dict[str, Any]:
    try:
        chain = get_chain()
        if chain:
            out = chain.invoke({"query": request.question})
            answer = out.get("result") or out.get("answer") or ""
            sources = []
            for doc in (out.get("source_documents") or []):
                sources.append({
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("row"),
                    "snippet": (doc.page_content[:240] + "...") if len(doc.page_content) > 240 else doc.page_content,
                })
            if not answer:
                answer = llm_only_answer(request.question)
            return {"answer": answer, "sources": sources}

        # Fallback: LLM only
        return {"answer": llm_only_answer(request.question), "sources": []}

    except Exception as e:
        print(f"[/ask] Error: {e}\n{format_exc()}")
        fallback = get_mock_response(request.question)
        return {"answer": fallback["result"], "sources": []}