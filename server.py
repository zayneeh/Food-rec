import os
from traceback import format_exc
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from starlette.requests import Request

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
SA_KEY     = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")  # optional

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
        return ChatVertexAI(
            model="gemini-1.5-flash",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0.2,
        )
    except Exception as e:
        print(f"LLM init failed: {e}\n{format_exc()}")
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
            emb = VertexAIEmbeddings(
                project=PROJECT_ID,
                location=LOCATION,
                model_name="text-embedding-004",
            )
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
    if "jollof" in q:
        return {"result": "Jollof Rice is Nigeria's most famous dish—tomato-pepper base with onions and spices.", "source_documents": []}
    if "egusi" in q:
        return {"result": "Egusi soup uses ground melon seeds, palm oil, peppers and greens; great with pounded yam.", "source_documents": []}
    return {"result": "I can help with Nigerian recipes! Ask about Jollof, Egusi, Efo Riro, or share ingredients.", "source_documents": []}

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
    return {
        "status": "ok",
        "project": PROJECT_ID,
        "location": LOCATION,
        "chroma_present": chroma_present(CHROMA_DIR),
        "google_creds_set": bool(SA_KEY),
        "llm_initialized": bool(LLM),
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