import os
from traceback import format_exc
from typing import Dict, Any, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


# ────────────────────────────────────────────────────────────────────────────────
# ENV / CONFIG
# ────────────────────────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GCP_PROJECT")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")

# Comma-separated origins (edit in your Render env if needed)
ALLOWED_ORIGINS_ENV = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://zeesfoodarchive.netlify.app,http://localhost:5173,http://localhost:3000",
)
ALLOWED_ORIGINS: List[str] = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()]

# ────────────────────────────────────────────────────────────────────────────────
# APP + CORS
# ────────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Nigerian Food Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,     # explicit origins
    allow_credentials=False,           # keep False when using wildcard-y headers
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ────────────────────────────────────────────────────────────────────────────────
qa_chain = None  # retrieval chain (lazy)
def build_llm():
    try:
        return ChatVertexAI(
            model="gemini-1.5-flash",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0.2,
        )
    except Exception as e:
        print(f"[init] LLM init failed: {e}\n{format_exc()}")
        return None

LLM = build_llm()

# ────────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────────
def chroma_present(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    if not files:
        return False
    if any(f.endswith(".sqlite3") or f == "chroma.sqlite3" for f in files):
        return True
    # treat index subfolders as a weak positive
    return any(os.path.isdir(os.path.join(path, d)) for d in files)

def get_chain():
    global qa_chain
    if qa_chain:
        return qa_chain
    try:
        if not LLM:
            print("[init] No LLM; retrieval disabled.")
            return None
        if not chroma_present(CHROMA_DIR):
            print(f"[init] No valid Chroma store in '{CHROMA_DIR}'. Using LLM-only.")
            return None

        emb = VertexAIEmbeddings(
            project=PROJECT_ID,
            location=LOCATION,
            model_name="text-embedding-004",
        )
        vectordb = Chroma(
            collection_name="knowledge_base",
            embedding_function=emb,
            persist_directory=CHROMA_DIR,
        )
        qa = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
        )
        qa_chain = qa
        print("[init] RetrievalQA ready.")
        return qa_chain
    except Exception as e:
        print(f"[init] Retrieval init failed: {e}\n{format_exc()}")
        return None

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
# ROUTES
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
        "origins": ALLOWED_ORIGINS,
    }

# Handle preflight cleanly so the browser never errors before your code runs
@app.options("/ask")
async def ask_options():
    return {}

# Add a GET handler to avoid 405s if the frontend accidentally GETs /ask
@app.get("/ask")
async def ask_get_hint():
    return {"hint": "Use POST /ask with JSON body: {\"question\": \"...\"}"}

# Main endpoint: prefers retrieval, falls back to LLM, then to mock
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
                    "page": doc.metadata.get("row", None),
                    "snippet": (doc.page_content[:240] + "...") if len(doc.page_content) > 240 else doc.page_content
                })
            if not answer:
                answer = llm_only_answer(request.question)
            return {"answer": answer, "sources": sources}
        # no chain → LLM-only
        return {"answer": llm_only_answer(request.question), "sources": []}
    except Exception as e:
        print(f"[/ask] Error: {e}\n{format_exc()}")
        fallback = get_mock_response(request.question)
        return {"answer": fallback["result"], "sources": []}
