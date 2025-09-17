import os
from traceback import format_exc
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.requests import Request

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
CHROMA_DIR         = os.environ.get("CHROMA_DIR", "chroma_db")
OLLAMA_BASE_URL    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL       = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct")
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))

EMBED_BACKEND      = os.environ.get("EMBED_BACKEND", "hf")  # "hf" or "ollama"
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
HF_EMBED_MODEL     = os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval guardrail (0.0–1.0). Higher = stricter.
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.35"))

NOT_FOUND_TEXT = "This item is not in our database."

app = FastAPI(title="Nigerian Food Recommender API")

# ──────────  CORS  ──────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Log every request
@app.middleware("http")
async def log_requests(request: Request, call_next):
    resp = await call_next(request)
    print(f"{request.method} {request.url.path} -> {resp.status_code}")
    return resp

# ──────────────────────────────────────────
# LLM + Vector Store
# ──────────────────────────────────────────
def build_llm():
    try:
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=OLLAMA_TEMPERATURE,
        )
    except Exception as e:
        print(f"LLM init failed: {e}\n{format_exc()}")
        return None

def build_embeddings():
    try:
        if EMBED_BACKEND.lower() == "ollama":
            return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    except Exception as e:
        print(f"Embeddings init failed: {e}\n{format_exc()}")
        return None

def chroma_present(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return any(os.scandir(path))

LLM = build_llm()
EMB = build_embeddings()
VDB = None
if EMB and chroma_present(CHROMA_DIR):
    try:
        VDB = Chroma(
            collection_name="knowledge_base",
            embedding_function=EMB,
            persist_directory=CHROMA_DIR,
        )
        print("Chroma loaded.")
    except Exception as e:
        print(f"Chroma load failed: {e}\n{format_exc()}")

# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────
def format_sources(docs) -> List[Dict[str, Any]]:
    out = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        txt = getattr(d, "page_content", "")
        out.append({
            "source": meta.get("source", ""),
            "page": meta.get("row"),
            "snippet": (txt[:240] + "...") if len(txt) > 240 else txt,
        })
    return out

def build_prompt(context: str, question: str) -> str:
    # Hard rule: ONLY use the context. If insufficient, say NOT_FOUND_TEXT.
    return (
        "You are a precise Nigerian food assistant.\n"
        "Use ONLY the context below to answer. Do not invent or guess.\n"
        f"If the answer is not fully supported by the context, reply exactly: {NOT_FOUND_TEXT}\n\n"
        "=== CONTEXT START ===\n"
        f"{context}\n"
        "=== CONTEXT END ===\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

def retrieve_with_scores(query: str, k: int = 4) -> List[Tuple[Any, float]]:
    """
    Returns list of (Document, score) with score in [0..1], where 1 is best.
    Uses LangChain's relevance scoring wrapper if available; otherwise converts
    Chroma distance to a rough relevance value (1 - distance).
    """
    docs_scores = []
    try:
        # Preferred: relevance scores (higher is better)
        docs_scores = VDB.similarity_search_with_relevance_scores(query, k=k)
        # Already (doc, score) with score ~ cosine_sim in [0..1]
        return docs_scores
    except Exception:
        try:
            # Fallback: (doc, distance) where lower is better → convert to relevance
            docs_dist = VDB.similarity_search_with_score(query, k=k)
            return [(doc, max(0.0, 1.0 - float(dist))) for doc, dist in docs_dist]
        except Exception as e:
            print(f"Retrieval error: {e}\n{format_exc()}")
            return []

def answer_from_context(question: str) -> Dict[str, Any]:
    """
    Strict RAG:
    - Retrieve top-k.
    - If no doc above threshold -> NOT_FOUND_TEXT.
    - Otherwise pass ONLY those docs to the LLM with a 'no fabrication' instruction.
    """
    if not (LLM and VDB):
        # No LLM or no vector DB -> never fabricate
        return {"answer": NOT_FOUND_TEXT, "sources": []}

    pairs = retrieve_with_scores(question, k=4)
    if not pairs:
        return {"answer": NOT_FOUND_TEXT, "sources": []}

    # Filter by relevance
    kept = [(d, s) for (d, s) in pairs if s is not None and s >= RELEVANCE_THRESHOLD]
    if not kept:
        return {"answer": NOT_FOUND_TEXT, "sources": []}

    docs = [d for d, _ in kept]
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = build_prompt(context, question)

    try:
        msg = LLM.invoke(prompt)
        content = getattr(msg, "content", msg) or ""
        # If the LLM still tries to answer without context, enforce guardrail:
        if NOT_FOUND_TEXT.lower() in content.lower():
            return {"answer": NOT_FOUND_TEXT, "sources": format_sources(docs)}
        # Light sanity check: if the model ignored instructions and output is empty
        if not content.strip():
            return {"answer": NOT_FOUND_TEXT, "sources": format_sources(docs)}
        return {"answer": content.strip(), "sources": format_sources(docs)}
    except Exception as e:
        print(f"LLM invoke error: {e}\n{format_exc()}")
        return {"answer": NOT_FOUND_TEXT, "sources": format_sources(docs)}

# ──────────────────────────────────────────
# API
# ──────────────────────────────────────────
class AskRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Nigerian Food Recommender API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "chroma_present": chroma_present(CHROMA_DIR),
        "llm_initialized": bool(LLM),
        "embed_backend": EMBED_BACKEND,
        "ollama_model": OLLAMA_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "relevance_threshold": RELEVANCE_THRESHOLD,
    }

# Preflight (CORS)
@app.options("/ask")
async def ask_options():
    return {"message": "CORS preflight OK"}

@app.get("/ask")
async def ask_get_hint():
    return {"hint": "Use POST /ask with JSON body: {\"question\": \"...\"}"}

@app.post("/ask")
async def ask(request: AskRequest) -> Dict[str, Any]:
    try:
        result = answer_from_context(request.question)
        return result
    except Exception as e:
        print(f"[/ask] Error: {e}\n{format_exc()}")
        return {"answer": NOT_FOUND_TEXT, "sources": []}
