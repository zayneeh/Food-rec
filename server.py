import os
from traceback import format_exc
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.requests import Request

# LLM: Hugging Face hosted open-source models
from langchain_huggingface import HuggingFaceEndpoint
# Embeddings & Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ──────────────────────────────────────────
# Config (env-driven)
# ──────────────────────────────────────────
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")  # keep in repo on Render free
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "hf")   # "hf" or "ollama" (embeddings only)

# HF Inference API (open-source models)
HF_TOKEN   = os.environ.get("HF_TOKEN", "")  # required
HF_MODEL   = os.environ.get("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # pick any OSS chat model
HF_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
HF_MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", "512"))

# Embedding models
HF_EMBED_MODEL = os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Retrieval guardrail
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.35"))
NOT_FOUND_TEXT = "This item is not in our database."

app = FastAPI(title="Nigerian Food Recommender API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Logs
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
        if not os.getenv("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN is missing; set it in Render env.")
        llm = HuggingFaceEndpoint(
            repo_id=HF_MODEL,
            task="text-generation",
            temperature=HF_TEMPERATURE,
            max_new_tokens=HF_MAX_NEW_TOKENS,
        )
        _ = llm.invoke("ping")
        print(f"LLM ready: {HF_MODEL}")
        return llm
    except Exception as e:
        print(f"LLM init failed: {e}\n{format_exc()}")
        return None


def build_embeddings():
    try:
        if EMBED_BACKEND.lower() == "ollama":
            # only if you run an external Ollama for embeddings
            return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    except Exception as e:
        print(f"Embeddings init failed: {e}\n{format_exc()}")
        return None

def chroma_present(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))

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
# Retrieval helpers (strict, no fabrication)
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
    return (
        "You are a precise Nigerian food assistant.\n"
        "Use ONLY the context below to answer. Do not invent recipes or facts.\n"
        f"If the answer isn't fully supported by the context, reply exactly: {NOT_FOUND_TEXT}\n\n"
        "=== CONTEXT START ===\n"
        f"{context}\n"
        "=== CONTEXT END ===\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

def retrieve_with_scores(query: str, k: int = 4) -> List[Tuple[Any, float]]:
    if not VDB:
        return []
    try:
        # preferred: relevance in [0..1]
        return VDB.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        try:
            docs_dist = VDB.similarity_search_with_score(query, k=k)
            return [(doc, max(0.0, 1.0 - float(dist))) for doc, dist in docs_dist]
        except Exception as e:
            print(f"Retrieval error: {e}\n{format_exc()}")
            return []

def answer_from_context(question: str) -> Dict[str, Any]:
    if not (LLM and VDB):
        return {"answer": NOT_FOUND_TEXT, "sources": []}

    pairs = retrieve_with_scores(question, k=4)
    kept = [(d, s) for (d, s) in pairs if s is not None and s >= RELEVANCE_THRESHOLD]
    if not kept:
        return {"answer": NOT_FOUND_TEXT, "sources": []}

    docs = [d for d, _ in kept]
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = build_prompt(context, question)

    try:
        content = LLM.invoke(prompt) or ""
        if not content.strip() or NOT_FOUND_TEXT.lower() in content.lower():
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
        "hf_model": HF_MODEL,
        "relevance_threshold": RELEVANCE_THRESHOLD,
    }

@app.options("/ask")
async def ask_options():
    return {"message": "CORS preflight OK"}

@app.get("/ask")
async def ask_get_hint():
    return {"hint": "Use POST /ask with JSON body: {\"question\": \"...\"}"}

@app.post("/ask")
async def ask(request: AskRequest) -> Dict[str, Any]:
    try:
        return answer_from_context(request.question)
    except Exception as e:
        print(f"[/ask] Critical error: {e}\n{format_exc()}")
        return {"answer": NOT_FOUND_TEXT, "sources": []}
