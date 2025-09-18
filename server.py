import os
from traceback import format_exc
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.requests import Request

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

# ──────────────────────────────────────────
# Config (env)
# ──────────────────────────────────────────
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")

HF_TOKEN = os.environ.get("HF_TOKEN", "")  # REQUIRED
HF_MODEL = os.environ.get("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
HF_EMBED_MODEL = os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
HF_MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", "512"))
HF_TASK = os.environ.get("HF_TASK", "text-generation")  # requested task

RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.35"))
NOT_FOUND_TEXT = "This item is not in our database."

# Track actual task chosen + last init error (visible in /health)
LLM_TASK_CHOSEN: str | None = None
LAST_LLM_ERROR: str | None = None

# ──────────────────────────────────────────
# FastAPI + CORS  (UNCHANGED)
# ──────────────────────────────────────────
app = FastAPI(title="Nigerian Food Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    resp = await call_next(request)
    print(f"{request.method} {request.url.path} -> {resp.status_code}")
    return resp

# ──────────────────────────────────────────
# Builders
# ──────────────────────────────────────────
def chroma_present(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))

def _ensure_hf_env():
    """Expose the token the way HF libs expect."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN env var is missing.")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

def _try_make_llm(task: str):
    """Create an endpoint client for a given task."""
    return HuggingFaceEndpoint(
        repo_id=HF_MODEL,
        task=task,
        temperature=HF_TEMPERATURE,
        max_new_tokens=HF_MAX_NEW_TOKENS,
    )

# Add this at the top of server.py after imports

# Model-specific task mappings
MODEL_TASK_MAP = {
    "meta-llama/Llama-3.2-3B-Instruct": ["conversational"],
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": ["text-generation", "conversational"],
    # Add more models as needed
}

def get_supported_tasks(model_id: str) -> List[str]:
    """Get supported tasks for a specific model."""
    return MODEL_TASK_MAP.get(model_id, ["conversational", "text-generation"])

# Replace just the build_llm function with this:

def build_llm():
    """Try supported tasks for the specific model."""
    global LLM_TASK_CHOSEN, LAST_LLM_ERROR
    try:
        _ensure_hf_env()

        # Simple fix: just try conversational first for Llama models
        if "llama" in HF_MODEL.lower():
            candidates = ["conversational", "text-generation"]
        else:
            candidates = ["text-generation", "conversational"]

        print(f"[LLM] Will try tasks in order: {candidates}")

        for task in candidates:
            try:
                print(f"[LLM] Trying task={task}")
                llm = _try_make_llm(task)
                
                # Simple test - don't invoke, just create
                LLM_TASK_CHOSEN = task
                LAST_LLM_ERROR = None
                print(f"LLM ready: {HF_MODEL} (task={task})")
                return llm
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                LAST_LLM_ERROR = error_msg
                print(f"[LLM] Task {task} failed: {error_msg}")
                continue

        print("All tasks failed for this model.")
        return None

    except Exception as e:
        LAST_LLM_ERROR = f"{type(e).__name__}: {e}"
        print(f"LLM init failed (fatal): {LAST_LLM_ERROR}\n{format_exc()}")
        return None
def build_embeddings():
    try:
        _ensure_hf_env()
        emb = HuggingFaceEndpointEmbeddings(model=HF_EMBED_MODEL)
        print(f"Embeddings ready: {HF_EMBED_MODEL}")
        return emb
    except Exception as e:
        print(f"Embeddings init failed: {e}\n{format_exc()}")
        return None

# Build once at startup
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
        return VDB.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        try:
            docs_dist = VDB.similarity_search_with_score(query, k=k)
            return [(doc, max(0.0, 1.0 - float(dist))) for doc, dist in docs_dist]
        except Exception as e:
            print(f"Retrieval error: {e}\n{format_exc()}")
            return []

def answer_from_context(question: str) -> Dict[str, Any]:
    # If either LLM or VDB is unavailable, never fabricate
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
        text = (getattr(content, "content", content) or "").strip()
        if not text or NOT_FOUND_TEXT.lower() in text.lower():
            return {"answer": NOT_FOUND_TEXT, "sources": format_sources(docs)}
        return {"answer": text, "sources": format_sources(docs)}
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
        "hf_model": HF_MODEL,
        "hf_embed_model": HF_EMBED_MODEL,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "hf_task_requested": HF_TASK,
        "hf_task_chosen": LLM_TASK_CHOSEN,
        "last_llm_error": LAST_LLM_ERROR,
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
