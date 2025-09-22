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

CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")

HF_TOKEN = os.environ.get("HF_TOKEN", "")  
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")  
HF_EMBED_MODEL = os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
HF_MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", "100"))

RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.35"))
NOT_FOUND_TEXT = "I don't have specific information about that in my database."


LLM_TASK_CHOSEN: str | None = None
LAST_LLM_ERROR: str | None = None

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

def build_llm():
    global LLM_TASK_CHOSEN, LAST_LLM_ERROR
    try:
        _ensure_hf_env()

        llm = HuggingFaceEndpoint(
            repo_id=HF_MODEL,
            task="text-generation",         
            temperature=HF_TEMPERATURE,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            timeout=60,
            max_retries=3,
            return_full_text=False         
        )

        LLM_TASK_CHOSEN = "text-generation"
        LAST_LLM_ERROR = None
        print(f"LLM ready: {HF_MODEL}")
        return llm

    except Exception as e:
        LAST_LLM_ERROR = f"{type(e).__name__}: {e}"
        print(f"LLM init failed: {LAST_LLM_ERROR}")
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
# Retrieval helpers
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
    # If LLM not available, provide basic fallback
    if not LLM:
        return {"answer": "I'm sorry, the assistant is temporarily unavailable.", "sources": []}

    # Get context from database
    context = ""
    sources = []
    if VDB:
        try:
            pairs = retrieve_with_scores(question, k=4)
            kept = [(d, abs(s)) for (d, s) in pairs if s is not None and abs(s) >= RELEVANCE_THRESHOLD]
            if kept:
                docs = [d for d, _ in kept]
                context = "\n\n".join(d.page_content[:400] for d in docs)
                sources = format_sources(docs)
        except Exception as e:
            print(f"Context retrieval failed: {e}")

    # Simple prompt
    if context:
        prompt = f"Context: {context}\n\nUser: {question}\nAssistant:"
    else:
        prompt = f"User: {question}\nAssistant:"
    
    # Keep it short
    if len(prompt) > 1500:
        prompt = prompt[:1500] + "..."

    try:
        response = LLM.invoke(prompt)
        
        if hasattr(response, 'content'):
            text = response.content.strip()
        elif isinstance(response, str):
            text = response.strip()
        else:
            text = str(response).strip()
            
        if not text:
            return {"answer": "I couldn't generate a response. Please try rephrasing your question.", "sources": []}
            
        # Show sources only for recipe-related responses
        if sources and (len(text) > 80 or any(word in text.lower() for word in ['ingredient', 'cook', 'recipe', 'step', 'add', 'heat', 'preparation'])):
            return {"answer": text, "sources": sources}
        else:
            return {"answer": text, "sources": []}
        
    except Exception as e:
        print(f"LLM invoke error: {e}\n{format_exc()}")
        return {"answer": "Hello! I'm here to help with Nigerian food questions. What would you like to know?", "sources": []}

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