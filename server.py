import os
from traceback import format_exc
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.requests import Request
from dotenv import load_dotenv

# Open-source LLM / embeddings / vectorstore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ────────────────────────────────────────────────────────────────────────────────
# Environment & Config
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()

CHROMA_DIR   = os.environ.get("CHROMA_DIR", "chroma_db")
HF_MODEL     = os.environ.get("HF_MODEL", "distilgpt2")  # tiny, fast default
GEN_MAX_TOK  = int(os.environ.get("GEN_MAX_TOK", "160"))
GEN_TEMP     = float(os.environ.get("GEN_TEMP", "0.7"))
ENABLE_LLM   = os.environ.get("ENABLE_LLM", "1") == "1"  # allow hard-off
EMB_MODEL    = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="Nigerian Food Recommender API (Open-Source)")

# CORS: permissive; safe because allow_credentials=False
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Simple request logger
@app.middleware("http")
async def log_requests(request: Request, call_next):
    resp = await call_next(request)
    print(f"{request.method} {request.url.path} -> {resp.status_code}")
    return resp

# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────
def chroma_present(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    if not files:
        return False
    if any(f.endswith(".sqlite3") or f == "chroma.sqlite3" for f in files):
        return True
    return any(os.path.isdir(os.path.join(path, d)) for d in files)

# ────────────────────────────────────────────────────────────────────────────────
# LLM (open-source) + Retrieval
# ────────────────────────────────────────────────────────────────────────────────
LLM = None
qa_chain = None
SELECTED_MODEL = None

def build_llm():
    """Create a small HF text-gen pipeline; fall back gracefully."""
    if not ENABLE_LLM:
        print("ENABLE_LLM=0 → LLM disabled (mock mode).")
        return None
    try:
        print(f"Loading HF model: {HF_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL)
        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=GEN_MAX_TOK,
            temperature=GEN_TEMP,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            # device is auto (CPU). If you have GPU on your host, set device=0
        )
        llm = HuggingFacePipeline(pipeline=gen)
        global SELECTED_MODEL
        SELECTED_MODEL = HF_MODEL
        print(f"✓ HF model ready: {HF_MODEL}")
        return llm
    except Exception as e:
        print(f"Open-source LLM init failed: {e}\n{format_exc()}")
        return None

def get_chain():
    """Return RetrievalQA if LLM & Chroma are available; else None."""
    global qa_chain, LLM
    if qa_chain:
        return qa_chain
    if LLM is None:
        LLM = build_llm()
    if not LLM:
        print("No LLM; retrieval disabled (mock/LLM-off).")
        return None
    if not chroma_present(CHROMA_DIR):
        print(f"No valid Chroma store in '{CHROMA_DIR}'. Using LLM-only.")
        return None
    try:
        print(f"Loading embeddings: {EMB_MODEL}")
        emb = HuggingFaceEmbeddings(model_name=EMB_MODEL, model_kwargs={"device": "cpu"})
        vectordb = Chroma(
            collection_name="knowledge_base",
            embedding_function=emb,
            persist_directory=CHROMA_DIR,
        )
        chain = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
        )
        print("RetrievalQA ready.")
        qa_chain = chain
        return qa_chain
    except Exception as e:
        print(f"Retrieval init failed: {e}\n{format_exc()}")
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Fallback answers (fast, no model required)
# ────────────────────────────────────────────────────────────────────────────────
def fallback_answer(question: str) -> str:
    q = (question or "").lower()
    if "jollof" in q or "rice" in q:
        return ("Jollof Rice: tomato-pepper base with onions & spices. "
                "Fry paste, cook blended tomato till oil separates, add spices, "
                "stir in rice + stock, steam till fluffy; serve with plantain.")
    if "egusi" in q:
        return ("Egusi soup: ground melon seeds with palm oil, peppers, stock, iru, and greens. "
                "Toast egusi lightly, add stock gradually, add meat/fish, finish with greens. "
                "Great with pounded yam or eba.")
    if "suya" in q:
        return ("Suya: thin beef strips coated in yaji (peanut-ginger-garlic-chili mix), "
                "grilled hot; serve with onions & tomatoes.")
    if "pepper soup" in q:
        return ("Pepper Soup: broth with uda, uziza seed, calabash nutmeg, chiles & scent leaves. "
                "Simmer meat/fish till tender; add spices; finish with scent leaves.")
    return ("I can help with Nigerian recipes—Jollof, Egusi, Efo Riro, Suya, Pepper Soup. "
            "Tell me a dish or list your ingredients, and I’ll suggest options.")

def llm_only_answer(question: str) -> str:
    """Use HF LLM when available; otherwise fallback."""
    global LLM
    if LLM is None:
        LLM = build_llm()
    if not LLM:
        return fallback_answer(question)
    try:
        prompt = (
            "You are a friendly Nigerian food assistant. Be concise and practical.\n\n"
            f"Question: {question}\nAnswer:"
        )
        msg = LLM.invoke(prompt)
        return getattr(msg, "content", msg) or fallback_answer(question)
    except Exception as e:
        print(f"[llm_only] Error: {e}\n{format_exc()}")
        return fallback_answer(question)

# ────────────────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Nigerian Food Recommender API is running (open-source)."}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": "oss",
        "hf_model": SELECTED_MODEL or HF_MODEL,
        "embeddings": EMB_MODEL,
        "chroma_present": chroma_present(CHROMA_DIR),
        "cors": {"allow_origins": "*", "allow_credentials": False},
        "llm_enabled": ENABLE_LLM,
        "llm_initialized": bool(LLM),
    }

@app.get("/ask")
async def ask_hint():
    return {"hint": "POST /ask with JSON: {\"question\": \"...\"}"}

@app.post("/ask")
async def ask(req: AskRequest) -> Dict[str, Any]:
    try:
        chain = get_chain()
        if chain:
            out = chain.invoke({"query": req.question})
            answer = out.get("result") or out.get("answer") or ""
            sources = []
            for doc in (out.get("source_documents") or []):
                sources.append({
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("row"),
                    "snippet": (doc.page_content[:240] + "...") if len(doc.page_content) > 240 else doc.page_content,
                })
            if not answer:
                answer = llm_only_answer(req.question)
            return {"answer": answer, "sources": sources}

        # Fallbacks: LLM-only → static fallback
        return {"answer": llm_only_answer(req.question), "sources": []}

    except Exception as e:
        print(f"[/ask] Error: {e}\n{format_exc()}")
        return {
            "answer": ("I'm having technical difficulties, but I can still help with Nigerian recipes. "
                       "Ask about Jollof, Egusi, Suya, Pepper Soup, or share your ingredients."),
            "sources": [],
        }