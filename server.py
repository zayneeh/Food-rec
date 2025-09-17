import os
from traceback import format_exc
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.requests import Request

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

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

app = FastAPI(title="Nigerian Food Recommender API")

# ──────────  ✅  CORS (from working server) ──────────
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
# LLM + Retrieval Setup
# ──────────────────────────────────────────
qa_chain = None

def build_llm():
    try:
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=OLLAMA_TEMPERATURE,
        )
        return llm
    except Exception as e:
        print(f"LLM init failed: {e}\n{format_exc()}")
        return None

LLM = build_llm()

def chroma_present(path: str) -> bool:
    import os
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    return bool(files)

def build_embeddings():
    try:
        if EMBED_BACKEND.lower() == "ollama":
            return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        else:
            return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    except Exception as e:
        print(f"Embeddings init failed: {e}\n{format_exc()}")
        return None

def get_chain():
    global qa_chain
    if qa_chain:
        return qa_chain
    try:
        if not LLM or not chroma_present(CHROMA_DIR):
            return None
        emb = build_embeddings()
        if not emb:
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
        return qa_chain
    except Exception as e:
        print(f"Retrieval init failed: {e}\n{format_exc()}")
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

# ──────────────────────────────────────────
# Routes
# ──────────────────────────────────────────
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
    }

# ✅ Explicit OPTIONS route for preflight
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
        return {"answer": llm_only_answer(request.question), "sources": []}
    except Exception as e:
        print(f"[/ask] Error: {e}\n{format_exc()}")
        fallback = get_mock_response(request.question)
        return {"answer": fallback["result"], "sources": []}
