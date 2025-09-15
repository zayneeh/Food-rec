import os
from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from traceback import format_exc

# LangChain bits
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


# ────────────────────────────────────────────────────────────────────────────────
# ENV / CONFIG
# ────────────────────────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GCP_PROJECT")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")

# Comma separated list of origins. Example in Render/Netlify:
# ALLOWED_ORIGINS="https://zeesfoodarchive.netlify.app,http://localhost:5173"
ALLOWED_ORIGINS_ENV = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://zeesfoodarchive.netlify.app,http://localhost:5173,http://localhost:3000",
)
ALLOWED_ORIGINS: List[str] = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()]

# ────────────────────────────────────────────────────────────────────────────────
# FASTAPI APP + CORS
# ────────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Nigerian Food Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,     # IMPORTANT: explicit origins
    allow_credentials=False,           # keep False unless you use cookies
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],               # allow Content-Type, Authorization, etc.
    expose_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ────────────────────────────────────────────────────────────────────────────────
qa_chain = None  # retrieval chain (set lazily)

# Always-available chat LLM (used for retrieval and fallback)
def _build_llm():
    try:
        return ChatVertexAI(
            model="gemini-1.5-flash",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0.2,
        )
    except Exception as e:
        print(f"[init] LLM init failed: {e}")
        return None

LLM = _build_llm()


# ────────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ────────────────────────────────────────────────────────────────────────────────
def chroma_present(path: str) -> bool:
    """Heuristic check that a Chroma store exists."""
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    if not files:
        return False
    # typical persisted file(s)
    if any(f.endswith(".sqlite3") or f == "chroma.sqlite3" for f in files):
        return True
    # some versions keep sqlite under subfolders too; weak positive
    if any(os.path.isdir(os.path.join(path, d)) for d in files):
        return True
    return False


def get_chain():
    """Return a RetrievalQA chain if Chroma + LLM are usable; otherwise None."""
    global qa_chain
    if qa_chain:
        return qa_chain

    try:
        if not LLM:
            print("[init] No LLM available. Retrieval will be disabled.")
            return None

        if not chroma_present(CHROMA_DIR):
            print(f"[init] No valid Chroma store found in '{CHROMA_DIR}'. Using LLM-only.")
            return None

        print(f"[init] Loading Chroma from '{CHROMA_DIR}'...")
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


def get_mock_response(question: str) -> str:
    """Hard fallback that never touches Vertex/Chroma so we always answer."""
    q = question.lower()
    if "jollof" in q:
        return (
            "Jollof Rice is a tomato-pepper based rice cooked with onions, stock, "
            "bay leaf and spices. Serve with fried plantain or grilled chicken. "
            "Would you like a simple recipe?"
        )
    if "egusi" in q:
        return (
            "Egusi soup is made with ground melon seeds, palm oil, peppers, "
            "leafy greens, and stock. It’s great with pounded yam, eba or fufu."
        )
    return (
        "I can help with Nigerian dishes! Ask about Jollof, Egusi, Efo Riro, or tell me "
        "your ingredients and I’ll suggest meals."
    )


class AskRequest(BaseModel):
    question: str


def llm_only_answer(question: str) -> str:
    """Use the chat model directly (no retrieval)."""
    if not LLM:
        return get_mock_response(question)
    prompt = (
        "You are a friendly Nigerian food assistant. Be concise and practical.\n\n"
        f"Question: {question}\nAnswer:"
    )
    try:
        msg = LLM.invoke(prompt)
        return getattr(msg, "content", msg) or get_mock_response(question)
    except Exception as e:
        print(f"[llm_only] Error: {e}\n{format_exc()}")
        return get_mock_response(question)


# ────────────────────────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Nigerian Food Recommender API is running!"}


@app.get("/health")
async def health_check():
    # Basic sanity info for logs / probes
    return {
        "status": "ok",
        "project": PROJECT_ID,
        "location": LOCATION,
        "chroma_present": chroma_present(CHROMA_DIR),
        "origins": ALLOWED_ORIGINS,
    }


# Preflight helper (CORS middleware will respond, but this silences 405s)
@app.options("/ask")
async def options_ask():
    return {}


@app.post("/ask")
async def ask(request: AskRequest) -> Dict[str, Any]:
    """Answer using retrieval if available; otherwise LLM-only; otherwise mock."""
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

        # No retrieval (no DB or no LLM embeddings). Still answer.
        return {"answer": llm_only_answer(request.question), "sources": []}

    except Exception as e:
        print(f"[/ask] Error: {e}\n{format_exc()}")
        return {"answer": get_mock_response(request.question), "sources": []}


