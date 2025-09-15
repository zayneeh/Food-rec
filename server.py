import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- Env ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GCP_PROJECT")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
NETLIFY_ORIGIN = os.environ.get("NETLIFY_ORIGIN", "*")
CHROMA_DIR = "chroma_db"   # ensure it contains chroma.sqlite3 + index dirs

app = FastAPI(title="Nigerian Food Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[NETLIFY_ORIGIN] if NETLIFY_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globals ---
qa_chain = None

# Always-initialized LLM fallback
llm = ChatVertexAI(
    model="gemini-1.5-flash",
    project=PROJECT_ID,
    location=LOCATION,
    temperature=0.2,
)

def chroma_present(path: str) -> bool:
    """Heuristic: directory exists, not empty, and sqlite file present."""
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    if not files:
        return False
    # typical persisted file; name can vary, but this catches most cases
    has_sqlite = any(f.endswith(".sqlite3") or f == "chroma.sqlite3" for f in files)
    return has_sqlite

def get_chain():
    global qa_chain
    if qa_chain:
        return qa_chain

    try:
        if not chroma_present(CHROMA_DIR):
            print(f"[init] No valid Chroma store in '{CHROMA_DIR}'. Using LLM-only fallback.")
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
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
        )

        qa_chain = qa
        print("[init] RetrievalQA chain initialized.")
        return qa_chain

    except Exception as e:
        print(f"[init] Error initializing chain: {e}. Using LLM-only fallback.")
        return None

def get_mock_response(question: str):
    mock_data = {
        "jollof": "Jollof Rice is Nigeria's most famous dish. Made with rice, tomatoes, onions, and spices...",
        "default": "I can help with Nigerian recipes! Try asking about Jollof Rice, Egusi, or other dishes."
    }
    q = question.lower()
    for k in mock_data:
        if k != "default" and k in q:
            return mock_data[k]
    return mock_data["default"]

class AskRequest(BaseModel):
    question: str

def llm_only_answer(question: str) -> str:
    # simple prompt; tune as you like
    prompt = (
        "You are a friendly Nigerian food assistant. "
        "Answer clearly and concisely. If you don't know, say so.\n\n"
        f"Question: {question}\nAnswer:"
    )
    msg = llm.invoke(prompt)  # sync call works inside async route
    # msg can be a str or a AIMessage depending on LC version
    return getattr(msg, "content", msg)

@app.post("/ask")
async def ask(request: AskRequest):
    try:
        chain = get_chain()

        if chain is None:
            # Prefer a real LLM answer over the canned mock
            answer_text = llm_only_answer(request.question)
            return {"answer": answer_text, "sources": []}

        # Retrieval path
        out = chain.invoke({"query": request.question})
        answer = out.get("result", "") or out.get("answer", "")
        sources = []
        for doc in out.get("source_documents", []) or []:
            sources.append({
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("row", None),
                "snippet": (doc.page_content[:240] + "...") if len(doc.page_content) > 240 else doc.page_content
            })
        return {"answer": answer, "sources": sources}

    except Exception as e:
        print(f"[ask] Error: {e}")
        # Hard fallback
        try:
            return {"answer": llm_only_answer(request.question), "sources": []}
        except Exception:
            return {"answer": get_mock_response(request.question), "sources": []}

@app.get("/")
async def root():
    return {"message": "Nigerian Food Recommender API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "project": PROJECT_ID}
