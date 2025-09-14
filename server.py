# server.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION   = os.environ.get("GCP_LOCATION", "us-central1")
CHROMA_DIR = "chroma_db"

# --- CORS: allow your Netlify domain ---
NETLIFY_ORIGIN = os.environ.get("NETLIFY_ORIGIN", "*")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[NETLIFY_ORIGIN] if NETLIFY_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lazy globals so we don't rebuild each request ---
qa_chain = None

def get_chain():
    global qa_chain
    if qa_chain:
        return qa_chain

    # Embeddings + Vector DB (Chroma persisted)
    emb = VertexAIEmbeddings(
        project=PROJECT_ID,
        location=LOCATION,
        model_name="text-embedding-004"
    )
    vectordb = Chroma(
        collection_name="knowledge_base",  # FIXED: Match build_index.py
        embedding_function=emb,
        persist_directory=CHROMA_DIR,
    )

    # LLM (Gemini 1.5 Flash is fast; use Pro for higher quality)
    llm = ChatVertexAI(
        model="gemini-1.5-flash",
        project=PROJECT_ID,
        location=LOCATION,
        temperature=0.2,
    )

    # RetrievalQA (stuff) – exactly your logic
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )
    return qa_chain

class AskReq(BaseModel):
    question: str

@app.post("/ask")
def ask(req: AskReq):
    chain = get_chain()
    result = chain.invoke({"query": req.question})
    answer = result.get("result", "")
    sources = []
    for d in result.get("source_documents", []):
        sources.append({
            "source": d.metadata.get("source", ""),
            "page": d.metadata.get("row", None),
            "snippet": d.page_content[:240]
        })
    return {"answer": answer, "sources": sources}

@app.get("/")
def root():
    return {"message": "Nigerian Food Recommender API is running!"}