# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # works with any OpenAI-compatible API
import os

app = FastAPI()

# --- Embeddings (open-source) ---
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")  # multilingual, strong for RAG
# Put your curated KB files here (md/txt/pdf preprocessed to text)
loader = DirectoryLoader("knowledge_base", glob="**/*.md")
docs = loader.load()
splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
store = FAISS.from_documents(splits, emb)
retriever = store.as_retriever(search_kwargs={"k": 6})

# --- LLM via vLLM (OpenAI-compatible) ---
# IMPORTANT: no OpenAI key; just point to your local base_url
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.3,
    openai_api_key="not-needed",
    openai_api_base=os.getenv("LLM_BASE_URL", "http://localhost:8001/v1")
)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

class RecoRequest(BaseModel):
    mood: str
    dietary: str | None = None
    region: str | None = None

@app.post("/api/recommend")
def recommend(body: RecoRequest):
    prompt = (
        "You are a Nigerian food guide. Suggest 3–5 authentic dishes.\n"
        f"Mood: {body.mood}\nDietary: {body.dietary or 'none'}\nRegion: {body.region or 'any'}\n"
        "Return JSON array: [{name, why, source}]. Use sources from context."
    )
    result = qa(prompt)
    return {"recommendations": result["result"]}
