# build_index.py  — open-source embeddings (no Vertex)
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Choose ONE of these backends:
USE_BACKEND = os.environ.get("EMBED_BACKEND", "hf")  # "hf" or "ollama"

if USE_BACKEND.lower() == "ollama":
    from langchain_community.embeddings import OllamaEmbeddings
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    def make_embeddings():
        print(f"Using Ollama embeddings: {EMBED_MODEL} @ {OLLAMA_BASE_URL}")
        return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
else:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    EMBED_MODEL = os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    def make_embeddings():
        print(f"Using HF embeddings: {EMBED_MODEL}")
        return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

load_dotenv()

DATA_CSV   = os.environ.get("DATA_CSV", str(Path("data") / "Nigerian meals.csv"))
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
COLLECTION = os.environ.get("CHROMA_COLLECTION", "knowledge_base")

loader = CSVLoader(file_path=DATA_CSV, encoding="utf-8-sig")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

for d in chunks:
    src = d.metadata.get("source", DATA_CSV)
    d.metadata["source"] = str(src)


emb = make_embeddings()
vs = Chroma.from_documents(
    documents=chunks,
    embedding=emb,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION,
)
vs.persist()
print(f"Index built to '{CHROMA_DIR}' using embeddings: {EMBED_MODEL} (backend={USE_BACKEND})")
