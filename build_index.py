"""
build_index.py
Builds a Chroma vector database using Hugging Face Inference API embeddings.
Reads configuration from environment variables or a local .env file.
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma

# ── Load .env file if present ──
load_dotenv()

# ── Environment variables with defaults ──
HF_TOKEN       = os.getenv("HF_TOKEN", "")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_CSV       = os.getenv("DATA_CSV", "data/Nigerian meals.csv")
CHROMA_DIR     = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION     = os.getenv("CHROMA_COLLECTION", "knowledge_base")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is missing.  Set it in a .env file or your environment."
    )

# The embeddings client looks for this specific variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

print(f"[build] Loading CSV: {DATA_CSV}")
loader = CSVLoader(file_path=DATA_CSV, encoding="utf-8-sig")
docs = loader.load()

print("[build] Splitting documents…")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
for d in chunks:
    d.metadata["source"] = d.metadata.get("source", DATA_CSV)

print(f"[build] Creating HF Endpoint embeddings: {HF_EMBED_MODEL}")
emb = HuggingFaceEndpointEmbeddings(model=HF_EMBED_MODEL)

# Optional: clear old DB so we always rebuild from scratch
if os.path.isdir(CHROMA_DIR):
    print(f"[build] Removing old DB at {CHROMA_DIR}")
    shutil.rmtree(CHROMA_DIR)

print(f"[build] Writing Chroma DB to: {CHROMA_DIR}")
Chroma.from_documents(
    documents=chunks,
    embedding=emb,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION,
)

print("[build] ✅ Finished building Chroma index.")
