import os
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# ---- Vertex AI auth/region ----
load_dotenv()
PROJECT_ID = os.environ["GCP_PROJECT_ID"]          
LOCATION   = os.environ.get("GCP_LOCATION", "us-central1")
SA_KEY     = os.environ["GCP_SA_KEY"]

# ---- Load CSV and split ----
loader = CSVLoader(
    file_path=r"data\Nigerian meals.csv",  
    encoding="utf-8-sig"                    
)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# ---- Vertex AI embeddings ----
emb = VertexAIEmbeddings(project=PROJECT_ID, location=LOCATION, model_name="text-embedding-004")

# ---- Persist to Chroma ----
CHROMA_DIR = "chroma_db"
Chroma.from_documents(chunks, emb, persist_directory=CHROMA_DIR, collection_name="knowledge_base")
print("Index built to", CHROMA_DIR)