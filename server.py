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

# Log every request (helps debug CORS / preflight)
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
        print(f"Initializing LLM with model: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=OLLAMA_TEMPERATURE,
        )
        # Test the connection
        test_response = llm.invoke("Hello")
        print(f"LLM test successful: {test_response}")
        return llm
    except Exception as e:
        print(f"LLM init failed: {e}\n{format_exc()}")
        return None

def chroma_present(path: str) -> bool:
    if not os.path.isdir(path):
        print(f"Chroma directory not found: {path}")
        return False
    files = set(os.listdir(path))
    has_files = bool(files)
    print(f"Chroma directory check - Path: {path}, Has files: {has_files}, Files: {list(files)[:5]}")
    return has_files

def build_embeddings():
    try:
        print(f"Building embeddings with backend: {EMBED_BACKEND}")
        if EMBED_BACKEND.lower() == "ollama":
            emb = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        else:
            emb = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
        print("Embeddings initialized successfully")
        return emb
    except Exception as e:
        print(f"Embeddings init failed: {e}\n{format_exc()}")
        return None

def get_chain():
    global qa_chain
    if qa_chain:
        print("Using cached QA chain")
        return qa_chain
    
    print("Building new QA chain...")
    try:
        # Check LLM first
        if not LLM:
            print("LLM not available, cannot build chain")
            return None
            
        # Check if Chroma exists
        if not chroma_present(CHROMA_DIR):
            print("Chroma database not present, cannot build retrieval chain")
            return None
            
        # Build embeddings
        emb = build_embeddings()
        if not emb:
            print("Embeddings not available, cannot build chain")
            return None
            
        print("Loading Chroma vector database...")
        vectordb = Chroma(
            collection_name="knowledge_base",
            embedding_function=emb,
            persist_directory=CHROMA_DIR,
        )
        
        # Test if vectordb has documents
        try:
            test_docs = vectordb.similarity_search("test", k=1)
            print(f"Vector database loaded, found {len(test_docs)} test documents")
        except Exception as e:
            print(f"Warning: Could not test vector database: {e}")
        
        print("Building RetrievalQA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
        )
        print("QA chain built successfully!")
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
        print("LLM not available for direct answer")
        return get_mock_response(question)["result"]
    
    try:
        print(f"Getting LLM-only answer for: {question}")
        prompt = (
            "You are a friendly Nigerian food assistant. Be concise and practical. "
            "Answer questions about Nigerian cuisine, recipes, ingredients, and cooking methods.\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        msg = LLM.invoke(prompt)
        
        # Handle different response types
        if hasattr(msg, 'content'):
            answer = msg.content
        elif isinstance(msg, str):
            answer = msg
        else:
            answer = str(msg)
            
        if not answer or answer.strip() == "":
            print("Empty LLM response, using fallback")
            answer = get_mock_response(question)["result"]
            
        print(f"LLM answer: {answer[:100]}...")
        return answer
        
    except Exception as e:
        print(f"[llm_only] Error: {e}\n{format_exc()}")
        return get_mock_response(question)["result"]

# Initialize LLM at startup
print("=== Initializing API ===")
LLM = build_llm()

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
    chain_available = get_chain() is not None
    return {
        "status": "ok",
        "chroma_present": chroma_present(CHROMA_DIR),
        "llm_initialized": bool(LLM),
        "chain_available": chain_available,
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
    print(f"\n=== Processing question: {request.question} ===")
    
    try:
        # Try to get the retrieval chain first
        chain = get_chain()
        
        if chain:
            print("Using retrieval chain")
            try:
                out = chain.invoke({"query": request.question})
                print(f"Chain output type: {type(out)}")
                print(f"Chain output keys: {out.keys() if isinstance(out, dict) else 'Not a dict'}")
                
                # Extract answer from chain output
                answer = out.get("result") or out.get("answer") or ""
                
                # Process sources
                sources = []
                source_docs = out.get("source_documents") or []
                for doc in source_docs:
                    sources.append({
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("row"),
                        "snippet": (doc.page_content[:240] + "...") if len(doc.page_content) > 240 else doc.page_content,
                    })
                
                if answer and answer.strip():
                    print(f"Retrieval answer: {answer[:100]}...")
                    return {"answer": answer, "sources": sources}
                else:
                    print("Empty retrieval answer, falling back to LLM-only")
                    
            except Exception as e:
                print(f"Error in retrieval chain: {e}\n{format_exc()}")
        else:
            print("No retrieval chain available, using LLM-only")
        
        # Fallback to LLM-only
        llm_answer = llm_only_answer(request.question)
        return {"answer": llm_answer, "sources": []}
        
    except Exception as e:
        print(f"[/ask] Critical error: {e}\n{format_exc()}")
        fallback = get_mock_response(request.question)
        return {"answer": fallback["result"], "sources": []}