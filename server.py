import os
from traceback import format_exc
from typing import Dict, Any, List, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.requests import Request
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_huggingface import ChatHuggingFace
import warnings

CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")

HF_TOKEN = os.environ.get("HF_TOKEN", "")  
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")  
HF_EMBED_MODEL = os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
HF_MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", "256"))

RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.35"))
NOT_FOUND_TEXT = "I don't have specific information about that in my database."

LLM_TASK_CHOSEN: str | None = None
LAST_LLM_ERROR: str | None = None

app = FastAPI(title="Nigerian Food Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    resp = await call_next(request)
    print(f"{request.method} {request.url.path} -> {resp.status_code}")
    return resp

# ──────────────────────────────────────────
# Builders
# ──────────────────────────────────────────
def chroma_present(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))

def _ensure_hf_env():
    """Expose the token the way HF libs expect."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN env var is missing.")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

def build_llm():
    global LLM_TASK_CHOSEN, LAST_LLM_ERROR
    try:
        _ensure_hf_env()
        
        # Try different approaches for Qwen
        print(f"Attempting to initialize {HF_MODEL}...")
        
        # Method 1: Try with text-generation task explicitly
        try:
            base_llm = HuggingFaceEndpoint(
                repo_id=HF_MODEL,
                task="text-generation",  # Explicit text-generation for Qwen
                temperature=HF_TEMPERATURE,
                max_new_tokens=HF_MAX_NEW_TOKENS,
                timeout=60,
                model_kwargs={
                    "max_length": HF_MAX_NEW_TOKENS,
                    "temperature": HF_TEMPERATURE,
                }
            )
            
            # Test the base LLM first
            test_response = base_llm.invoke("Hello")
            print(f"Base LLM test successful: {test_response[:50]}...")
            
            # Wrap with ChatHuggingFace
            llm = ChatHuggingFace(llm=base_llm)
            
            LLM_TASK_CHOSEN = "text-generation + chat wrapper"
            LAST_LLM_ERROR = None
            print(f"LLM ready: {HF_MODEL} with text-generation task")
            return llm
            
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            
            # Method 2: Try without explicit task
            try:
                base_llm = HuggingFaceEndpoint(
                    repo_id=HF_MODEL,
                    temperature=HF_TEMPERATURE,
                    max_new_tokens=HF_MAX_NEW_TOKENS,
                    timeout=60,
                )
                
                llm = ChatHuggingFace(llm=base_llm)
                
                LLM_TASK_CHOSEN = "auto-detect + chat wrapper"
                LAST_LLM_ERROR = None
                print(f"LLM ready: {HF_MODEL} with auto-detected task")
                return llm
                
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
                
                # Method 3: Try just the base endpoint without chat wrapper
                try:
                    llm = HuggingFaceEndpoint(
                        repo_id=HF_MODEL,
                        task="text-generation",
                        temperature=HF_TEMPERATURE,
                        max_new_tokens=HF_MAX_NEW_TOKENS,
                        timeout=60,
                    )
                    
                    LLM_TASK_CHOSEN = "text-generation only"
                    LAST_LLM_ERROR = None
                    print(f"LLM ready: {HF_MODEL} with text-generation (no chat wrapper)")
                    return llm
                    
                except Exception as e3:
                    raise Exception(f"All methods failed. Method 1: {e1}, Method 2: {e2}, Method 3: {e3}")

    except Exception as e:
        LAST_LLM_ERROR = f"{type(e).__name__}: {e}"
        print(f"LLM init failed: {LAST_LLM_ERROR}")
        print(f"Full traceback: {format_exc()}")
        return None
    
def build_embeddings():
    try:
        _ensure_hf_env()
        emb = HuggingFaceEndpointEmbeddings(
            model=HF_EMBED_MODEL,
            timeout=30
        )
        print(f"Embeddings ready: {HF_EMBED_MODEL}")
        
        # Test embeddings
        test_embed = emb.embed_query("test")
        print(f"Embeddings test successful, dimension: {len(test_embed)}")
        return emb
        
    except Exception as e:
        print(f"Embeddings init failed: {e}\n{format_exc()}")
        return None

# Build once at startup
print("Initializing components...")
LLM = build_llm()
EMB = build_embeddings()
VDB = None

if EMB and chroma_present(CHROMA_DIR):
    try:
        VDB = Chroma(
            collection_name="knowledge_base",
            embedding_function=EMB,
            persist_directory=CHROMA_DIR,
        )
        # Test if the database actually works
        try:
            test_count = VDB._collection.count()
            print(f"Chroma loaded with {test_count} documents.")
        except:
            # Fallback count method
            test_docs = VDB.similarity_search("test", k=1)
            print(f"Chroma loaded successfully.")
            
    except Exception as e:
        print(f"Chroma load failed: {e}\n{format_exc()}")
        VDB = None
else:
    if not EMB:
        print("Skipping Chroma - embeddings not available")
    if not chroma_present(CHROMA_DIR):
        print(f"Skipping Chroma - directory {CHROMA_DIR} not present or empty")

# ──────────────────────────────────────────
# Retrieval helpers
# ──────────────────────────────────────────
def format_sources(docs) -> List[Dict[str, Any]]:
    out = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        txt = getattr(d, "page_content", "")
        out.append({
            "source": meta.get("source", ""),
            "page": meta.get("row"),
            "snippet": (txt[:240] + "...") if len(txt) > 240 else txt,
        })
    return out

def retrieve_with_scores(query: str, k: int = 4) -> List[Tuple[Any, float]]:
    if not VDB:
        return []
    try:
        results = VDB.similarity_search_with_relevance_scores(query, k=k)
        print(f"Retrieved {len(results)} documents with scores: {[score for _, score in results]}")
        return results
        
    except Exception as e:
        print(f"Primary retrieval failed: {e}")
        try:
            # Fallback method
            docs_dist = VDB.similarity_search_with_score(query, k=k)
            results = []
            for doc, dist in docs_dist:
                # Convert distance to similarity score
                similarity_score = max(0.0, min(1.0, 1.0 - abs(float(dist))))
                results.append((doc, similarity_score))
            print(f"Fallback retrieval got {len(results)} documents")
            return results
            
        except Exception as fallback_error:
            print(f"All retrieval methods failed: {fallback_error}")
            return []

def answer_from_context(question: str) -> Dict[str, Any]:
    # If LLM not available, provide basic fallback
    if not LLM:
        error_msg = f"LLM not initialized. Last error: {LAST_LLM_ERROR}"
        print(error_msg)
        return {"answer": "I'm sorry, the assistant is temporarily unavailable.", "sources": []}

    # Get context from database
    context = ""
    sources = []
    if VDB:
        try:
            pairs = retrieve_with_scores(question, k=4)
            # Filter by relevance threshold
            kept = [(d, score) for (d, score) in pairs if score is not None and score >= RELEVANCE_THRESHOLD]
            
            if kept:
                docs = [d for d, _ in kept]
                context = "\n\n".join(d.page_content[:400] for d in docs)
                sources = format_sources(docs)
                print(f"Using {len(kept)} relevant documents")
                
        except Exception as e:
            print(f"Context retrieval failed: {e}")

    # Create prompt based on LLM type
    if context:
        if LLM_TASK_CHOSEN and "chat" in LLM_TASK_CHOSEN:
            # For chat models, use a more conversational prompt
            prompt = f"You are a helpful assistant specializing in Nigerian food. Based on the following information, please answer the user's question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            # For text generation models, use a simpler prompt
            prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    else:
        if LLM_TASK_CHOSEN and "chat" in LLM_TASK_CHOSEN:
            prompt = f"You are a helpful assistant specializing in Nigerian food. Please answer this question: {question}"
        else:
            prompt = f"Question about Nigerian food: {question}\nAnswer:"
    
    # Keep prompt reasonable length
    if len(prompt) > 1500:
        prompt = prompt[:1500] + "..."

    try:
        print(f"Sending prompt to LLM (task: {LLM_TASK_CHOSEN}, length: {len(prompt)})")
        response = LLM.invoke(prompt)
        
        # Handle different response types
        if hasattr(response, 'content'):
            text = response.content.strip()
        elif isinstance(response, str):
            text = response.strip()
        else:
            text = str(response).strip()
            
        if not text:
            return {"answer": "I couldn't generate a response. Please try rephrasing your question.", "sources": []}
        
        # Clean up response if it repeats the prompt
        if "Question:" in text and "Answer:" in text:
            parts = text.split("Answer:")
            if len(parts) > 1:
                text = parts[-1].strip()
        
        print(f"Generated response: {text[:100]}...")
            
        # Return response with sources for detailed answers
        if sources and (len(text) > 80 or any(word in text.lower() for word in ['ingredient', 'cook', 'recipe', 'step', 'add', 'heat', 'preparation'])):
            return {"answer": text, "sources": sources}
        else:
            return {"answer": text, "sources": []}
        
    except Exception as e:
        error_details = f"LLM invoke error: {e}\nTask chosen: {LLM_TASK_CHOSEN}"
        print(error_details)
        print(f"Full traceback: {format_exc()}")
        
        # Fallback response
        if sources:
            return {"answer": "I found some information about this in my database. Could you please rephrase your question more specifically?", "sources": sources}
        else:
            return {"answer": "Hello! I'm here to help with Nigerian food questions. What would you like to know?", "sources": []}

# ──────────────────────────────────────────
# API
# ──────────────────────────────────────────
class AskRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Nigerian Food Recommender API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "chroma_present": chroma_present(CHROMA_DIR),
        "chroma_initialized": bool(VDB),
        "llm_initialized": bool(LLM),
        "embeddings_initialized": bool(EMB),
        "hf_model": HF_MODEL,
        "hf_embed_model": HF_EMBED_MODEL,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "llm_task_chosen": LLM_TASK_CHOSEN,
        "last_llm_error": LAST_LLM_ERROR,
    }

@app.options("/ask")
async def ask_options():
    return {"message": "CORS preflight OK"}

@app.get("/ask")
async def ask_get_hint():
    return {"hint": "Use POST /ask with JSON body: {\"question\": \"...\"}"}

@app.post("/ask")
async def ask(request: AskRequest) -> Dict[str, Any]:
    try:
        return answer_from_context(request.question)
    except Exception as e:
        print(f"[/ask] Critical error: {e}\n{format_exc()}")
        return {"answer": NOT_FOUND_TEXT, "sources": []}

