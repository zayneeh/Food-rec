import os
from traceback import format_exc
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from starlette.requests import Request

# Import for open source models
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ────────────────────────────────────────────────────────────────────────────────
# Environment & Config
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()

CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")

app = FastAPI(title="Nigerian Food Recommender API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────────
# Open Source LLM Setup
# ────────────────────────────────────────────────────────────────────────────────
qa_chain = None
LLM = None

def build_llm():
    """Initialize open source LLM via HuggingFace"""
    try:
        # Check if we have GPU or use CPU
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
        
        # Try different lightweight models suitable for Render
        models_to_try = [
            "microsoft/DialoGPT-medium",  # Conversational, lightweight
            "distilgpt2",                 # Very small but functional
            "gpt2",                       # Classic, reliable
        ]
        
        for model_name in models_to_try:
            try:
                print(f"Loading model: {model_name}")
                
                # Create text generation pipeline
                pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=50256,
                    device=device,
                    return_full_text=False
                )
                
                # Wrap in LangChain
                llm = HuggingFacePipeline(pipeline=pipe)
                
                # Test it
                test_response = llm("Hello")
                print(f"✓ Successfully loaded {model_name}")
                return llm
                
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {str(e)[:100]}...")
                continue
                
        print("All models failed to load")
        return None
        
    except Exception as e:
        print(f"LLM initialization failed: {e}")
        return None

def build_embeddings():
    """Initialize open source embeddings"""
    try:
        # Try different embedding models
        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",  # Small, fast
            "sentence-transformers/all-mpnet-base-v2", # Better quality
        ]
        
        for model_name in embedding_models:
            try:
                print(f"Loading embeddings: {model_name}")
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'}  # Force CPU for stability on Render
                )
                print(f"✓ Successfully loaded embeddings: {model_name}")
                return embeddings
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {str(e)[:100]}...")
                continue
                
        print("All embedding models failed")
        return None
        
    except Exception as e:
        print(f"Embeddings initialization failed: {e}")
        return None

# Log every request (helps confirm OPTIONS preflight + POST are hitting)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    resp = await call_next(request)
    print(f"{request.method} {request.url.path} -> {resp.status_code}")
    return resp

# ────────────────────────────────────────────────────────────────────────────────
# LLM + Retrieval Setup
# ────────────────────────────────────────────────────────────────────────────────
qa_chain = None

def build_llm():
    # Temporarily disable LLM initialization
    print("LLM initialization disabled - using fallback responses only")
    return None

LLM = build_llm()

def chroma_present(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    if not files:
        return False
    if any(f.endswith(".sqlite3") or f == "chroma.sqlite3" for f in files):
        return True
    return any(os.path.isdir(os.path.join(path, d)) for d in files)

def get_chain():
    """Return RetrievalQA if embeddings/Chroma are ready; else None."""
    global qa_chain
    if qa_chain:
        return qa_chain
    try:
        if not LLM:
            print("No LLM; retrieval disabled.")
            return None
        if not chroma_present(CHROMA_DIR):
            print(f"No valid Chroma store in '{CHROMA_DIR}'. Using LLM-only.")
            return None

        try:
            # Try different embedding models
            embedding_models = [
                "text-embedding-004",
                "textembedding-gecko@003",
                "textembedding-gecko@001",
            ]
            
            for emb_model in embedding_models:
                try:
                    print(f"Trying embedding model: {emb_model}")
                    emb = VertexAIEmbeddings(
                        project=PROJECT_ID,
                        location=LOCATION,
                        model_name=emb_model,
                        credentials=credentials,
                    )
                    print(f"Successfully initialized embedding model: {emb_model}")
                    break
                except Exception as e:
                    print(f"Embedding model {emb_model} failed: {e}")
                    continue
            else:
                raise Exception("All embedding models failed")
        except Exception as e:
            print(f"Embeddings init failed: {e}\n{format_exc()}")
            return None
        except Exception as e:
            print(f"Embeddings init failed: {e}\n{format_exc()}")
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
        print("RetrievalQA ready.")
        return qa_chain
    except Exception as e:
        print(f"Retrieval init failed: {e}\n{format_exc()}")
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Helpers & Mock
# ────────────────────────────────────────────────────────────────────────────────
def get_enhanced_response(question: str) -> Dict[str, Any]:
    """Enhanced fallback responses with detailed Nigerian recipe information"""
    q = question.lower()
    
    # Jollof Rice responses
    if "jollof" in q or "jolloff" in q:
        return {
            "result": """Jollof Rice is Nigeria's most beloved dish! Here's how to make it:

**Ingredients:**
- 3 cups long-grain parboiled rice
- 1/4 cup vegetable oil  
- 1 medium onion, chopped
- 3-4 tomatoes, blended
- 2 tbsp tomato paste
- 2-3 scotch bonnet peppers
- 2 tsp curry powder
- 1 tsp thyme
- 2 bay leaves
- 4 cups chicken/beef stock
- Salt and seasoning cubes to taste

**Method:**
1. Heat oil, fry onions until golden
2. Add tomato paste, fry for 2 minutes
3. Add blended tomatoes, cook until oil separates
4. Add spices, cook for 5 minutes
5. Add rice, mix well, add stock
6. Cover, cook on medium heat for 20-25 minutes
7. Reduce heat, let it steam for 10 minutes

Serve with fried plantain, chicken, or salad!""",
            "source_documents": []
        }
    
    # Egusi Soup
    if "egusi" in q:
        return {
            "result": """Egusi Soup is a delicious Nigerian soup made with ground melon seeds:

**Ingredients:**
- 2 cups ground egusi (melon seeds)
- 1 lb meat (beef/goat meat)
- 1/2 cup palm oil
- 1 onion, chopped
- 2-3 scotch bonnet peppers
- 2 cups spinach or bitter leaf
- 1 cup locust beans (iru)
- Seasoning cubes, salt
- 4 cups meat stock

**Method:**
1. Cook meat with onions and seasoning until tender
2. Heat palm oil, add chopped onions
3. Add ground egusi, stir for 5 minutes
4. Gradually add meat stock, stirring constantly
5. Add cooked meat, peppers, locust beans
6. Simmer for 15 minutes
7. Add leafy vegetables, cook for 5 minutes

Best served with pounded yam, eba, or rice!""",
            "source_documents": []
        }
    
    # Suya
    if "suya" in q:
        return {
            "result": """Suya is Nigeria's favorite grilled meat snack:

**Suya Spice (Yaji) Mix:**
- 1 cup roasted groundnuts
- 2 tbsp ginger powder
- 1 tbsp garlic powder
- 1 tsp cloves
- 1 tsp nutmeg
- 2 scotch bonnet peppers (dried)
- 1 seasoning cube
- Salt to taste

**Method:**
1. Blend all spice ingredients together
2. Cut beef into thin strips
3. Thread onto skewers
4. Rub with oil and spice mix
5. Grill over hot coals, turning frequently
6. Sprinkle more spice mix while grilling
7. Serve with onions, tomatoes, and cucumber

Street food perfection!""",
            "source_documents": []
        }
    
    # Pepper Soup
    if "pepper soup" in q:
        return {
            "result": """Nigerian Pepper Soup - perfect for cold days or when you're feeling unwell:

**Pepper Soup Spice Mix:**
- Uda (Negro pepper)
- Uziza seeds
- Calabash nutmeg
- Scent leaves
- Ginger
- Garlic

**Method:**
1. Wash and season meat/fish
2. Boil until tender
3. Blend pepper soup spices
4. Add spices to pot with stock
5. Add scotch bonnet peppers
6. Season with salt and cubes
7. Add scent leaves, simmer 5 minutes

Great with white rice or yam. Known for its medicinal properties!""",
            "source_documents": []
        }
    
    # General ingredient-based responses
    if any(word in q for word in ["rice", "tomato", "onion", "pepper"]):
        return {
            "result": """With rice, tomato, onion, and pepper, you can make several Nigerian dishes:

**1. Jollof Rice** - The classic! Rice cooked in tomato-pepper sauce
**2. Fried Rice** - Stir-fried with vegetables and proteins  
**3. Coconut Rice** - Rice cooked in coconut milk with spices
**4. Tomato Rice** - Simple rice with fresh tomato sauce

Would you like a detailed recipe for any of these?""",
            "source_documents": []
        }
    
    if any(word in q for word in ["yam", "plantain"]):
        return {
            "result": """Yam and plantain are Nigerian staples! Here are popular preparations:

**Yam dishes:**
- Pounded Yam (served with soups)
- Boiled Yam with stew
- Yam Porridge (Asaro)
- Fried Yam

**Plantain dishes:**
- Dodo (fried sweet plantain)
- Plantain Porridge
- Bole (roasted plantain)
- Mosa (plantain pancakes)

Which would you like to learn how to make?""",
            "source_documents": []
        }
    
    # General cooking questions
    if any(word in q for word in ["cook", "recipe", "make", "how", "ingredient"]):
        return {
            "result": """I can help you with Nigerian recipes! Popular dishes include:

**Rice dishes:** Jollof Rice, Fried Rice, Coconut Rice
**Soups:** Egusi, Pepper Soup, Okra Soup, Bitterleaf Soup  
**Swallows:** Pounded Yam, Eba, Amala, Fufu
**Grilled:** Suya, Kilishi, Barbecue Fish
**Snacks:** Puff Puff, Akara, Moi Moi, Chin Chin

Tell me what ingredients you have or which dish interests you!""",
            "source_documents": []
        }
    
    # Default response
    return {
        "result": """Hello! I'm your Nigerian food assistant. I specialize in traditional Nigerian recipes and can help you:

- Cook popular dishes like Jollof Rice, Egusi Soup, Suya
- Suggest recipes based on ingredients you have
- Explain cooking techniques and ingredients
- Share cultural food knowledge

What would you like to cook today?""",
        "source_documents": []
    }

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

# ────────────────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Nigerian Food Recommender API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "mode": "enhanced_fallback_only",
        "project": PROJECT_ID,
        "location": LOCATION,
        "google_creds_set": bool(credentials),
        "note": "Using detailed Nigerian recipe responses - no AI models needed!"
    }

# requirements.txt should include:
# langchain
# langchain-community  
# transformers
# torch
# sentence-transformers
# chromadb
# tiktoken
# pandas
# fastapi
# uvicorn
# python-dotenv

@app.get("/ask")
async def ask_get_hint():
    return {"hint": "Use POST /ask with JSON body: {\"question\": \"...\"}"}

@app.post("/ask")
async def ask(request: AskRequest) -> Dict[str, Any]:
    try:
        # Always use enhanced fallback responses for now
        fallback = get_enhanced_response(request.question)
        return {"answer": fallback["result"], "sources": fallback["source_documents"]}
        
    except Exception as e:
        print(f"[/ask] Error: {e}\n{format_exc()}")
        # Ultimate fallback
        return {
            "answer": "I'm having technical difficulties, but I can still help with Nigerian recipes! Ask me about Jollof Rice, Egusi Soup, Suya, or share your ingredients.",
            "sources": []
        }