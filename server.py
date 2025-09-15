
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Environment variables
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
NETLIFY_ORIGIN = os.environ.get("NETLIFY_ORIGIN", "*")
CHROMA_DIR = "chroma_db"

app = FastAPI(title="Nigerian Food Recommender API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[NETLIFY_ORIGIN] if NETLIFY_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
qa_chain = None

def get_chain():
    global qa_chain
    if qa_chain:
        return qa_chain

    try:
        # Check if chroma_db exists
        if not os.path.exists(CHROMA_DIR):
            print(f"Warning: {CHROMA_DIR} not found, using mock responses")
            return None

        # Initialize embeddings and vector store
        emb = VertexAIEmbeddings(
            project=PROJECT_ID,
            location=LOCATION,
            model_name="text-embedding-004"
        )
        
        vectordb = Chroma(
            collection_name="knowledge_base",
            embedding_function=emb,
            persist_directory=CHROMA_DIR,
        )

        # Initialize LLM
        llm = ChatVertexAI(
            model="gemini-1.5-flash",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0.2,
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
        )
        
        return qa_chain
    
    except Exception as e:
        print(f"Error initializing chain: {e}")
        return None

def get_mock_response(question):
    """Fallback mock responses if database isn't available"""
    mock_data = {
        'jollof': {
            "result": "Jollof Rice is Nigeria's most famous dish. Made with rice, tomatoes, onions, and spices...",
            "source_documents": []
        },
        'default': {
            "result": "I can help with Nigerian recipes! Try asking about Jollof Rice, Egusi, or other dishes.",
            "source_documents": []
        }
    }
    
    question_lower = question.lower()
    for key in mock_data:
        if key in question_lower and key != 'default':
            return mock_data[key]
    
    return mock_data['default']

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: AskRequest):
    try:
        chain = get_chain()
        
        if not chain:
            # Use mock data if chain initialization failed
            result = get_mock_response(request.question)
        else:
            # Use real chain
            result = chain.invoke({"query": request.question})
        
        # Format response
        answer = result.get("result", "")
        sources = []
        
        for doc in result.get("source_documents", []):
            sources.append({
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("row", None),
                "snippet": doc.page_content[:240] + "..." if len(doc.page_content) > 240 else doc.page_content
            })
        
        return {"answer": answer, "sources": sources}
        
    except Exception as e:
        print(f"Error in ask endpoint: {e}")
        # Return mock response on error
        mock_result = get_mock_response(request.question)
        return {
            "answer": mock_result["result"],
            "sources": []
        }

@app.get("/")
async def root():
    return {"message": "Nigerian Food Recommender API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "project": PROJECT_ID}