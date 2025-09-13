from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import openai
import pinecone
import requests

load_dotenv()

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize Pinecone
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
)

index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pinecone.Index(index_name)

# In-memory session storage
sessions: Dict[str, List[Dict[str, Any]]] = {}

class ChatMessage(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

def get_embeddings(text: str) -> List[float]:
    """Get embeddings using OpenAI API directly"""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting embeddings: {str(e)}")

def query_pinecone(query_text: str, k: int = 3) -> List[str]:
    """Query Pinecone directly and return relevant documents"""
    try:
        # Get embeddings for the query
        query_embedding = get_embeddings(query_text)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        # Extract text content from results
        docs = []
        for match in results.matches:
            if 'text' in match.metadata:
                docs.append(match.metadata['text'])
        
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Pinecone: {str(e)}")

def chat_with_openai(messages: List[Dict[str, str]]) -> str:
    """Chat with OpenAI directly"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=1,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with OpenAI: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("static/index.html")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    try:
        session_id = chat_message.session_id
        user_message = chat_message.message
        
        # Initialize session if it doesn't exist
        if session_id not in sessions:
            sessions[session_id] = [
                {"role": "system", "content": "You are an assistant for question-answering tasks."}
            ]
        
        # Add user message to session
        sessions[session_id].append({"role": "user", "content": user_message})
        
        # Query Pinecone for relevant documents
        docs = query_pinecone(user_message)
        docs_text = "\n".join(docs)
        
        # Create system prompt with context
        system_prompt = f"""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Context: {docs_text}"""
        
        # Prepare messages for OpenAI
        messages = sessions[session_id].copy()
        messages.append({"role": "system", "content": system_prompt})
        
        # Get response from OpenAI
        result = chat_with_openai(messages)
        
        # Add AI response to session
        sessions[session_id].append({"role": "assistant", "content": result})
        
        return ChatResponse(response=result, session_id=session_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.delete("/api/chat/{session_id}")
async def clear_chat(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Chat history cleared"}

# For Vercel deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
