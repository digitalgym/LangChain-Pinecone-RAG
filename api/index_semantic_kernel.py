from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
import asyncio

# Import Semantic Kernel
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding, OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings

# Import Pinecone
from pinecone import Pinecone

load_dotenv()

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class SemanticKernelRAGService:
    def __init__(self):
        # Initialize Semantic Kernel
        self.kernel = sk.Kernel()
        
        # Add OpenAI services
        self.kernel.add_service(
            OpenAITextEmbedding(
                ai_model_id="text-embedding-3-large",
                api_key=os.environ.get("OPENAI_API_KEY"),
                service_id="text-embedding"
            )
        )
        
        self.kernel.add_service(
            OpenAIChatCompletion(
                ai_model_id="gpt-4o",
                api_key=os.environ.get("OPENAI_API_KEY"),
                service_id="chat-completion"
            )
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = os.environ.get("PINECONE_INDEX_NAME")
        self.index = self.pc.Index(self.index_name)
        
        # Get embedding service for manual operations
        self.embedding_service = self.kernel.get_service(service_id="text-embedding")
        self.chat_service = self.kernel.get_service(service_id="chat-completion")
    
    async def get_embedding(self, text):
        """Get embedding for text using Semantic Kernel"""
        embeddings = await self.embedding_service.generate_embeddings([text])
        # Convert numpy array to Python list for Pinecone compatibility
        embedding = embeddings[0]
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        elif hasattr(embedding, '__iter__'):
            return list(embedding)
        else:
            return embedding
    
    async def search_similar_documents(self, query, k=3, score_threshold=0.5):
        """Search for similar documents using Semantic Kernel and Pinecone"""
        # Get embedding for the query
        query_embedding = await self.get_embedding(query)
        
        # Search in Pinecone
        search_results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            include_values=False
        )
        
        # Filter by score threshold and format results
        results = []
        for match in search_results['matches']:
            if match['score'] >= score_threshold:
                results.append({
                    'content': match['metadata'].get('text', ''),
                    'metadata': match['metadata'],
                    'score': match['score']
                })
        
        return results
    
    async def generate_response_with_context(self, query, context_docs, conversation_history):
        """Generate response using retrieved context and conversation history"""
        # Prepare context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in context_docs])
        
        # Create system prompt with context
        system_prompt = f"""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        
        Context: {context}"""
        
        # Create chat history
        chat_history = ChatHistory()
        
        # Add system message
        chat_history.add_system_message(system_prompt)
        
        # Add conversation history
        for msg in conversation_history:
            if msg["type"] == "human":
                chat_history.add_user_message(msg["content"])
            elif msg["type"] == "ai":
                chat_history.add_assistant_message(msg["content"])
        
        # Add current user query
        chat_history.add_user_message(query)
        
        # Create execution settings
        settings = OpenAIChatPromptExecutionSettings(
            ai_model_id="gpt-4o",
            max_tokens=1000,
            temperature=1
        )
        
        # Generate response
        response = await self.chat_service.get_chat_message_contents(
            chat_history=chat_history,
            settings=settings
        )
        
        return str(response[0].content) if response else "No response generated"

# Initialize the RAG service
rag_service = SemanticKernelRAGService()

# In-memory session storage (in production, use Redis or database)
sessions: Dict[str, List[Dict[str, Any]]] = {}

class ChatMessage(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

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
            sessions[session_id] = []
        
        # Add user message to session
        sessions[session_id].append({"type": "human", "content": user_message})
        
        # Search for relevant documents
        docs = await rag_service.search_similar_documents(
            user_message, 
            k=3, 
            score_threshold=0.5
        )
        
        # Generate response with context and conversation history
        result = await rag_service.generate_response_with_context(
            user_message, 
            docs, 
            sessions[session_id][:-1]  # Exclude the current message from history
        )
        
        # Add AI response to session
        sessions[session_id].append({"type": "ai", "content": result})
        
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
