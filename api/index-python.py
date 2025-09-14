from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import json

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Pinecone and other services
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Initialize embeddings model + vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ.get("OPENAI_API_KEY")
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=1,
    api_key=os.environ.get("OPENAI_API_KEY")
)

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
            sessions[session_id] = [
                {"type": "system", "content": "You are an assistant for question-answering tasks."}
            ]
        
        # Add user message to session
        sessions[session_id].append({"type": "human", "content": user_message})
        
        # Create retriever and get relevant documents
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.5},
        )
        
        docs = retriever.invoke(user_message)
        docs_text = "".join(d.page_content for d in docs)
        
        # Create system prompt with context
        system_prompt = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Context: {context}"""
        
        system_prompt_fmt = system_prompt.format(context=docs_text)
        
        # Convert session messages to LangChain format
        langchain_messages = []
        for msg in sessions[session_id]:
            if msg["type"] == "system":
                langchain_messages.append(SystemMessage(msg["content"]))
            elif msg["type"] == "human":
                langchain_messages.append(HumanMessage(msg["content"]))
            elif msg["type"] == "ai":
                langchain_messages.append(AIMessage(msg["content"]))
        
        # Add the context-enhanced system message
        langchain_messages.append(SystemMessage(system_prompt_fmt))
        
        # Get response from LLM
        result = llm.invoke(langchain_messages).content
        
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
