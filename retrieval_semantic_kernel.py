# import basics
import os
import asyncio
from dotenv import load_dotenv

# import semantic kernel
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding, OpenAIChatCompletion

# import pinecone for vector storage
from pinecone import Pinecone

load_dotenv()

class SemanticKernelRAG:
    def __init__(self):
        # Initialize Semantic Kernel
        self.kernel = sk.Kernel()
        
        # Add OpenAI services
        self.kernel.add_service(
            OpenAITextEmbedding(
                ai_model_id="text-embedding-3-large",
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        )
        
        self.kernel.add_service(
            OpenAIChatCompletion(
                ai_model_id="gpt-3.5-turbo",
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = os.environ.get("PINECONE_INDEX_NAME")
        self.index = self.pc.Index(self.index_name)
        
        # Get embedding service for manual operations
        self.embedding_service = self.kernel.get_service(type=OpenAITextEmbedding)
    
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
    
    async def search_similar_documents(self, query, k=5, score_threshold=0.5):
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
    
    async def generate_response(self, query, context_docs):
        """Generate response using retrieved context"""
        # Prepare context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in context_docs])
        
        # Create prompt with context
        prompt = f"""
        Context information:
        {context}
        
        Question: {query}
        
        Please provide a comprehensive answer based on the context provided above.
        """
        
        # Get chat completion service and generate response
        chat_service = self.kernel.get_service(type=OpenAIChatCompletion)
        
        # Create chat history with the prompt
        from semantic_kernel.contents import ChatHistory
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
        
        chat_history = ChatHistory()
        chat_history.add_user_message(prompt)
        
        # Create proper execution settings
        settings = OpenAIChatPromptExecutionSettings(
            ai_model_id="gpt-3.5-turbo",
            max_tokens=1000,
            temperature=0.7
        )
        
        # Generate response
        response = await chat_service.get_chat_message_contents(
            chat_history=chat_history,
            settings=settings
        )
        
        return str(response[0].content) if response else "No response generated"
    
    async def rag_pipeline(self, query, k=5, score_threshold=0.5):
        """Complete RAG pipeline: retrieve and generate"""
        # Retrieve relevant documents
        documents = await self.search_similar_documents(query, k, score_threshold)
        
        if not documents:
            return "No relevant documents found for your query."
        
        # Generate response
        response = await self.generate_response(query, documents)
        
        return {
            'answer': response,
            'sources': documents,
            'query': query
        }

# Example usage
async def main():
    rag = SemanticKernelRAG()
    
    query = "what is retrieval augmented generation?"
    result = await rag.rag_pipeline(query, k=5, score_threshold=0.5)
    
    print("QUERY:", result['query'])
    print("\nANSWER:")
    print(result['answer'])
    print("\nSOURCES:")
    for i, doc in enumerate(result['sources'], 1):
        print(f"{i}. {doc['content'][:200]}... [Score: {doc['score']:.3f}]")

if __name__ == "__main__":
    asyncio.run(main())
