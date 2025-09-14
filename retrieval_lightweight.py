# import basics
import os
import openai
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone

load_dotenv()

# initialize OpenAI client
openai.api_key = os.environ.get("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# set the pinecone index
index_name = os.environ.get("PINECONE_INDEX_NAME") 
index = pc.Index(index_name)

def get_embedding(text, model="text-embedding-3-large"):
    """Get embedding for a text using OpenAI API directly"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def search_similar_documents(query, k=5, score_threshold=0.5):
    """Search for similar documents in Pinecone index"""
    # Get embedding for the query
    query_embedding = get_embedding(query)
    
    # Search in Pinecone
    search_results = index.query(
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

# Example usage
if __name__ == "__main__":
    query = "what is retrieval augmented generation?"
    results = search_similar_documents(query, k=5, score_threshold=0.5)
    
    # show results
    print("RESULTS:")
    
    for res in results:
        print(f"* {res['content'][:200]}... [Score: {res['score']:.3f}] [{res['metadata']}]")
