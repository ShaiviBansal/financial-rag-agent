import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHROMA_DIR = Path("chroma_db")
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection("financial_docs")

def semantic_search(query: str, ticker: str = None, n_results: int = 5):
    collection = get_collection()
    query_embedding = model.encode(query).tolist()
    
    where = {"ticker": ticker} if ticker else None
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where
    )
    
    docs = []
    for i, doc in enumerate(results["documents"][0]):
        docs.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i]
        })
    return docs