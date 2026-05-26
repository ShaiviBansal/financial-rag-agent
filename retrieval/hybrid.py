from retrieval.semantic import semantic_search
from retrieval.bm25_search import bm25_search

def hybrid_search(query: str, ticker: str = None, n_results: int = 5, alpha: float = 0.7):
    semantic_results = semantic_search(query, ticker, n_results)
    bm25_results = bm25_search(query, ticker, n_results)
    
    combined = {}
    
    for i, doc in enumerate(semantic_results):
        key = doc["text"][:100]
        combined[key] = {
            "text": doc["text"],
            "metadata": doc["metadata"],
            "score": alpha * doc["score"]
        }
    
    max_bm25 = max([d["score"] for d in bm25_results], default=1)
    
    for i, doc in enumerate(bm25_results):
        key = doc["text"][:100]
        normalized_score = doc["score"] / max_bm25 if max_bm25 > 0 else 0
        if key in combined:
            combined[key]["score"] += (1 - alpha) * normalized_score
        else:
            combined[key] = {
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": (1 - alpha) * normalized_score
            }
    
    sorted_docs = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return sorted_docs[:n_results]