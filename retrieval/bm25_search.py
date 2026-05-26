import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

BM25_PATH = Path("bm25_index.pkl")

def load_bm25():
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["docs"]

def bm25_search(query: str, ticker: str = None, n_results: int = 5):
    bm25, all_docs = load_bm25()
    
    if ticker:
        filtered = [(i, d) for i, d in enumerate(all_docs) if d["metadata"]["ticker"] == ticker]
        indices, docs = zip(*filtered) if filtered else ([], [])
        tokenized = [d["text"].lower().split() for d in docs]
        bm25_filtered = BM25Okapi(tokenized)
        scores = bm25_filtered.get_scores(query.lower().split())
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        return [{"text": docs[i]["text"], "metadata": docs[i]["metadata"], "score": float(scores[i])} for i in top_n]
    else:
        scores = bm25.get_scores(query.lower().split())
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        return [{"text": all_docs[i]["text"], "metadata": all_docs[i]["metadata"], "score": float(scores[i])} for i in top_n]