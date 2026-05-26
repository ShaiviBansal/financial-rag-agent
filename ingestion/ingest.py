import os
import re
from pathlib import Path
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
BM25_PATH = Path("bm25_index.pkl")
COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc."
}
YEARS = [2022, 2023]

# --- Download 10-K filings ---
def download_filings():
    DATA_DIR.mkdir(exist_ok=True)
    dl = Downloader("FinancialRAG", "shaivi@example.com", DATA_DIR)
    for ticker in COMPANIES:
        for year in YEARS:
            print(f"Downloading {ticker} 10-K for {year}...")
            try:
                dl.get("10-K", ticker, after=f"{year}-01-01", before=f"{year}-12-31", limit=1)
            except Exception as e:
                print(f"Error downloading {ticker} {year}: {e}")

# --- Parse HTML filings ---
def parse_filing(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Chunk text ---
def chunk_text(text, ticker, year):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append({
            "text": chunk,
            "metadata": {
                "ticker": ticker,
                "year": str(year),
                "chunk_id": i,
                "source": f"{ticker}_{year}_10K"
            }
        })
    return docs

# --- Index into ChromaDB ---
def index_to_chroma(all_docs):
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    try:
        client.delete_collection("financial_docs")
    except:
        pass
    
    collection = client.create_collection("financial_docs")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = [d["text"] for d in all_docs]
    metadatas = [d["metadata"] for d in all_docs]
    ids = [f"doc_{i}" for i in range(len(all_docs))]
    
    print(f"Embedding {len(texts)} chunks...")
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        embeddings = model.encode(batch_texts).tolist()
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )
        print(f"Indexed {min(i+batch_size, len(texts))}/{len(texts)} chunks")
    
    print("ChromaDB indexing complete.")
    return collection

# --- Build BM25 index ---
def build_bm25(all_docs):
    texts = [d["text"] for d in all_docs]
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": all_docs}, f)
    print(f"BM25 index saved with {len(texts)} documents.")

# --- Main ---
def main():
    print("=== Downloading SEC 10-K Filings ===")
    download_filings()
    
    all_docs = []
    for ticker in COMPANIES:
        ticker_dir = DATA_DIR / "sec-edgar-filings" / ticker / "10-K"
        if not ticker_dir.exists():
            print(f"No filings found for {ticker}")
            continue
        for filing_dir in ticker_dir.iterdir():
            for file in filing_dir.iterdir():
                if file.suffix in [".html", ".htm", ".txt"]:
                    print(f"Parsing {file}...")
                    text = parse_filing(file)
                    if len(text) > 500:
                        # Extract year from folder name
                        year = "2023" if "23" in filing_dir.name else "2022"
                        docs = chunk_text(text, ticker, year)
                        all_docs.extend(docs)
                    break
    
    if not all_docs:
        print("No documents found. Check downloads.")
        return
    
    print(f"\nTotal chunks: {len(all_docs)}")
    print("=== Building ChromaDB Index ===")
    index_to_chroma(all_docs)
    print("=== Building BM25 Index ===")
    build_bm25(all_docs)
    print("\n=== Ingestion Complete ===")

if __name__ == "__main__":
    main()