# Agentic Financial RAG System

Multi-document financial analysis over SEC 10-K filings using an agentic retrieval pipeline with hybrid search and LLM-based evaluation.

## What it does

- Downloads and indexes SEC 10-K filings for Apple, Microsoft, and Google (2022-2023)
- Routes queries intelligently — factual queries go to BM25, conceptual to semantic search, comparisons to hybrid
- Retrieves relevant chunks using hybrid search (BM25 + ChromaDB vector similarity fusion)
- Generates answers via Groq-hosted Llama 3.1 through a LangGraph agent with three nodes: router, retriever, generator
- Evaluates every response on faithfulness, answer relevancy, and context precision using LLM-based scoring
- JWT-authenticated FastAPI REST API with query history and session stats
- Streamlit dashboard with live evaluation scores and query history

## Tech Stack

- Agent: LangGraph with query routing, retrieval, and generation nodes
- Retrieval: ChromaDB semantic search + BM25 keyword search + hybrid fusion
- LLM: Llama 3.1 via Groq API (free tier)
- Embeddings: sentence-transformers all-MiniLM-L6-v2
- API: FastAPI with JWT authentication
- Dashboard: Streamlit
- Data: SEC EDGAR API (free, no key needed)

## Run Locally

### 1. Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file in the root folder with the following:

```
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

### 3. Ingest documents (first time only, takes 10-15 minutes)

```
python ingestion/ingest.py
```

### 4. Run the API

```
python -m uvicorn api.main:app --port 8000
```

### 5. Run the Dashboard

```
streamlit run dashboard/app.py
```

## Example Queries

- What are the main risk factors for Apple?
- What business segments does Microsoft operate in?
- How does Google generate its revenue?
- Compare Apple and Microsoft's approach to AI
- What are the competitive risks in Google's 10-K?
