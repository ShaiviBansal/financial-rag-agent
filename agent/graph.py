from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from retrieval.hybrid import hybrid_search
from retrieval.semantic import semantic_search
from retrieval.bm25_search import bm25_search
from dotenv import load_dotenv
import os
import re

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.1
)

COMPANIES = ["AAPL", "MSFT", "GOOGL"]

# --- State ---
class AgentState(TypedDict):
    query: str
    query_type: str
    tickers: List[str]
    search_strategy: str
    retrieved_docs: List[dict]
    answer: str
    contexts: List[str]

# --- Node 1: Query Router ---
def route_query(state: AgentState) -> AgentState:
    query = state["query"]
    
    # Detect tickers mentioned
    # Check for company names too
    name_map = {"APPLE": "AAPL", "MICROSOFT": "MSFT", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL"}
    tickers = [t for t in COMPANIES if t in query.upper()]
    for name, ticker in name_map.items():
        if name in query.upper() and ticker not in tickers:
            tickers.append(ticker)
    if not tickers:
        tickers = list(COMPANIES)
    
    # Detect query type
    comparison_keywords = ["compare", "versus", "vs", "difference", "better", "more than", "less than"]
    factual_keywords = ["what is", "how much", "revenue", "profit", "earnings", "eps", "debt"]
    
    query_lower = query.lower()
    if any(k in query_lower for k in comparison_keywords):
        query_type = "comparison"
        search_strategy = "hybrid"
    elif any(k in query_lower for k in factual_keywords):
        query_type = "factual"
        search_strategy = "bm25"
    else:
        query_type = "conceptual"
        search_strategy = "semantic"
    
    return {
        **state,
        "query_type": query_type,
        "tickers": tickers,
        "search_strategy": search_strategy
    }

# --- Node 2: Retriever ---
def retrieve_docs(state: AgentState) -> AgentState:
    query = state["query"]
    tickers = state["tickers"]
    strategy = state["search_strategy"]
    
    all_docs = []
    
    for ticker in tickers:
        if strategy == "hybrid":
            docs = hybrid_search(query, ticker=ticker, n_results=3)
        elif strategy == "bm25":
            docs = bm25_search(query, ticker=ticker, n_results=3)
        else:
            docs = semantic_search(query, ticker=ticker, n_results=3)
        all_docs.extend(docs)
    
    # Sort by score and take top 8
    all_docs = sorted(all_docs, key=lambda x: x["score"], reverse=True)[:8]
    contexts = [d["text"] for d in all_docs]
    
    return {**state, "retrieved_docs": all_docs, "contexts": contexts}

# --- Node 3: Answer Generator ---
def generate_answer(state: AgentState) -> AgentState:
    query = state["query"]
    contexts = state["contexts"]
    tickers = state["tickers"]
    query_type = state["query_type"]
    
    context_str = "\n\n---\n\n".join(contexts)
    
    system_prompt = f"""You are a financial analyst assistant specializing in SEC 10-K filings.
You have access to 10-K filings for: {', '.join(tickers)}.
Query type detected: {query_type}

Answer the question based ONLY on the provided context from the filings.
Be specific, cite figures when available, and be concise.
If the context doesn't contain enough information, say so clearly."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context from 10-K filings:\n{context_str}\n\nQuestion: {query}")
    ]
    
    response = llm.invoke(messages)
    
    return {**state, "answer": response.content}

# --- Build Graph ---
def build_graph():
    graph = StateGraph(AgentState)
    
    graph.add_node("router", route_query)
    graph.add_node("retriever", retrieve_docs)
    graph.add_node("generator", generate_answer)
    
    graph.set_entry_point("router")
    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)
    
    return graph.compile()

agent = build_graph()

def run_agent(query: str) -> dict:
    initial_state = {
        "query": query,
        "query_type": "",
        "tickers": [],
        "search_strategy": "",
        "retrieved_docs": [],
        "answer": "",
        "contexts": []
    }
    result = agent.invoke(initial_state)
    return result