import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Financial RAG", page_icon="📊", layout="wide")
st.title("📊 Agentic Financial RAG System")
st.markdown("Multi-document analysis over SEC 10-K filings — Apple, Microsoft, Google")

# --- Session state ---
if "token" not in st.session_state:
    st.session_state.token = None
if "history" not in st.session_state:
    st.session_state.history = []

# --- Auth sidebar ---
with st.sidebar:
    st.header("Authentication")
    if st.session_state.token:
        st.success("Logged in")
        if st.button("Logout"):
            st.session_state.token = None
            st.rerun()
    else:
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                res = requests.post(f"{API_URL}/login", data={"username": email, "password": password})
                if res.status_code == 200:
                    st.session_state.token = res.json()["access_token"]
                    st.success("Logged in!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        with tab2:
            reg_email = st.text_input("Email", key="reg_email")
            reg_pass = st.text_input("Password", type="password", key="reg_pass")
            if st.button("Register"):
                res = requests.post(f"{API_URL}/register", json={"email": reg_email, "password": reg_pass})
                if res.status_code == 200:
                    st.success("Registered! Please login.")
                else:
                    st.error(res.json().get("detail", "Error"))

# --- Main interface ---
if not st.session_state.token:
    st.info("Please login or register in the sidebar to start querying.")
else:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}

    # --- Query section ---
    st.header("Query Financial Documents")
    
    example_queries = [
        "What was Apple's total revenue in 2023?",
        "Compare Microsoft and Google's R&D spending",
        "What are the main risk factors for Apple?",
        "How has MSFT's cloud revenue grown?",
        "Compare AAPL and GOOGL net income"
    ]
    
    selected = st.selectbox("Example queries:", ["Custom query..."] + example_queries)
    
    if selected == "Custom query...":
        query = st.text_input("Enter your question:")
    else:
        query = selected
    
    evaluate = st.checkbox("Run RAGAS evaluation", value=True)
    
    if st.button("Ask") and query:
        with st.spinner("Agent is thinking..."):
            res = requests.post(
                f"{API_URL}/query",
                json={"query": query, "evaluate": evaluate},
                headers=headers
            )
        
        if res.status_code == 200:
            data = res.json()
            
            # Answer
            st.subheader("Answer")
            st.write(data["answer"])
            
            # Metadata
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Query Type", data["query_type"])
            col2.metric("Strategy", data["search_strategy"])
            col3.metric("Docs Retrieved", data["n_docs_retrieved"])
            col4.metric("Latency", f"{data['latency_seconds']}s")
            
            # Eval scores
            if data["eval_scores"] and "error" not in data["eval_scores"]:
                st.subheader("RAGAS Evaluation Scores")
                e = data["eval_scores"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Faithfulness", e.get("faithfulness", "N/A"))
                c2.metric("Answer Relevancy", e.get("answer_relevancy", "N/A"))
                c3.metric("Context Precision", e.get("context_precision", "N/A"))
            
            st.session_state.history.append(data)
        else:
            st.error(f"Error: {res.json().get('detail', 'Unknown error')}")

    # --- Stats section ---
    if st.session_state.history:
        st.header("Session Stats")
        res = requests.get(f"{API_URL}/stats", headers=headers)
        if res.status_code == 200:
            stats = res.json()
            if "total_queries" in stats:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Queries", stats["total_queries"])
                c2.metric("Avg Latency", f"{stats['avg_latency_seconds']}s")
                c3.metric("Avg Faithfulness", stats["avg_faithfulness"])
                c4.metric("Avg Relevancy", stats["avg_answer_relevancy"])
        
        # --- History ---
        st.header("Query History")
        for i, h in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Q{len(st.session_state.history)-i}: {h['query'][:60]}..."):
                st.write(h["answer"])
                if h["eval_scores"]:
                    st.json(h["eval_scores"])