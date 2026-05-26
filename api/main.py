from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import time

from agent.graph import run_agent
from evaluation.evaluator import evaluate_response

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

app = FastAPI(title="Financial RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- In-memory user store ---
users_db = {}
query_history = []

# --- Auth helpers ---
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None or email not in users_db:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- Models ---
class UserRegister(BaseModel):
    email: str
    password: str

class QueryRequest(BaseModel):
    query: str
    evaluate: bool = True

# --- Routes ---
@app.get("/")
def root():
    return {"message": "Financial RAG API is running"}

@app.post("/register")
def register(user: UserRegister):
    if user.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    users_db[user.email] = hash_password(user.password)
    return {"message": "User registered successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(form_data.password, users_db[form_data.username]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token({"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/query")
def query(request: QueryRequest, current_user: str = Depends(get_current_user)):
    start = time.time()
    
    result = run_agent(request.query)
    latency = round(time.time() - start, 2)
    
    eval_scores = {}
    if request.evaluate and result["contexts"]:
        eval_scores = evaluate_response(
            request.query,
            result["answer"],
            result["contexts"]
        )
    
    response = {
        "query": request.query,
        "answer": result["answer"],
        "query_type": result["query_type"],
        "tickers": result["tickers"],
        "search_strategy": result["search_strategy"],
        "latency_seconds": latency,
        "eval_scores": eval_scores,
        "n_docs_retrieved": len(result["retrieved_docs"])
    }
    
    query_history.append({
        "user": current_user,
        "query": request.query,
        "latency": latency,
        "eval_scores": eval_scores,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return response

@app.get("/history")
def get_history(current_user: str = Depends(get_current_user)):
    user_history = [h for h in query_history if h["user"] == current_user]
    return {"history": user_history}

@app.get("/stats")
def get_stats(current_user: str = Depends(get_current_user)):
    user_history = [h for h in query_history if h["user"] == current_user]
    if not user_history:
        return {"message": "No queries yet"}
    
    avg_latency = sum(h["latency"] for h in user_history) / len(user_history)
    avg_faithfulness = sum(h["eval_scores"].get("faithfulness", 0) for h in user_history) / len(user_history)
    avg_relevancy = sum(h["eval_scores"].get("answer_relevancy", 0) for h in user_history) / len(user_history)
    
    return {
        "total_queries": len(user_history),
        "avg_latency_seconds": round(avg_latency, 2),
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_answer_relevancy": round(avg_relevancy, 3)
    }