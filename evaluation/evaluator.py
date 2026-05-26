# from ragas.metrics import faithfulness, answer_relevancy, context_precision
# from ragas import evaluate
# from datasets import Dataset
# import os
# from dotenv import load_dotenv

# load_dotenv()

# os.environ["OPENAI_API_KEY"] = "dummy"

# def evaluate_response(query: str, answer: str, contexts: list) -> dict:
#     try:
#         data = {
#             "question": [query],
#             "answer": [answer],
#             "contexts": [contexts],
#             "ground_truth": [answer]
#         }
#         dataset = Dataset.from_dict(data)
        
#         result = evaluate(
#             dataset,
#             metrics=[faithfulness, answer_relevancy, context_precision]
#         )
        
#         return {
#             "faithfulness": round(float(result["faithfulness"]), 3),
#             "answer_relevancy": round(float(result["answer_relevancy"]), 3),
#             "context_precision": round(float(result["context_precision"]), 3),
#         }
#     except Exception as e:
#         return {
#             "faithfulness": 0.0,
#             "answer_relevancy": 0.0,
#             "context_precision": 0.0,
#             "error": str(e)
#         }





from groq import Groq
from dotenv import load_dotenv
import os

import re

def extract_score(text: str) -> float:
    match = re.search(r'\b(0(\.\d+)?|1(\.0+)?)\b', text)
    return float(match.group()) if match else 0.5

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def evaluate_response(query: str, answer: str, contexts: list) -> dict:
    try:
        context_str = "\n\n".join(contexts[:3])
        
        # Faithfulness check
        faithfulness_prompt = f"""Rate how faithful this answer is to the given context on a scale of 0 to 1.
Context: {context_str[:2000]}
Answer: {answer}
Return only a number between 0 and 1."""

        faith_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": faithfulness_prompt}],
            max_tokens=10
        )
        faithfulness = extract_score(faith_res.choices[0].message.content.strip())
        faithfulness = max(0.0, min(1.0, faithfulness))

        # Relevancy check
        relevancy_prompt = f"""Rate how relevant this answer is to the question on a scale of 0 to 1.
Question: {query}
Answer: {answer}
Return only a number between 0 and 1."""

        rel_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": relevancy_prompt}],
            max_tokens=10
        )
        relevancy = extract_score(rel_res.choices[0].message.content.strip())
        relevancy = max(0.0, min(1.0, relevancy))

        # Context precision check
        precision_prompt = f"""Rate how precisely the context supports answering the question on a scale of 0 to 1.
Question: {query}
Context: {context_str[:2000]}
Return only a number between 0 and 1."""

        prec_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": precision_prompt}],
            max_tokens=10
        )
        precision = extract_score(prec_res.choices[0].message.content.strip())
        precision = max(0.0, min(1.0, precision))

        return {
            "faithfulness": round(faithfulness, 3),
            "answer_relevancy": round(relevancy, 3),
            "context_precision": round(precision, 3)
        }

    except Exception as e:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "error": str(e)
        }
