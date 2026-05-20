import json
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging
import warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# 1. LOAD FAISS
print("Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 100, "lambda_mult": 0.9},
)
print("FAISS loaded.")

# 2. LOAD QUESTIONS
with open("questions.json", "r", encoding="utf-8") as f:
    questions_data = json.load(f)

print(f"Loaded {len(questions_data)} questions from questions.json")

# 3. RETRIEVE
results = []
for i, item in enumerate(questions_data):
    question = item["question"]
    print(f"[{i+1}/{len(questions_data)}] Retrieving: {question}")
    docs = retriever.invoke(question)
    chunks = [
        {
            "source": d.metadata.get("naziv", "neznan"),
            "content": d.page_content
        }
        for d in docs
    ]
    results.append({
        "question": question,
        "chunks": chunks,
        "expected_answer": item.get("expected_answer", ""),
        "answers": {}
    })

# 4. SAVE
output_file = "answers.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nDone! Saved {len(results)} questions with chunks to {output_file}")