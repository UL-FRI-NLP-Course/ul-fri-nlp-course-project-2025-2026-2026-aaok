
import json
from datetime import datetime

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#1. Load embeddings FAISS
print("Loading FAISS index...")

embeddings = HuggingFaceEmbeddings(
    # model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
# print(vectorstore.index.d)
# print(len(embeddings.embed_query("test")))

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6},
)

print("FAISS loaded.")

#2. Format docs
def format_docs(docs):
    return [
        {
            "source": d.metadata.get("naziv", "neznan"),
            "content": d.page_content
        }
        for d in docs
    ]

#3. question
question = "Koliko minimalnega letnega dopusta pripada zaposlenemu?"
question = "Ali me lahko šef tepe?"
question = "Se lahko študentje zaposlijo v več delovnih mestih naenkrat?"
print("Running retrieval...")

docs = retriever.invoke(question)

formatted_docs = format_docs(docs)

#4. Save to file
output = {
    "timestamp": datetime.now().isoformat(),
    "question": question,
    "chunks": formatted_docs
}

output_file = "rag_output.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved RAG output to {output_file}")