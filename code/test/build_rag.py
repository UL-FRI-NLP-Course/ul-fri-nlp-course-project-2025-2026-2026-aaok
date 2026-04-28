import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re

#1. Load your filtered laws
docs = []
with open("laws.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        law = json.loads(line)
        docs.append(Document(
            page_content=law["text"],
            metadata={
                "naziv": law["naziv"],
                "id":    law["id"],
                "sop":   law.get("sop", "")
            }
        ))

print(f"Loaded {len(docs)} laws")

#2. Quick text cleaning
def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)   # remove excessive newlines
    text = re.sub(r' {2,}', ' ', text)        # remove excessive spaces
    return text.strip()

for doc in docs:
    doc.page_content = clean_text(doc.page_content)

#3. Chunk by člen
def chunk_by_clen(doc):
    # Split on article markers like "1. člen", "22. člen" etc.
    parts = re.split(r'(?=\n\d+\.\s*člen)', doc.page_content)
    chunks = []
    for part in parts:
        if len(part.strip()) > 50:  # skip tiny fragments
            chunks.append(Document(
                page_content=part.strip(),
                metadata=doc.metadata
            ))
    return chunks

all_chunks = []
for doc in docs:
    all_chunks.extend(chunk_by_clen(doc))

print(f"Total chunks: {len(all_chunks)}")
print(f"\nExample chunk from '{all_chunks[0].metadata['naziv']}':")
print(all_chunks[0].page_content[:300])

#4. Load embedding model
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cpu"},  # change to "cuda" if running on GPU
    encode_kwargs={"normalize_embeddings": True},
)
print("Embedding model loaded.")

#5. Build FAISS index
print("Embedding chunks and building FAISS index...")
vectorstore = FAISS.from_documents(all_chunks, embeddings)
print(f"Index built with {vectorstore.index.ntotal} vectors.")

#6. Save to disk
vectorstore.save_local("faiss_index")
print("Saved to faiss_index/")