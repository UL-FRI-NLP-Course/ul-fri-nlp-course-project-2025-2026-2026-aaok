from langchain_core.documents import Document
import re
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

df = pd.read_csv("filtered_pisrs.csv", encoding="utf-8")

# zakon = df[df["naziv"] == "Zakon o delovnih razmerjih (ZDR-1)"]
# df = df.iloc[:10]
# df = pd.concat([zakon, df])
# print(df)

# df["clen_count"] = df["text"].str.lower().str.count("člen")
# counts = df["clen_count"].value_counts().sort_index()
# print(counts)
# df["word_count"] = df["text"].fillna("").str.split().str.len()
# row = df[df["clen_count"] == 1]
# print(row)

docs = []
for _, row in df.iterrows():
    docs.append(
        Document(
            page_content=row["text"],
            metadata={
                "naziv": row["naziv"],
                # "id":    row["id"],
                "sop":   row["sop"],
            }
        )
    )

print(f"Loaded {len(docs)} laws")

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)   # remove excessive newlines
    text = re.sub(r' {2,}', ' ', text)       # remove excessive spaces
    return text.strip()

for doc in docs:
    doc.page_content = clean_text(doc.page_content)

TARGET_SIZE = 800
MAX_SIZE = 1200
OVERLAP = 1

def split_by_clen(text):
    return re.split(r'(?=\n\d+\.\s*člen)', text)

def split_by_paragraphs(text):
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

def pack_paragraphs(paragraphs):
    chunks = []
    current = []

    current_size = 0

    for p in paragraphs:
        p_size = len(p)

        # if adding exceeds max → flush chunk
        if current_size + p_size > MAX_SIZE and current:
            chunks.append("\n\n".join(current))
            current = current[-OVERLAP:]  # overlap
            current_size = sum(len(x) for x in current)

        current.append(p)
        current_size += p_size

    if current:
        chunks.append("\n\n".join(current))

    return chunks

def add_title(chunk_text, doc):
    title = doc.metadata.get("naziv", "UNKNOWN TITLE")
    return f"TITLE: {title}\n\n{chunk_text}"

def chunk_document(doc):
    text = doc.page_content

    # try splitting by člen
    clen_parts = split_by_clen(text)

    # fallback if no člen structure
    if len(clen_parts) == 1:
        sections = [text]
    else:
        sections = clen_parts

    final_chunks = []

    for sec in sections:

        sec = sec.strip()
        if not sec:
            continue

        # if small enough -> keep as is
        if len(sec) <= MAX_SIZE:
            final_chunks.append(sec)
            continue

        # otherwise split by paragraphs
        paragraphs = split_by_paragraphs(sec)

        # pack into balanced chunks
        packed = pack_paragraphs(paragraphs)
        final_chunks.extend(packed)

    # convert to Document objects
    return [
        Document(
            page_content=add_title(c, doc), # add title to each chunk
            metadata=doc.metadata
        )
        for c in final_chunks
    ]

all_chunks = []
for doc in docs:
    all_chunks.extend(chunk_document(doc))

print(f"Total chunks: {len(all_chunks)}")
print(f"\nExample chunk from '{all_chunks[4].metadata['naziv']}':")
print(all_chunks[1].page_content)

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    # model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cuda"},  # "cuda" "cpu"
    encode_kwargs={"normalize_embeddings": True},
)
print("Embedding model loaded.")

print("Embedding chunks and building FAISS index...")
vectorstore = FAISS.from_documents(all_chunks, embeddings)
print(f"Index built with {vectorstore.index.ntotal} vectors.")

vectorstore.save_local("faiss_index")
print("Saved to faiss_index/")

