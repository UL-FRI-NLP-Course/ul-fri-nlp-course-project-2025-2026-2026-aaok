import json
import pandas as pd
from tqdm import tqdm
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import re
import torch
from rank_bm25 import BM25Okapi
import pickle
import re
from datetime import datetime, timezone

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

CSV_PATH = "data/filtered_pisrs.csv"
METADATA_PATH = "data/pisrs_metadata.jsonl"

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

print("Loading metadata JSONL...")

metadata_map = {}

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        moped_id = obj.get("mopedId")
        data = obj.get("data", {})
        inner = data.get("data", {})
        evid = inner.get("evidencniPodatki", inner)

        metadata_map[moped_id] = evid
print(f"Loaded metadata for {len(metadata_map)} documents")

print("Checking integrity...")
missing = []
for _, row in df.iterrows():
    moped_id = row["mopedId"]
    if moped_id not in metadata_map:
        missing.append(moped_id)

if missing:
    print("\nMISSING METADATA")
    print(f"Missing count: {len(missing)}")
    print("Example missing IDs:", missing[:10])
    print("\nYou need to run scrape_pisrs.py before building FAISS.")
    exit(1)

print("All documents have metadata. Proceeding...")

def to_ts(date_str):
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str)
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except:
        return None

def safe_get(meta, key):
    return meta.get(key) if meta else None

unified_docs = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    moped_id = row["mopedId"]
    meta = metadata_map[moped_id]

    sprejeto = safe_get(meta, "sprejeto")
    objavljeno = safe_get(meta, "objavljeno")
    velja_od = safe_get(meta, "veljaOd")
    velja_do = safe_get(meta, "veljaDo")
    uporablja_od = safe_get(meta, "uporabljaOd")
    uporablja_do = safe_get(meta, "uporabljaDo")
    vrsta_akta = safe_get(meta, "vrstaAkta")

    source_url = f"https://pisrs.si/pregledPredpisa?id={moped_id}"

    doc = {
        "id": row["id"],
        "naziv": row["naziv"],
        "sop": row["sop"],
        "mopedId": moped_id,
        "eva": row.get("eva"),
        "epa": row.get("epa"),
        "text": row["text"],

        "sprejeto": sprejeto,
        "objavljeno": objavljeno,
        "veljaOd": velja_od,
        "veljaDo": velja_do,
        "uporabljaOd": uporablja_od,
        "uporabljaDo": uporablja_do,
        "sprejeto_ts": to_ts(sprejeto),
        "objavljeno_ts": to_ts(objavljeno),
        "veljaOd_ts": to_ts(velja_od),
        "veljaDo_ts": to_ts(velja_do),
        "uporabljaOd_ts": to_ts(uporablja_od),
        "uporabljaDo_ts": to_ts(uporablja_do),
        "vrstaAkta": vrsta_akta,

        "source_url": source_url
    }

    unified_docs.append(doc)

TARGET_SIZE = 800
MAX_SIZE = 1200
OVERLAP = 1

def split_by_clen(text):
    return re.split(r'(?=\n\d+\.\s*člen)', text)

def split_by_paragraphs(text):
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def pack_paragraphs(paragraphs):
    chunks = []
    current = []
    current_size = 0

    for p in paragraphs:
        p_size = len(p)

        if current_size + p_size > MAX_SIZE and current:
            chunks.append("\n\n".join(current))
            current = current[-OVERLAP:]
            current_size = sum(len(x) for x in current)

        current.append(p)
        current_size += p_size

    if current:
        chunks.append("\n\n".join(current))

    return chunks

def add_title(chunk_text, doc):
    return f"TITLE: {doc['naziv']}\n\n{chunk_text}"

def chunk_document(doc, start_chunk_id):
    text = clean_text(doc["text"])

    clen_parts = split_by_clen(text)

    if len(clen_parts) == 1:
        sections = [text]
    else:
        sections = clen_parts

    final_chunks = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        if len(sec) <= MAX_SIZE:
            final_chunks.append(sec)
            continue

        paragraphs = split_by_paragraphs(sec)
        final_chunks.extend(pack_paragraphs(paragraphs))

    docs = []
    for i, c in enumerate(final_chunks):
        docs.append(
            Document(
                page_content=add_title(c, doc),
                metadata={
                    **doc,
                    "chunk_type": "legal_chunk",
                    "chunk_id": start_chunk_id + i
                }
            )
        )
    return docs

print("Chunking documents...")

all_chunks = []
chunk_id = 0
for doc in unified_docs:
    chunks = chunk_document(doc, chunk_id)
    all_chunks.extend(chunks)
    chunk_id += len(chunks)

print(f"Total chunks: {len(all_chunks)}")

print("Building document-level index...")

def build_doc_embedding_text(doc):
    return f"""
{doc['naziv']}

{doc['text'][:1200]}
"""

doc_docs = [
    Document(
        # page_content=f"{d['naziv']} {d['sop']} {d.get('vrstaAkta')} {d.get('eva')} {d.get('epa')}",
        page_content=build_doc_embedding_text(d),
        metadata={**d, "level": "document"}
    )
    for d in unified_docs
]

def tokenize(text):
    return  re.findall(r"[a-zA-ZčšžČŠŽ0-9]+", text.lower())

print("Building BM25 index...")

bm25_corpus = [tokenize(c.page_content) for c in all_chunks]
bm25 = BM25Okapi(bm25_corpus)
bm25_chunks = all_chunks

print("Saving document-level index...")

faiss_doc = FAISS.from_documents(doc_docs, embeddings)
faiss_doc.save_local("faiss_doc_index")

print("Saving chunk-level index...")
faiss_chunk = FAISS.from_documents(all_chunks, embeddings)
faiss_chunk.save_local("faiss_chunk_index")

print("Saving BM25 index...")

bm25_package = {
    "bm25": bm25,
    "chunks": bm25_chunks
}

with open("bm25_chunk_index.pkl", "wb") as f:
    pickle.dump(bm25_package, f)

print("BM25 saved.")
