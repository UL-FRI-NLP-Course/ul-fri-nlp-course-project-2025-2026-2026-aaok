import pickle
import re
from datetime import datetime, timezone
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import json

class Retriever:
    def __init__(
        self,
        doc_index_path="faiss_doc_index",
        chunk_index_path="faiss_chunk_index",
        bm25_index_path="bm25_chunk_index.pkl",

        doc_faiss_enabled=False,
        chunk_faiss_enabled=False,
        bm25_enabled=False,
        sop_expansion_enabled=False,
        time_filter_enabled=False,
        add_year_to_query=False,

        doc_k=5,
        chunk_k=10,
        bm_k=10,
        final_top_k=10,

        reranker=None,
    ):
        self.doc_faiss_enabled = doc_faiss_enabled
        self.chunk_faiss_enabled = chunk_faiss_enabled
        self.bm25_enabled = bm25_enabled
        self.sop_expansion_enabled = sop_expansion_enabled
        self.time_filter_enabled = time_filter_enabled
        self.add_year_to_query = add_year_to_query
        self.doc_k = doc_k
        self.chunk_k = chunk_k
        self.bm_k = bm_k
        self.final_top_k = final_top_k
        self.reranker = reranker

        self.tokenize = lambda t: re.findall(r"[a-zA-ZčšžČŠŽ0-9]+", t.lower())

        print("Loading embeddings...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

        print("Loading FAISS doc index...")
        self.doc_store = FAISS.load_local(
            doc_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        print("Loading FAISS chunk index...")
        self.chunk_store = FAISS.load_local(
            chunk_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.bm25 = None
        self.bm25_chunks = None

        if self.bm25_enabled and bm25_index_path:
            with open(bm25_index_path, "rb") as f:
                data = pickle.load(f)

            self.bm25 = data["bm25"]
            self.bm25_chunks = data["chunks"]

        self.sop_map = {}
        self._build_sop_map()

    # ------------------------------------------------------------
    # Build SOP map
    # ------------------------------------------------------------
    def _build_sop_map(self):
        print("Building SOP map...")

        for doc in self.chunk_store.docstore._dict.values():
            sop = doc.metadata.get("sop")
            if sop:
                self.sop_map.setdefault(sop, []).append(doc)
            
    # ------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------
    def retrieve_doc_chunks(self, query):
        if not self.doc_faiss_enabled:
            return []

        docs = self.doc_store.similarity_search(query, k=self.doc_k)
        sops = set()
        for d in docs:
            sop = d.metadata.get("sop")
            if sop:
                sops.add(sop)

        chunks = []
        for sop in sops:
            if sop in self.sop_map:
                chunks.extend(self.sop_map[sop])
        return chunks

    def retrieve_chunks(self, query):
        if not self.chunk_faiss_enabled:
            return []
        return self.chunk_store.similarity_search(query, k=self.chunk_k)

    def retrieve_bm25(self, query):
        if not self.bm25_enabled or self.bm25 is None:
            return []

        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.bm_k]
        return [self.bm25_chunks[i] for i in top_idx]

    # ------------------------------------------------------------
    # SOP expansion
    # ------------------------------------------------------------
    def expand_sop(self, chunks):
        if not self.sop_expansion_enabled:
            return chunks

        expanded = list(chunks)

        sops = set()
        for c in chunks:
            sop = c.metadata.get("sop")
            if sop:
                sops.add(sop)

        for sop in sops:
            if sop in self.sop_map:
                expanded.extend(self.sop_map[sop])

        return expanded

    # ------------------------------------------------------------
    # TIME FILTER
    # ------------------------------------------------------------
    def detect_time_window(self, query):
        years = list(
            y for y in map(int, re.findall(r"\b\d{4}\b", query))
            if 1991 <= y <= 2100
        )

        # default: dataset cutoff
        if not years:
            return "2024-12-31", "2024-12-31", ["2024"]

        # if any year is beyond dataset
        if min(years) > 2024:
            return "2024-12-31", "2024-12-31"

        # normal case
        start_year = min(years)
        end_year = max(years)

        return f"{start_year}-01-01", f"{end_year}-12-31", years

    def to_ts(self, date_str):
        if not date_str:
            return None
        try:
            dt = datetime.fromisoformat(date_str)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except:
            return None
            
    def from_ts(self, ts):
        if ts is None:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()

    def time_filter(self, chunks, query_start="2024-12-31", query_end="2024-12-31"):
        if not self.time_filter_enabled:
            return chunks

        query_start_ts = self.to_ts(query_start)
        query_end_ts = self.to_ts(query_end)
        filtered = []
        for c in chunks:
            m = c.metadata

            # PRIORITY 1: uporablja
            start = m.get("uporabljaOd_ts")
            end = m.get("uporabljaDo_ts")

            # PRIORITY 2: velja
            # only if both uporablja fields absent
            if start is None and end is None:
                start = m.get("veljaOd_ts")
                end = m.get("veljaDo_ts")

            # PRIORITY 3: objavljeno / sprejeto
            # only if both velja fields absent too
            if start is None and end is None:
                start = m.get("objavljeno_ts") or m.get("sprejeto_ts")
                end = None

            # no dates = keep
            if start is None and end is None:
                filtered.append(c)
                continue

            # check overlap
            doc_start = start if start is not None else float("-inf")
            doc_end = end if end is not None else float("inf")
            overlaps = (
                doc_start <= query_end_ts and
                doc_end >= query_start_ts
            )

            if overlaps:
                filtered.append(c)

        return filtered
    
    # ------------------------------------------------------------
    # MERGE + DEDUP
    # ------------------------------------------------------------
    def merge(self, doc_results, chunk_results, bm25_results):
        seen = set()
        merged = []
        for item in (doc_results + chunk_results + bm25_results):
            key = item.metadata.get("chunk_id")

            if key not in seen:
                seen.add(key)
                merged.append(item)

        return merged

    # ------------------------------------------------------------
    # RERANK
    # ------------------------------------------------------------
    def rerank(self, query, chunks):
        if not self.reranker or not chunks:
            return chunks

        pairs = [(query, c.page_content) for c in chunks]
        scores = self.reranker.predict(pairs)

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [c for c, _ in scored]

    # ------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------
    def retrieve(self, _query, output_mode="w", output_file=None):
        query = _query
        start_date, end_date, years = self.detect_time_window(query)

        if self.add_year_to_query and len(years) == 1:
            query += f" {years[0]}"

        doc_results = []
        chunk_results = []
        bm25_results = []

        if self.doc_faiss_enabled:
            doc_results = self.retrieve_doc_chunks(query)

        if self.chunk_faiss_enabled:
            chunk_results = self.retrieve_chunks(query)

        if self.bm25_enabled:
            bm25_results = self.retrieve_bm25(query)

        if self.sop_expansion_enabled:
            chunk_results = self.expand_sop(chunk_results)

        merged = self.merge(doc_results, chunk_results, bm25_results)

        if self.time_filter_enabled:
            merged = self.time_filter(merged, start_date, end_date)

        merged = self.rerank(query, merged)
        final_chunks = merged[:self.final_top_k]

        if output_file:
            self.dump_results(_query, query, final_chunks, output_mode, output_file)

        return final_chunks
    
    def serialize_chunks(self, chunks):
        return [{
                "chunk_id": c.metadata.get("chunk_id"),
                "mopedId": c.metadata.get("mopedId"),
                "naziv": c.metadata.get("naziv"),
                "sop": c.metadata.get("sop"),
                "text": c.page_content,
                "veljaOd": c.metadata.get("veljaOd"),
                "veljaDo": c.metadata.get("veljaDo"),
                "uporabljaOd": c.metadata.get("uporabljaOd"),
                "uporabljaDo": c.metadata.get("uporabljaDo"),
                "sprejeto": c.metadata.get("sprejeto"),
                "objavljeno": c.metadata.get("objavljeno"),
            } for c in chunks]
    
    def dump_results(self, og_query, query, chunks, output_mode="w", output_file="rag_results.json"):
        entry = {
            "query": og_query,
            "edited_query": query,
            "doc_faiss_enabled": self.doc_faiss_enabled,
            "chunk_faiss_enabled": self.chunk_faiss_enabled,
            "bm25_enabled": self.bm25_enabled,
            "sop_expansion_enabled": self.sop_expansion_enabled,
            "time_filter_enabled": self.time_filter_enabled,
            "doc_k": self.doc_k,
            "chunk_k": self.chunk_k,
            "bm_k": self.bm_k,
            "final_top_k": self.final_top_k,
            "results": self.serialize_chunks(chunks),
        }

        with open(output_file, output_mode, encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")
        
        print(f"Saved results to {output_file}")

# ------------------------------------------------------------
# TEST RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    from sentence_transformers import CrossEncoder

    print("Loading reranker...")

    reranker = CrossEncoder(
        "BAAI/bge-reranker-base",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Initializing retrieval system...")

    retrieval = Retriever(
        doc_faiss_enabled=True,
        chunk_faiss_enabled=True,
        bm25_enabled=True,
        sop_expansion_enabled=True,
        time_filter_enabled=True,
        add_year_to_query=True,

        doc_k=10,
        chunk_k=100,
        bm_k=100,

        reranker=reranker
    )

    # --------------------------------------------------------
    # TEST QUESTION
    # --------------------------------------------------------
    question = "Koliko znaša minimalna plača v Sloveniji?"

    print("\nQUESTION:")
    print(question)

    print("\nRunning retrieval...\n")

    results = retrieval.retrieve(question, output_mode="w", output_file="rag_results.json")

    print(f"Retrieved {len(results)} chunks.\n")

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------
    for i, doc in enumerate(results):
        print("naziv", doc.metadata["naziv"]) 
        print("  sprejeto", doc.metadata["sprejeto"])
        print("  objavljeno", doc.metadata["objavljeno"])
        print("  veljaOd", doc.metadata["veljaOd"])
        print("  veljaDo", doc.metadata["veljaDo"])
        print("  uporabljaOd", doc.metadata["uporabljaOd"])
        print("  uporabljaDo", doc.metadata["uporabljaDo"])

    