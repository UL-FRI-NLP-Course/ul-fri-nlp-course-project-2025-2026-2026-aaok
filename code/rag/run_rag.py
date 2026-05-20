import pickle
import re
from datetime import datetime, timezone
from langchain_community.vectorstores import FAISS

class LegalRAGRetrieval:
    def __init__(
        self,
        embeddings,

        doc_index_path="faiss_doc_index",
        chunk_index_path="faiss_chunk_index",
        bm25_index_path="bm25_chunk_index.pkl",

        doc_faiss_enabled=False,
        chunk_faiss_enabled=False,
        bm25_enabled=False,
        query_rewrite_enabled=False,
        sop_expansion_enabled=False,
        time_filter_enabled=False,

        doc_k=5,
        chunk_k=10,
        bm_k=10,
        final_top_k=10,

        reranker=None,
        query_rewriter=None
    ):
        self.embeddings = embeddings

        self.doc_faiss_enabled = doc_faiss_enabled
        self.chunk_faiss_enabled = chunk_faiss_enabled
        self.bm25_enabled = bm25_enabled
        self.query_rewrite_enabled = query_rewrite_enabled
        self.sop_expansion_enabled = sop_expansion_enabled
        self.time_filter_enabled = time_filter_enabled

        self.doc_k = doc_k
        self.chunk_k = chunk_k
        self.bm_k = bm_k
        self.final_top_k = final_top_k

        self.reranker = reranker
        self.query_rewriter = query_rewriter

        self.chunk_id_map = {}

        self.tokenize = lambda t: re.findall(r"[a-zA-ZčšžČŠŽ0-9]+", t.lower())

        print("Loading FAISS doc index...")
        self.doc_store = FAISS.load_local(
            doc_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("Loading FAISS chunk index...")
        self.chunk_store = FAISS.load_local(
            chunk_index_path,
            embeddings,
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
    # QUERY PROCESSING
    # ------------------------------------------------------------
    def process_query(self, query):
        rewritten = None

        if self.query_rewrite_enabled and self.query_rewriter:
            rewritten = self.query_rewriter(query)

        return {
            "original": query,
            "rewritten": rewritten or query
        }

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
    def to_ts(date_str):
        if not date_str:
            return None
        try:
            dt = datetime.fromisoformat(date_str)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except:
            return None
            
    def from_ts(ts):
        if ts is None:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()

    def time_filter(self, chunks, as_of_ts="2024-12-31"):
        if not self.time_filter_enabled:
            return chunks

        as_of_ts = LegalRAGRetrieval.to_ts(as_of_ts)
        filtered = []
        for c in chunks:
            m = c.metadata
            start = m.get("uporabljaOd_ts") or m.get("veljaOd_ts")
            end = m.get("uporabljaDo_ts") or m.get("veljaDo_ts")

            # check validity window if exists
            if start or end:
                if start and as_of_ts < start:
                    continue
                if end and as_of_ts > end:
                    continue
                filtered.append(c)
                continue

            # fallback to publication time
            objavljeno = m.get("objavljeno_ts")
            sprejeto = m.get("sprejeto_ts")
            if objavljeno or sprejeto:
                if sprejeto and sprejeto > as_of_ts:
                    continue
                if objavljeno and objavljeno > as_of_ts:
                    continue
                filtered.append(c)
                continue

            # no time metadata = keep
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
    def retrieve(self, query, output_file=None):

        q = self.process_query(query)
        final_query = q["rewritten"]

        doc_results = self.retrieve_doc_chunks(final_query)
        chunk_results = self.retrieve_chunks(final_query)
        bm25_results = self.retrieve_bm25(final_query)
        
        # for c in doc_results:
        #     print(c.metadata["chunk_id"], c.metadata["mopedId"], c.metadata["naziv"])
        
        # for c in chunk_results:
        #     print(c.metadata["chunk_id"], c.metadata["mopedId"], c.metadata["naziv"])
        #     for cc in self.sop_map[c.metadata["sop"]]:
        #         print(" ", cc.metadata["chunk_id"], c.metadata["mopedId"], cc.metadata["naziv"])
        
        # for c in bm25_results:
        #     print(c.metadata["chunk_id"], c.metadata["mopedId"], c.metadata["naziv"])

        chunk_results = self.expand_sop(chunk_results)
        merged = self.merge(doc_results, chunk_results, bm25_results)
        merged = self.time_filter(merged)
        merged = self.rerank(final_query, merged)
        final = merged[:self.final_top_k]
        if output_file:
            self.dump_results(query, final, output_file)

        return final
    
    def serialize_chunks(self, chunks):
        return [
            {
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
            }
            for c in chunks
        ]
    
    def dump_results(self, query, chunks, output_file="rag_results.json"):
        entry = {
            "query": query,
            "doc_faiss_enabled": self.doc_faiss_enabled,
            "chunk_faiss_enabled": self.chunk_faiss_enabled,
            "bm25_enabled": self.bm25_enabled,
            "query_rewrite_enabled": self.query_rewrite_enabled,
            "sop_expansion_enabled": self.sop_expansion_enabled,
            "time_filter_enabled": self.time_filter_enabled,
            "doc_k": self.doc_k,
            "chunk_k": self.chunk_k,
            "bm_k": self.bm_k,
            "final_top_k": self.final_top_k,
            "results": self.serialize_chunks(chunks)
        }

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")

# ------------------------------------------------------------
# TEST RUN
# ------------------------------------------------------------
if __name__ == "__main__":

    import torch
    import json
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder

    print("Loading embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    print("Loading reranker...")

    reranker = CrossEncoder(
        "BAAI/bge-reranker-base",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Initializing retrieval system...")

    retrieval = LegalRAGRetrieval(
        embeddings=embeddings,

        doc_faiss_enabled=True,
        chunk_faiss_enabled=True,
        bm25_enabled=True,
        # sop_expansion_enabled=True,
        time_filter_enabled=True,

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

    results = retrieval.retrieve(question, output_file="rag_results.jsonl")

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

    