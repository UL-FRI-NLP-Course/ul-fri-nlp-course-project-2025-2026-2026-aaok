import torch
from sentence_transformers import CrossEncoder
from rag.retriever import Retriever

DOC_INDEX_PATH="rag/faiss_doc_index"
CHUNK_INDEX_PATH="rag/faiss_chunk_index"
BM25_INDEX_PATH="rag/bm25_chunk_index.pkl"

print("Loading reranker...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# class placeholder_with_all_params_off(Retriever):
#     def __init__(self):
#         super().__init__(
#             doc_index_path=DOC_INDEX_PATH,
#             chunk_index_path=CHUNK_INDEX_PATH,
#             bm25_index_path=BM25_INDEX_PATH,
#             doc_faiss_enabled=False,
#             doc_k=999,
#             chunk_faiss_enabled=False,
#             chunk_k=999,
#             bm25_enabled=False,
#             bm_k=999,
#             sop_expansion_enabled=False,
#             time_filter_enabled=False,
#             add_year_to_query=False,
#             final_top_k=999,
#             reranker=None
#         )

# Basic indexing rags
class Retriever_Chunk(Retriever):
    def __init__(self):
        super().__init__(
            doc_index_path=DOC_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            bm25_index_path=BM25_INDEX_PATH,
            chunk_faiss_enabled=True,
            chunk_k=20,
            final_top_k=10,
            reranker=None
        )

class Retriever_Chunk_with_year(Retriever):
    def __init__(self):
        super().__init__(
            doc_index_path=DOC_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            bm25_index_path=BM25_INDEX_PATH,
            chunk_faiss_enabled=True,
            chunk_k=20,
            final_top_k=10,
            add_year_to_query=True,
            reranker=None
        )

class Retriever_Chunk_with_year_SOPexpand(Retriever):
    def __init__(self):
        super().__init__(
            doc_index_path=DOC_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            bm25_index_path=BM25_INDEX_PATH,
            chunk_faiss_enabled=True,
            chunk_k=20,
            sop_expansion_enabled=True,
            final_top_k=10,
            add_year_to_query=True,
            reranker=None
        )

class Retriever_Doc(Retriever):
    def __init__(self):
        super().__init__(
            doc_index_path=DOC_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            bm25_index_path=BM25_INDEX_PATH,
            doc_faiss_enabled=True,
            doc_k=5,
            final_top_k=10,
            reranker=None
        )

class Retriever_Doc_with_year(Retriever):
    def __init__(self):
        super().__init__(
            doc_index_path=DOC_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            bm25_index_path=BM25_INDEX_PATH,
            doc_faiss_enabled=True,
            doc_k=5,
            final_top_k=10,
            add_year_to_query=True,
            reranker=None
        )

class Retriever_BM25(Retriever):
    def __init__(self):
        super().__init__(
            doc_index_path=DOC_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            bm25_index_path=BM25_INDEX_PATH,
            bm25_enabled=True,
            bm_k=20,
            final_top_k=10,
            reranker=None
        )

class Retriever_BM25_with_year(Retriever):
    def __init__(self):
        super().__init__(
            doc_index_path=DOC_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            bm25_index_path=BM25_INDEX_PATH,
            bm25_enabled=True,
            bm_k=20,
            final_top_k=10,
            add_year_to_query=True,
            reranker=None
        )

# combining indexing rags + reranking
class Retriever_Chunk_Doc(Retriever):
    def __init__(self):
        super().__init__(
            doc_index_path=DOC_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            bm25_index_path=BM25_INDEX_PATH,
            chunk_faiss_enabled=True,
            chunk_k=20,
            doc_faiss_enabled=True,
            doc_k=5,
            final_top_k=10,
            add_year_to_query=True,
            reranker=reranker
        )

class Retriever_Chunk_BM25(Retriever):
    def __init__(self):
        super().__init__(
            doc_index_path=DOC_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            bm25_index_path=BM25_INDEX_PATH,
            chunk_faiss_enabled=True,
            chunk_k=20,
            bm25_enabled=True,
            bm_k=20,
            final_top_k=10,
            add_year_to_query=True,
            reranker=reranker
        )

class Retriever_Doc_BM25(Retriever):
    def __init__(self):
        super().__init__(
            doc_index_path=DOC_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            bm25_index_path=BM25_INDEX_PATH,
            doc_faiss_enabled=True,
            doc_k=5,
            bm25_enabled=True,
            bm_k=20,
            final_top_k=10,
            add_year_to_query=True,
            reranker=reranker
        )

class Retriever_Full(Retriever):
    def __init__(self):
        super().__init__(
            doc_index_path=DOC_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            bm25_index_path=BM25_INDEX_PATH,
            doc_faiss_enabled=True,
            chunk_faiss_enabled=True,
            bm25_enabled=True,
            sop_expansion_enabled=True,
            time_filter_enabled=True,
            doc_k=10,
            chunk_k=100,
            bm_k=100,
            final_top_k=10,
            add_year_to_query=True,
            reranker=reranker
        )
class Retriever_Full_best(Retriever):
    def __init__(self):
        super().__init__(
            doc_index_path=DOC_INDEX_PATH,
            chunk_index_path=CHUNK_INDEX_PATH,
            bm25_index_path=BM25_INDEX_PATH,
            doc_faiss_enabled=False,
            chunk_faiss_enabled=True,
            bm25_enabled=True,
            sop_expansion_enabled=False,
            time_filter_enabled=True,
            chunk_k=100,
            bm_k=100,
            final_top_k=10,
            add_year_to_query=True,
            reranker=None
        )

# EXAMPLE
if __name__ == "__main__":
    retrieval = Retriever_Full()
    question = "Koliko znaša minimalna plača v Sloveniji?"

    results = retrieval.retrieve(
        question,
        output_file="rag_results.json"
    )

    print(f"\nRetrieved {len(results)} chunks.\n")
    
    for doc in results:
        print(doc.metadata.get("naziv"))



