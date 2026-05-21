import json
from datetime import datetime
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class Retrieval:
    def __init__(
        self,
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        # faiss
        faiss_path="faiss_index",
        # retrieval params
        search_type="mmr",
        k=6,
        fetch_k=40,
        lambda_mult=0.6,
        # reranker
        use_reranker=False,
        reranker_model="BAAI/bge-reranker-base",
        reranker_top_k=5,
        device=None,
    ):
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult

        self.use_reranker = use_reranker
        self.reranker_top_k = reranker_top_k

        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print("Loading embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={
                "device": self.device
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

        print("Loading FAISS index...")
        self.vectorstore = FAISS.load_local(
            faiss_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
            },
        )
        print("FAISS loaded.")

        self.reranker = None
        if self.use_reranker:
            print("Loading reranker...")

            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(
                reranker_model,
                device=self.device
            )

            print("Reranker loaded.")
            
    def format_docs(self, docs):
        return [
            {
                "source": d.metadata.get("naziv", "neznan"),
                "content": d.page_content
            }
            for d in docs
        ]
    
    def rerank(self, question, docs):
        pairs = [(question, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in ranked[:self.reranker_top_k]]
        return reranked_docs

    def retrieve(self, question):
        docs = self.retriever.invoke(question)
        if self.use_reranker:
            docs = self.rerank(question, docs)
        return docs

    def retrieve_formatted(self, question):
        docs = self.retrieve(question)
        return self.format_docs(docs)

    def retrieve_and_save(self, question, output_file="rag_output.json"):
        docs = self.retrieve(question)
        formatted_docs = self.format_docs(docs)

        output = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "retrieval_config": {
                "k": self.k,
                "fetch_k": self.fetch_k,
                "lambda_mult": self.lambda_mult,
                "use_reranker": self.use_reranker,
                "reranker_top_k": self.reranker_top_k,
            },
            "chunks": formatted_docs
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Saved retrieval output to {output_file}")
        return output

retrieval = Retrieval(
    k=10,
    fetch_k=40,
    lambda_mult=0.6,

    use_reranker=False,
    # use_reranker=True,
    reranker_top_k=10,
)

question = "Koliko minimalnega letnega dopusta pripada zaposlenemu?"
question = "Ali me lahko šef tepe?"
question = "Se lahko študentje zaposlijo v več delovnih mestih naenkrat?"
question = "Koliko dodatnega dopusta mi pripada nad 50 let v kovinski industriji?"
question = "Si po dopolnjenem 55 letu res starejši delavec in kakšne so tvoje pravice?"
question = "Delavcu v gostinstvu in turizmu res pripada 1 cel prosti vikend na mesec po zakonu?"
question = "Katere so osnovne obveznosti delodajalca glede varnosti pri delu?"
question = "Kako dolgo se hranijo evidence o delovnem času?"
question = "Koliko znaša minimalna plača v Sloveniji? 2024"

results = retrieval.retrieve_formatted(question)

print(json.dumps(results, ensure_ascii=False, indent=2))




