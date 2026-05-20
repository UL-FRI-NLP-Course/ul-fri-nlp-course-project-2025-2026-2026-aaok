## RAG
Scripts related to generating the RAG vector index.

- `pisrs_tematsko_kazalo_delovno_pravo.csv` - collected documents from pisrs [tematsko kazalo](https://pisrs.si/zbirke/register-predpisov) under delovno pravo.

Steps for generating the index:
0) First download and unzip the COLESLAW 1.0 dataset, steps explained in `source/`

1) `pisrs.py` - collect all documents into one `pisrs.csv` and embed them into `doc_embs.npy`

2) `query_similarity.py` - embed queries and save similarity for each document for each query in `similarities.csv`

3) `filter_pisrs.py` - use `similarities.csv` to filter `pisrs.csv` into `filtered_pisrs.csv`

4) `scrape_pisrs.py` - read `filtered_pisrs.csv` and scrape metadata like validity dates for each document and store in `pisrs_metadata.jsonl`.

5) `build_index.py` - use `filtered_pisrs.csv` and `pisrs_metadata.jsonl` to chunk the documents and create FAISS index inside newly created folder `faiss_chunk_index/` and `faiss_doc_index/` and a BM25 index `bm25_chunk_index.pkl`.

6) `run_rag.py` - run simple rag test

You may also just skip most steps and start from 5) since `filtered_pisrs.csv` and `pisrs_metadata.jsonl` are already included.



