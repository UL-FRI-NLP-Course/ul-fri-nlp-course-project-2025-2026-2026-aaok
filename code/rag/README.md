## RAG
Scripts related to generating the RAG vector index.

- `pisrs_tematsko_kazalo_delovno_pravo.csv` - collected documents from pisrs [tematsko kazalo](https://pisrs.si/zbirke/register-predpisov) under delovno pravo.

Steps for generating the index:
0) First download and unzip the COLESLAW 1.0 dataset, steps explained in `source/`

1) `pisrs.py` - collect all documents into one `pisrs.csv` and embed them into `doc_embs.npy`

2) `query_similarity.py` - embed queries and save similarity for each document for each query in `similarities.csv`

3) `filter_pisrs.py` - use `similarities.csv` to filter `pisrs.csv` into `filtered_pisrs.csv`

4) `build_index.py` - use `filtered_pisrs.csv` to chunk the documents and create FAISS index inside newly created folder `faiss_index`

5) `run_rag.py` - run simple rag test

You may also just skip these steps and unzip our pregenerated index inside `faiss_index/faiss_index.zip`.
