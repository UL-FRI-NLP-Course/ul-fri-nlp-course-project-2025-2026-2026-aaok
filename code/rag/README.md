## RAG
Scripts related to generating the required retrieval indexes.

`pisrs_tematsko_kazalo_delovno_pravo.csv` - manually collected documents from PISRS [tematsko kazalo](https://pisrs.si/zbirke/register-predpisov) under delovno pravo.

#### Steps for generating the indexes
You may also just skip steps 1-4 and start from 5) since `filtered_pisrs.csv` and `pisrs_metadata.jsonl` are already included.
```bash
# 0) Install the required modules by following the steps in code/

# 1) Download and unzip the COLESLAW 1.0 dataset
# Steps for that are explained in ../source/

# Make sure to run scripts inside this folder. 
cd code/rag/

# 2) Collect all documents into one pisrs.csv
#    and embed them into doc_embs.npy
python pisrs.py

# 3) Embed queries and save similarity for each document
#    for each query in similarities.csv
python query_similarity.py

# 4) Use similarities.csv to filter pisrs.csv
#    into filtered_pisrs.csv
python filter_pisrs.py

# 5) Read filtered_pisrs.csv and scrape metadata
#    like validity dates for each document and store
#    in pisrs_metadata.jsonl
python scrape_pisrs.py

# 6) Use filtered_pisrs.csv and pisrs_metadata.jsonl
#    to chunk the documents and create:
#      - FAISS indexes inside:
#          faiss_chunk_index/
#          faiss_doc_index/
#      - BM25 index:
#          bm25_chunk_index.pkl
python build_index.py

# 7) (Optional) Run a simple RAG test
python run_rag.py

# Return to code/ to run the evaluation
cd ..
```




