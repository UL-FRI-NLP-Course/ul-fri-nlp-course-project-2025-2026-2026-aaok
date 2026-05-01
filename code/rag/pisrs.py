import pandas as pd
import json

import numpy as np
from sentence_transformers import SentenceTransformer

def to_df(path):
    file = open(path, "r", encoding="utf-8")
    entries = []
    for line in file.readlines():
        entry = json.loads(line)
        entries.append(entry)
    return pd.DataFrame(entries)

files = [
    "../source/COLESLAW 1.0/PISRS/register-predpisov.jsonl",
    "../source/COLESLAW 1.0/PISRS/drugi-splosni-in-posamicni-akti.jsonl",
    "../source/COLESLAW 1.0/PISRS/evidenca-normodajalcev.jsonl",
    # "../source/COLESLAW 1.0/PISRS/neveljavni-predpisi.jsonl",
    # "../source/COLESLAW 1.0/PISRS/obsoletni-in-konzumirani-predpisi.jsonl",
    # "../source/COLESLAW 1.0/PISRS/predpisi-v-pripravi.jsonl",
    "../source/COLESLAW 1.0/PISRS/splosni-akti-za-izvrsevanje-javnih-pooblastil.jsonl"
]

dfs = [to_df(f) for f in files]
pisrs = pd.concat(dfs, ignore_index=True)
pisrs = pisrs.drop_duplicates(subset='sop', keep='first')

delovno_pravo = pd.read_csv("pisrs_tematsko_kazalo_delovno_pravo.csv")
pisrs["delovno_pravo"] = pisrs["sop"].isin(delovno_pravo["SOP"])

pisrs.to_csv("pisrs.csv", index=False, encoding="utf-8")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

doc_texts = pisrs["text"].fillna("").tolist()
doc_embs = model.encode(doc_texts, batch_size=64, show_progress_bar=True, device="cuda")
doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
np.save("doc_embs.npy", doc_embs)



# missing = delovno_pravo[~delovno_pravo["SOP"].isin(pisrs["sop"])]
# print(missing)


