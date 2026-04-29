import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

queries = [ # 1
    "pogodba o zaposlitvi sklenitev pogoji delovno razmerje",
    "vrste pogodb o zaposlitvi določila pravice obveznosti",
    "odpoved pogodbe o zaposlitvi odpovedni rok prenehanje delovnega razmerja",
    "izredna odpoved delavec delodajalec razlogi za odpoved",
    "plača nadomestilo regres izplačilo plače minimalna plača",
    "zamuda pri izplačilu plače pravice delavca",
    "delovni čas nadure razpored dela počitek tedenski in dnevni",
    "organizacija delovnega časa nočno delo zakon omejitve",
    "dopust letni dopust bolniška odsotnost z dela pravice delavca",
    "pravice delavcev varstvo delavca diskriminacija mobing delovno mesto",
    "varnost in zdravje pri delu obveznosti delodajalca zaščita delavca",
    "zaposlovanje tujcev delovna dovoljenja pogoji za delo tujcev",
    "delo tujcev v Sloveniji omejitve in postopki",
    "obveznosti delodajalca pogodba o zaposlitvi varnost delovno mesto",
    "odgovornosti delodajalca sankcije kršitve delovnega prava",
    "kolektivna pogodba delovno pravo sindikati",
    "kolektivne pogodbe plače tarifni del",
    "aneks kolektivna pogodba spremembe",
    "javni sektor plače kolektivne pogodbe RTV zdravstvo",
    "pravilnik uredba zaposlovanje delovna razmerja",
    "uskladitev plač bruto minimalna mesečna plača odredba delavci",
    "bruto plača neto plača izračun plače davki prispevki",
]

query_embs = model.encode(queries, show_progress_bar=True)
query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)

print("reading pisrs.csv")
pisrs_all = pd.read_csv("pisrs.csv", encoding="utf-8")
print(pisrs_all)

doc_texts = pisrs_all["text"].tolist()
# doc_embs = model.encode(doc_texts, batch_size=64, show_progress_bar=True)
doc_embs = np.load("doc_embs.npy")

doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
sim_matrix = doc_embs @ query_embs.T

df_sim = pd.DataFrame(sim_matrix, columns=[f"sim_q{i+1}__{queries[i]}" for i in range(len(queries))])
df_sim["sop"] = pisrs_all["sop"]
df_sim["delovno_pravo"] = pisrs_all["delovno_pravo"]
df_sim["naziv"] = pisrs_all["naziv"]

sim_cols = [f"sim_q{i+1}__{queries[i]}" for i in range(len(queries))]
df_sim = df_sim[["sop", "naziv", "delovno_pravo"] + sim_cols]

df_sim.to_csv("similarities.csv", index=False)
print("Saved similarities.csv")



