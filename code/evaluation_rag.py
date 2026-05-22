
import torch
import json
import os
from datetime import datetime
import logging
import warnings
from RAG import Retriever_Full_best
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

open("rag_results.jsonl", "w").close()

# 1. Choose retriever
retriever = Retriever_Full_best()

# 2. EVALUATION SET
queries = [
    "Koliko dni letnega dopusta pripada delavcu?",
    "Katere so obveznosti delodajalca glede varnosti pri delu?",
    "Katere so pravice nosečnice pri delu?",
    "Koliko znaša minimalna plača v Sloveniji?",
    "Kakšne so pristojnosti inšpektorja za delo pri nadzoru?",
    "Kakšni so pogoji za sklenitev pogodbe o zaposlitvi za določen čas?",
    "Koliko dodatnega dopusta mi pripada nad 50 let v kovinski industriji?",
    "Si po dopolnjenem 55 letu res starejši delavec in kakšne so tvoje pravice?",
    "Delavcu v gostinstvu in turizmu res pripada 1 cel prosti vikend na mesec po zakonu?",
    "Imam pogodbo za določen čas. Koliko časa vnaprej mi mora delodajalec povedati, da je ne bo podaljšal?",
    "Ali mora delodajalec delavcu izplačati regres za letni dopust?",
    "Ali ima delavec pravico do plačanega odmora med delom?",
    "Ali se letni dopust lahko izrabi v več delih?",
    "Ali mora biti pogodba o zaposlitvi sklenjena v pisni obliki?",
    "Ali lahko delodajalec odpove pogodbo o zaposlitvi ustno?",
    "Koliko ur znaša polni delovni čas na teden?",
    "Koliko tednov znaša minimalni letni dopust?",
    "Koliko ur počitka mora imeti delavec med dvema delovnima dnevoma?",
    "Koliko ur odmora med delom pripada delavcu pri polnem delovnem času?",
    "Koliko dni ima delavec za izredno odpoved pogodbe?",
]

# 3. RUN
for i, q in enumerate(queries):
    print(f"[{i+1}/{len(queries)}] Processing: {q[:60]}...")
    retriever.retrieve(q, "a", "rag_results.jsonl")
    print(f"  -> done\n")

print("Finished. Saved to rag_results.jsonl")