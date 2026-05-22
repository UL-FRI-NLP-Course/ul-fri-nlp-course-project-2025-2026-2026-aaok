import json
import os
from datetime import datetime

RAG_NAME = "Retriever_Full_test"  # change this for each run

RAG_RESULTS_FILE = "rag_results.jsonl"
QUESTIONS_FILE = "questions_for_rag.json"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
OUTPUT_FILE = f"results/rag_eval_{timestamp}.json"

# load questions
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

expected_map = {q["question"]: q for q in questions if "expected_sources" in q}

# load rag results
rag_results = []
with open(RAG_RESULTS_FILE, "r", encoding="utf-8") as f:
    content = f.read()

decoder = json.JSONDecoder()
pos = 0
while pos < len(content):
    content = content[pos:].lstrip()
    if not content:
        break
    try:
        obj, idx = decoder.raw_decode(content)
        rag_results.append(obj)
        pos = idx
    except json.JSONDecodeError:
        break

# BUILD RESULTS
results = []
total_sources = 0
correct_sources = 0
total_cleni = 0
correct_cleni = 0

for rag in rag_results:
    query = rag.get("query", "")
    chunks = rag.get("results", [])

    if query not in expected_map:
        continue

    exp = expected_map[query]
    expected_sources = exp.get("expected_sources", [])

    source_results = []
    for exp_source in expected_sources:
        expected_naziv = exp_source["source"]
        expected_cleni = exp_source.get("cleni", [])

        source_hit = any(c.get("naziv") == expected_naziv for c in chunks)
        total_sources += 1
        if source_hit:
            correct_sources += 1

        clen_results = {}
        for clen in expected_cleni:
            clen_hit = any(
                c.get("naziv") == expected_naziv and
                clen in c.get("text", "")
                for c in chunks
            )
            clen_results[clen] = clen_hit
            total_cleni += 1
            if clen_hit:
                correct_cleni += 1

        source_results.append({
            "expected_source": expected_naziv,
            "source_found": source_hit,
            "cleni": clen_results
        })

    results.append({
        "question": query,
        "source_results": source_results
    })

# build entry
source_score = f"{correct_sources}/{total_sources}"
clen_score = f"{correct_cleni}/{total_cleni}" if total_cleni > 0 else "n/a"

new_entry = {
    "rag": RAG_NAME,
    "sources_correct": source_score,
    "cleni_correct": clen_score,
    "details": results
}

# load existing output and append
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        all_results = json.load(f)
else:
    all_results = []

all_results.append(new_entry)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

# print summary
print(f"\n{'='*60}")
print(f"RAG: {RAG_NAME}")
print(f"Sources: {source_score} correct")
print(f"Členi:   {clen_score} correct")
print(f"{'='*60}\n")

for r in results:
    print(f"Q: {r['question'][:60]}...")
    for s in r["source_results"]:
        status = "✅" if s["source_found"] else "❌"
        print(f"  {status} {s['expected_source']}")
        for clen, hit in s["cleni"].items():
            c_status = "✅" if hit else "❌"
            print(f"      {c_status} {clen}")
    print()

print(f"Saved to {OUTPUT_FILE}")