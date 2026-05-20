import json

INPUT_FILE = "answers.json"
QUESTIONS_FILE = "questions.json"
OUTPUT_FILE = "evaluationRAG_results.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    answers = json.load(f)

with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

expected = {q["question"]: q for q in questions}

results = []

for item in answers:
    question = item["question"]
    chunks = item["chunks"]

    if question not in expected or "expected_source" not in expected[question]:
        continue

    exp = expected[question]
    expected_source = exp["expected_source"]
    expected_clen = exp.get("expected_clen", "")

    source_hit = any(
        c["source"] == expected_source and
        expected_clen in c["content"]
        for c in chunks
    )

    retrieved_sources = [c["source"] for c in chunks]

    results.append({
        "question": question,
        "expected_source": expected_source,
        "expected_clen": expected_clen,
        "retrieved_chunks": chunks,
        "retrieval_correct": source_hit
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

correct = sum(1 for r in results if r["retrieval_correct"])
print(f"\nRetrieval: {correct}/{len(results)} correct")
print(f"Saved to {OUTPUT_FILE}")