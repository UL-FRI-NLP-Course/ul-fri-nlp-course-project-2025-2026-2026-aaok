import torch
import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
OUTPUT_FILE = f"results/qwen_answers_{timestamp}.json"
RAG_RESULTS_FILE = "rag_results.jsonl"
QUESTIONS_FILE = "questions_for_llm.json"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# load questions for expected answers
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)
expected_map = {q["question"]: q.get("expected_answer", "") for q in questions}

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

print(f"Loaded {len(rag_results)} queries from {RAG_RESULTS_FILE}")

# load model
print("Loading model...", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("Model loaded.")

def build_context(chunks):
    return "\n\n---\n\n".join(
        f"[Vir: {c.get('naziv', 'neznan')}]\n{c.get('text', c.get('content', ''))}"
        for c in chunks[:10]
    )

def build_prompt(context, question):
    system = (
        "You are a Slovenian legal assistant specializing in employment law.\n"
        "Answer the question in detail using ONLY the provided legal sources.\n"
        "Your answer must:\n"
        "- Explain the conditions and requirements clearly\n"
        "- Cite the specific law name and article number\n"
        "- Be written in Slovenian\n"
        "- Answer shortly in 2-3 sentences\n"
        "If the answer is not in the sources, say: 'Za to vprašanje vam priporočam posvet s pravnikom.'\n\n"
        f"SOURCES:\n{context}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# build rag lookup map
rag_map = {r.get("query", ""): r for r in rag_results}

# run - iterate over questions, look up rag results
results = []
for i, q in enumerate(questions):
    question = q["question"]

    if question not in rag_map:
        print(f"[{i+1}/{len(questions)}] SKIPPED (no RAG result): {question[:60]}")
        continue

    chunks = rag_map[question].get("results", [])
    print(f"[{i+1}/{len(questions)}] {question[:60]}...")

    context = build_context(chunks)
    prompt = build_prompt(context, question)

    output = pipe(prompt, max_new_tokens=500, temperature=0.3, do_sample=True)
    full_text = output[0]["generated_text"]
    answer = full_text[len(prompt):].strip()

    results.append({
        "question": question,
        "expected_answer": q.get("expected_answer", ""),
        "answer": answer,
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat()
    })

    print(f" -> {answer[:80]}...\n")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nDone. Saved {len(results)} answers to {OUTPUT_FILE}")