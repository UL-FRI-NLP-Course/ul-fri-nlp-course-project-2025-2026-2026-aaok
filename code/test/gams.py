import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
import warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ── 1. LOAD EVALUATION FILE ───────────────────────────────────────
INPUT_FILE = "answers.json"
MODEL_KEY = "gams"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

print(f"Loaded {len(results)} questions from {INPUT_FILE}")

# ── 2. FORMAT CHUNKS ──────────────────────────────────────────────
def build_context(chunks):
    return "\n\n---\n\n".join(
        f"[Vir: {c['source']}]\n{c['content']}"
        for c in chunks
    )

# ── 3. LOAD GaMS ──────────────────────────────────────────────────
MODEL_NAME = "cjvt/GaMS-9B-Instruct"
print("Loading GaMS...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
print("GaMS loaded.")

# ── 4. PROMPT ─────────────────────────────────────────────────────
def build_prompt(context, question):
    system = (
        "Si pravni pomočnik za slovensko delovno pravo.\n"
        "Odgovori na vprašanje na podlagi spodnjih virov.\n"
        "Če vir vsebuje relevantne informacije, jih VEDNO uporabi in navedi.\n"
        "Samo če viri sploh ne vsebujejo relevantnih informacij, reci: "
        "'Za to vprašanje vam priporočam posvet s pravnikom.'\n\n"
        f"VIRI:\n{context}"
    )
    messages = [
        {"role": "user", "content": f"{system}\n\nVprašanje: {question}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

# ── 5. GENERATE ───────────────────────────────────────────────────
for i, item in enumerate(results):
    question = item["question"]
    chunks = item["chunks"]
    print(f"[{i+1}/{len(results)}] {question}")

    context = build_context(chunks)
    prompt = build_prompt(context, question)

    output = pipe(prompt, max_new_tokens=500, do_sample=False)
    full_text = output[0]["generated_text"]
    answer = full_text[len(prompt):].strip()

    item["answers"][MODEL_KEY] = answer
    print(f"Done.\n")

# ── 6. SAVE ───────────────────────────────────────────────────────
with open(INPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved answers to {INPUT_FILE}")