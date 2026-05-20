import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import logging
import warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ── 1. LOAD EVALUATION FILE ───────────────────────────────────────
INPUT_FILE = "answers.json"
MODEL_KEY = "llama"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

print(f"Loaded {len(results)} questions from {INPUT_FILE}")

# ── 2. FORMAT CHUNKS ──────────────────────────────────────────────
def build_context(chunks):
    return "\n\n---\n\n".join(
        f"[Vir: {c['source']}]\n{c['content']}"
        for c in chunks
    )

# ── 3. LOAD LLAMA ─────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
print("Loading LLaMA...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
print("LLaMA loaded.")

# ── 4. PROMPT ─────────────────────────────────────────────────────
def build_prompt(context, question):
    system = (
        "You are a Slovenian legal assistant specializing in employment law.\n"
        "Answer the question concisely in 2-3 sentences using ONLY the provided legal sources.\n"
        "Your answer must:\n"
        "- Cite the specific law name and article number\n"
        "- Be written in Slovenian\n"
        "If the answer is not in the sources, say: 'Za to vprašanje vam priporočam posvet s pravnikom.'\n\n"
        f"SOURCES:\n{context}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
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

    output = pipe(
        prompt,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True,
    )

    full_text = output[0]["generated_text"]

    if "<|start_header_id|>assistant<|end_header_id|>" in full_text:
        answer = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    elif "assistant" in full_text:
        answer = full_text.split("assistant")[-1]
    else:
        answer = full_text

    answer = answer.replace("<|eot_id|>", "").replace("<|end_header_id|>", "").strip()

    item["answers"][MODEL_KEY] = answer
    print(f"Done.\n")

# ── 6. SAVE ───────────────────────────────────────────────────────
with open(INPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved answers to {INPUT_FILE}")