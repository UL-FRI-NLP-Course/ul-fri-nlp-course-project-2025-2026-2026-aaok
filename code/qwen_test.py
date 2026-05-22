import torch
import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from RAG import Retriever_Full_best

QUESTION_FILE = "question.json"
OUTPUT_FILE = "answer.json"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# load question
with open(QUESTION_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

question = questions[0]["question"]
print(f"Question: {question}")

# retrieve
print("Retrieving...")
retriever = Retriever_Full_best()
docs = retriever.retrieve(question, "a", "rag_results_single.jsonl")
chunks = [{"naziv": d.metadata.get("naziv", "neznan"), "text": d.page_content} for d in docs]

# load model
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("Model loaded.")

def build_context(chunks):
    return "\n\n---\n\n".join(
        f"[Vir: {c.get('naziv', 'neznan')}]\n{c.get('text', '')}"
        for c in chunks[:10]
    )

def build_prompt(context, question):
    system = (
        "You are a Slovenian legal assistant specializing in employment law.\n"
        "Answer the question using ONLY the provided legal sources.\n"
        "Your answer must:\n"
        "- Explain the conditions and requirements clearly\n"
        "- Cite the specific law name and article number\n"
        "- Be written in Slovenian\n"
        "If the answer is not in the sources, say: 'Za to vprašanje vam priporočam posvet s pravnikom.'\n\n"
        f"SOURCES:\n{context}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

context = build_context(chunks)
prompt = build_prompt(context, question)

output = pipe(prompt, max_new_tokens=500, temperature=0.3, do_sample=True)
full_text = output[0]["generated_text"]
answer = full_text[len(prompt):].strip()

print(f"\nAnswer:\n{answer}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump({
        "question": question,
        "answer": answer,
        "chunks": chunks,
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat()
    }, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {OUTPUT_FILE}")