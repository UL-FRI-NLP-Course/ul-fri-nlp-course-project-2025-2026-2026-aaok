import torch
import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging
import warnings
from RAG import Retriever_Full
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# 1. Choose retriever
# retriever = Retriever_Chunk()
retriever = Retriever_Full()

# 2. FORMAT CHUNKS
def format_docs(docs):
    return [
        {
            "source": d.metadata.get("naziv", "neznan"),
            "content": d.page_content
        }
        for d in docs
    ]

def build_context(chunks):
    return "\n\n---\n\n".join(
        f"[Vir: {c['source']}]\n{c['content']}"
        for c in chunks
    )


# 3. LOAD MODEL

# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("Loading Model...", MODEL_NAME)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     llm_int8_enable_fp32_cpu_offload=True
# )

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=bnb_config,
#     device_map="auto",
# )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

print("Model loaded.")

# 4. PROMPT
def build_prompt(context, question):
    system = (
        "You are a Slovenian legal assistant specializing in employment law.\n"
        "Answer the question in detail using ONLY the provided legal sources.\n"
        "Your answer must:\n"
        "- Explain the conditions and requirements clearly\n"
        "- Cite the specific law name and article number\n"
        "- Be written in Slovenian\n"
        "- Be at least 3-4 sentences long\n"
        "If the answer is not in the sources, say: 'Za to vprašanje vam priporočam posvet s pravnikom.'\n\n"
        f"SOURCES:\n{context}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# 5. SINGLE QUERY FUNCTION
def ask(question):
    docs = retriever.retrieve(question, "a", "rag_results.jsonl")
    chunks = format_docs(docs)
    context = build_context(chunks)

    prompt = build_prompt(context, question)

    output = pipe(
        prompt,
        max_new_tokens=500,
        temperature=0.3,
        do_sample=True,
    )

    full_text = output[0]["generated_text"]
    answer = full_text[len(prompt):].strip()

    return {
        "question": question,
        "answer": answer,
        "chunks": chunks
    }

# 6. EVALUATION SET
queries = [
    "Koliko dodatnega dopusta mi pripada nad 50 let v kovinski industriji?", # https://www.mojmalipravnik.net/index.php/objavljeni-odgovori/zaposlovanje/10022-koliko-dodatnega-dopusta-mi-pripada-nad-50-let-v-kovinski-industriji
    "Si po dopolnjenem 55 letu res starejši delavec in kakšne so tvoje pravice?", # https://mojmalipravnik.net/index.php/objavljeni-odgovori/zaposlovanje/9682-si-po-dopolnjenem-55-letu-res-starejsi-delavec-in-kaksne-so-tvoje-pravice
    "Delavcu v gostinstvu in turizmu res pripada 1 cel prosti vikend na mesec po zakonu?", # https://mojmalipravnik.net/index.php/objavljeni-odgovori/zaposlovanje/9609-delavcu-v-gostinstvu-in-turizmu-res-pripada-1-cel-prosti-vikend-na-mesec-po-zakonu
    "Katere so osnovne obveznosti delodajalca glede varnosti pri delu?",
    "Kako dolgo se hranijo evidence o delovnem času?",
    "Koliko znaša minimalna plača v Sloveniji?",
    # "Koliko minimalnega letnega dopusta pripada zaposlenemu?",
    # "Kdo je upravičen do denarnega nadomestila za brezposelnost?",
    # "Kako natančno mora biti evidentiran delovni čas zaposlenih?",
    # "Kakšna pooblastila ima inšpektor za delo pri nadzoru",
]

# 7. RUN EVAL
os.makedirs("results", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
for i, q in enumerate(queries):
    print(f"[{i+1}/{len(queries)}] Processing: {q}")

    result = ask(q)

    # add metadata
    result["query"] = q
    result["model"] = MODEL_NAME
    result["index"] = i
    result["timestamp"] = datetime.now().isoformat()

    safe_q = "".join(c if c.isalnum() or c in " _-" else "_" for c in q)[:60]

    filename = f"results/{MODEL_NAME.split('/')[-1]}_{timestamp}_{i:02d}_{safe_q}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved to {filename}\n")

    