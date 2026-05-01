import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging
import warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# 1. LOAD FAISS (RAG retrieval)
print("Loading FAISS index...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.load_local(
    "rag/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
)

print("FAISS loaded.")

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


# 3. LOAD Model
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

# 5. RAG + LLM PIPELINE
def ask(question):

    # retrieve
    docs = retriever.invoke(question)
    chunks = format_docs(docs)
    context = build_context(chunks)

    # build prompt
    prompt = build_prompt(context, question)

    # generate
    output = pipe(
        prompt,
        max_new_tokens=500,
        temperature=0.3,
        do_sample=True,
    )

    full_text = output[0]["generated_text"]

    #extract only assistant answer
    if "<|start_header_id|>assistant<|end_header_id|>" in full_text:
        answer = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    elif "assistant" in full_text:
        answer = full_text.split("assistant")[-1]
    else:
        answer = full_text

    # clean up any remaining special tokens
    answer = answer.replace("<|eot_id|>", "").replace("<|end_header_id|>", "").strip()

    return answer, chunks

# 6. CHAT LOOP
print("\nRAG Chat ready. Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        print("Bye :)")
        break

    answer, chunks = ask(question)

    print("\nANSWER:\n")
    print(answer)
    #PRINT CHUNKS
    #print("\nRETRIEVED CHUNKS:\n")
    #print(json.dumps(chunks, ensure_ascii=False, indent=2))
    #print("\n" + "="*80 + "\n")


'''
Koliko minimalnega letnega dopusta pripada zaposlenemu?
Katere so osnovne obveznosti delodajalca glede varnosti pri delu?
Kdo je upravičen do denarnega nadomestila za brezposelnost?
Kako natančno mora biti evidentiran delovni čas zaposlenih?
Kakšna pooblastila ima inšpektor za delo pri nadzoru
'''