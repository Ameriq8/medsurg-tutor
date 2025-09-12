#!/usr/bin/env python3
import re
import os, logging, argparse, sqlite3
import faiss, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- config ----
DB_PATH = "textbook.db"
INDEX_PATH = "indexes/textbook.faiss"
EMB_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # small, stable
CHUNK_CHAR_LIMIT = 800
MAX_NEW_TOKENS = 256
SAFETY_MARGIN = 512

# quiet transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---- helpers ----
def clean(s: str) -> str:
    s = s.replace("", "- ").replace("\u2022", "- ").replace("\t", " ")
    s = s.encode("utf-8", "ignore").decode("utf-8")
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())

def connect_db(): return sqlite3.connect(DB_PATH)
def load_index(): return faiss.read_index(INDEX_PATH)

def embed(q: str, model: SentenceTransformer):
    return model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def fetch_chunks(conn, ids):
    if not ids: return []
    qmarks = ",".join(["?"] * len(ids))
    cur = conn.cursor()
    cur.execute(f"""
        SELECT c.id, c.text, c.chunk_index, d.title, d.path
        FROM chunks c JOIN documents d ON c.document_id = d.id
        WHERE c.id IN ({qmarks})
    """, [int(x) for x in ids])
    m = {row[0]: row for row in cur.fetchall()}
    return [m[i] for i in ids if i in m]

def get_budget(tokenizer):
    ml = getattr(tokenizer, "model_max_length", 4096) or 4096
    if ml > 100000: ml = 4096
    return max(1024, ml - SAFETY_MARGIN)

def build_messages(context: str, question: str):
    system = (
        "You are a factual RAG assistant. Answer ONLY from the given context. "
        "If the answer is not in context, say: \"I don't know from the textbook.\" "
        "Return concise Markdown with sections: **Summary**, **Key points**, **Sources** "
        "(list the [numbers] of used chunks)."
    )
    user = f"<context>\n{context}\n</context>\n\nQuestion: {question}\nAnswer:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def pack_context(rows, tokenizer, question):
    budget = get_budget(tokenizer)
    packed, used = [], []
    for i, (_, text, chunk_idx, title, path) in enumerate(rows, 1):
        t = clean(text)[:CHUNK_CHAR_LIMIT]
        block = f"[{i}] ({title} · chunk {chunk_idx})\n{t}"
        trial = "\n\n---\n\n".join(packed + [block])
        msgs = build_messages(trial, question)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        n_tokens = len(tokenizer(prompt).input_ids)
        if n_tokens <= budget:
            packed.append(block); used.append(i)
        else:
            break
    return "\n\n---\n\n".join(packed), used

def generate_answer(model, tokenizer, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,   # ensures .sequences exists
        )

    # handle both dict and tensor returns
    seq = out.sequences if hasattr(out, "sequences") else out[0].unsqueeze(0)

    # decode ONLY the tokens generated after the prompt
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = seq[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # strip any echoed role/label
    text = re.sub(r'^(?:<\|assistant\|>|assistant|Assistant)\s*:?', '', text).lstrip()
    text = re.sub(r'^(?:Answer)\s*:?', '', text).lstrip()

    return text, len(inputs["input_ids"][0])

# ---- main ----
def main():
    ap = argparse.ArgumentParser(description="Ask the textbook with long-context model")
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--gen_model", type=str, default=GEN_MODEL)
    args = ap.parse_args()

    emb_model = SentenceTransformer(EMB_MODEL)
    index = load_index()
    conn = connect_db()

    q_vec = embed(args.question, emb_model)
    D, I = index.search(q_vec, args.k)
    ids = [int(i) for i in I[0] if i != -1]
    rows = fetch_chunks(conn, ids)

    if not rows:
        print("\nNo retrieved chunks. Re-ingest or check PDF text extraction.\n")
        raise SystemExit(1)

    print("\nTop chunks (pre-pack):\n")
    for i, (_, text, chunk_idx, title, path) in enumerate(rows, 1):
        prev = clean(text).replace("\n", " ")
        print(f"[{i}] {title} · chunk {chunk_idx} · {prev[:160]}{'...' if len(prev)>160 else ''}")

    tokenizer = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.gen_model,
        trust_remote_code=True,
        device_map="auto",
        dtype="auto",
        attn_implementation="eager",
        low_cpu_mem_usage=False,
    )
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
    # sanitize gen config
    gc = model.generation_config
    gc.do_sample = False; gc.temperature = None; gc.top_p = None; gc.top_k = None
    model.generation_config = gc

    context, used_idx = pack_context(rows, tokenizer, args.question)
    messages = build_messages(context, args.question)
    answer, prompt_tokens = generate_answer(model, tokenizer, messages)

    print(f"\n[debug] prompt_tokens={prompt_tokens}  used_chunks={used_idx}\n")
    print("Answer:\n")
    print(answer)
    print("\nSources used:\n")
    for i in used_idx:
        _, _, chunk_idx, title, path = rows[i - 1]
        print(f"[{i}] {title} · chunk {chunk_idx} · {os.path.basename(path)}")

if __name__ == "__main__":
    main()
