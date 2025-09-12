#!/usr/bin/env python3
import os, re, json, logging, sqlite3, math
from functools import lru_cache
from typing import List, Tuple, Dict

import numpy as np
import faiss, torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- config ----
DB_PATH = os.getenv("RAG_DB", "textbook.db")
INDEX_PATH = os.getenv("RAG_INDEX", "indexes/textbook.faiss")
EMB_MODEL = os.getenv("RAG_EMB", "all-MiniLM-L6-v2")
GEN_DEFAULT = os.getenv("RAG_GEN", "Qwen/Qwen2.5-0.5B-Instruct")  # light + stable

CHUNK_CHAR_LIMIT = int(os.getenv("RAG_CHUNK_CHAR_LIMIT", "512"))
MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", "2048"))     # answer budget
REWRITE_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", "512"))
SAFETY_MARGIN = int(os.getenv("RAG_SAFETY_MARGIN", "512"))
FORCE_CPU = os.getenv("RAG_FORCE_CPU", "0") == "1"

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

app = FastAPI(title="Textbook RAG API")

# ---- models ----
class AskRequest(BaseModel):
    question: str
    k: int = 8

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict]
    prompt_tokens: int
    used_chunks: List[int]

# ---- caches/resources ----
@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL)

@lru_cache(maxsize=1)
def get_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    return faiss.read_index(INDEX_PATH)

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@lru_cache(maxsize=2)
def get_generator(model_name: str):
    device_map = "cpu" if FORCE_CPU else "auto"
    dtype = "float32" if device_map == "cpu" else "auto"
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device_map,
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if mdl.generation_config.pad_token_id is None:
        mdl.generation_config.pad_token_id = tok.eos_token_id
    gc = mdl.generation_config
    gc.do_sample = False; gc.temperature = None; gc.top_p = None; gc.top_k = None
    mdl.generation_config = gc
    return tok, mdl

# ---- utils ----
def clean(s: str) -> str:
    s = s.replace("", "- ").replace("\u2022", "- ").replace("\t", " ")
    s = s.encode("utf-8", "ignore").decode("utf-8")
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())

def get_budget(tokenizer) -> int:
    ml = getattr(tokenizer, "model_max_length", 4096) or 4096
    if ml > 100000: ml = 4096
    return max(1024, ml - SAFETY_MARGIN)

def embed(texts, embedder: SentenceTransformer):
    if isinstance(texts, str):
        texts = [texts]
    arr = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return arr

def fetch_chunks(conn, ids: List[int]) -> List[Tuple]:
    if not ids: return []
    q = ",".join(["?"]*len(ids))
    cur = conn.cursor()
    cur.execute(f"""
        SELECT c.id, c.text, c.chunk_index, d.title, d.path
        FROM chunks c JOIN documents d ON c.document_id = d.id
        WHERE c.id IN ({q})
    """, ids)
    m = {row[0]: row for row in cur.fetchall()}
    return [m[i] for i in ids if i in m]


# ---- multi-query rewrite & intent ----
REWRITE_SYS = (
    "You rewrite user questions into textbook-style search queries for Med-Surgical Nursing (15th Edition) "
    "and detect intent.\n"
    "Return ONLY JSON with: expanded (≤4 strings), intent (definition|explain|steps|compare|list|rationale|other), "
    "focus (1–6 key terms).\n"
    "If the user says 'in detail' or 'explain', broaden expansions to include: definition, classification/types, "
    "pathophysiology/mechanism, clinical features, diagnosis, management, complications, prognosis—IF the context has them.\n"
    "If the user mentions mechanism/physiology/how it works, expand with anatomy/physiology keywords.\n"
    "No prose."
)

def small_generate_json(tok, mdl, question:str) -> Dict:
    user = f"Question: {question}\nReturn JSON only."
    msgs = [{"role":"system","content":REWRITE_SYS},{"role":"user","content":user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
        **inputs,
        max_new_tokens=REWRITE_TOKENS,
        do_sample=False,                 # keep deterministic
        repetition_penalty=1.18,         # 1.1–1.3 works well
        no_repeat_ngram_size=6,          # block repeated 6-grams
        renormalize_logits=True,
        use_cache=True,
        eos_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
        )
    text = tok.decode(out.sequences[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return {"expanded": [], "intent": "other", "focus": []}

# ---- retrieval: multi-query + fusion + MMR ----
def mmr_select(q_vec: np.ndarray, c_vecs: np.ndarray, c_ids: List[int], top_k:int, lamb:float=0.65):
    sims = c_vecs @ q_vec  # (n,)
    selected = []
    if len(c_ids) == 0: return []
    candidates = list(range(len(c_ids)))
    while candidates and len(selected) < top_k:
        if not selected:
            i = int(np.argmax(sims[candidates]))
            selected.append(candidates.pop(i))
        else:
            best_i, best_score = None, -1e9
            for ci in candidates:
                diversity = 0.0
                for si in selected:
                    diversity = max(diversity, float(c_vecs[ci] @ c_vecs[si]))
                score = lamb * float(sims[ci]) - (1.0 - lamb) * diversity
                if score > best_score:
                    best_score, best_i = score, ci
            selected.append(best_i); candidates.remove(best_i)
    return [c_ids[i] for i in selected]

def multi_query_search(index, embedder, tok, mdl, question:str, k:int, k_per_q:int=24, cap:int=64):
    info = small_generate_json(tok, mdl, question)
    expanded = [q.strip() for q in (info.get("expanded") or []) if q and isinstance(q, str)]
    queries = [question] + expanded
    seen, mq = set(), []
    for q in queries:
        qn = q.lower().strip()
        if qn and qn not in seen:
            seen.add(qn); mq.append(q)

    id2score: Dict[int, float] = {}
    q_vec_main = embed(question, embedder)[0]
    for q in mq:
        qv = embed(q, embedder)[0:1]
        D, I = index.search(qv, k_per_q)
        for d, i in zip(D[0], I[0]):
            if i == -1: continue
            id2score[int(i)] = max(id2score.get(int(i), -1e9), float(d))

    cand_ids = [cid for cid, _ in sorted(id2score.items(), key=lambda x: x[1], reverse=True)[:cap]]
    return cand_ids, q_vec_main, info.get("intent", "other")

# --- intent helper ---
def infer_intent(question: str) -> str:
    q = (question or "").lower()

    if any(w in q for w in ["principle", "principles", "tenet", "core values"]):
        return "principles"
    if any(w in q for w in ["define", "definition", "what is", "what are", "meaning of"]):
        return "definition"
    if any(w in q for w in ["steps", "how to", "procedure", "protocol", "process", "algorithm"]):
        return "steps"
    if any(w in q for w in ["compare", "vs", "difference", "differences", "contrast"]):
        return "compare"
    if any(w in q for w in ["list", "types", "categories", "kinds", "classification"]):
        return "list"
    if any(w in q for w in ["why", "rationale", "reason", "pathogenesis", "etiology"]):
        return "rationale"

    # Physiology / mechanism anywhere (any organ/system)
    if any(w in q for w in [
        "mechanism", "how it works", "how does it work", "physiology",
        "function of", "functional anatomy", "conduction", "cycle", "pathway"
    ]):
        return "physiology"

    # Strong “explain fully” cues → explain
    if any(w in q for w in ["explain", "in detail", "detailed", "overview", "notes", "teach me"]):
        return "explain"

    return "explain"

# ---- prompt building (study-optimized v3 + principles + conclusion) ----
def build_messages(context: str, question: str, intent: str | None = None):
    intent = (intent or infer_intent(question)).lower()

    style_hint = {
        "principles": "Return bullets with EXACT principle name + ≤18-word summary on the same line.",
        "definition": "Begin with a crisp 1–2 sentence definition, then short bullets.",
        "steps": "Use a numbered procedure with brief rationale; include safety notes.",
        "compare": "Prefer a compact Markdown table; otherwise parallel bullets.",
        "list": "Return a clean bullet list with one-line explanations.",
        "rationale": "State the 'why' behind each point in one short clause.",
        "explain": "Teach step-by-step with subheadings and examples from context.",
        "physiology": (
            "Explain mechanism in ordered stages:\n"
            "1) Anatomy/structure\n"
            "2) Inputs\n"
            "3) Process/pathway (signals/flows)\n"
            "4) Output/effect\n"
            "5) Regulation (neural/hormonal/feedback)\n"
            "Keep bullets ≤20 words; avoid pathology/management unless asked."
        ),
        "other": "Be concise but thorough.",
    }.get(intent, "Be concise but thorough.")

    base_rules = (
        "You are a Med-Surgical Nursing (15th Edition) RAG tutor.\n"
        "RULES:\n"
        "1) Use ONLY facts inside <context>. If insufficient, reply exactly: I don't know from the textbook.\n"
        "2) Output MUST be valid Markdown. No role labels, no prefaces, no epilogues.\n"
        "3) Do NOT invent numbers, examples, mnemonics, or references outside <context>.\n"
        "4) Prefer textbook wording and terminology. Keep concise, concrete.\n"
        "5) For explain/physiology, ALWAYS include: Summary, Definition, Key Points, Detailed Explanation, Conclusion. "
        "If support is thin, write 1–2 cautious lines drawn from <context> rather than skipping.\n"
        "6) Lists ≤12 items; bullets ≤20 words; no duplication.\n"
        f"Style hint: {style_hint}\n\n"
    )

    layout_default = (
        "FORMAT — use these exact headings:\n"
        "## Summary\n"
        "- 2–4 sentences directly answering the question.\n\n"
        "## Definition\n"
        "- One-sentence definition (≤30 words). Optionally up to 3 bullets for hallmarks/subtypes.\n\n"
        "## Key Points\n"
        "- 3–8 concise bullets (exam-relevant).\n\n"
        "## Detailed Explanation\n"
        "### Core Concepts\n"
        "- Short, concrete paragraphs strictly from <context>.\n"
        "### If Mechanism/Physiology Is Relevant\n"
        "- Structure → flow/pathway → conduction → cycle/output (bullets).\n"
        "### If Types/Stages/Classes\n"
        "- 1–2 lines per type/stage/class.\n"
        "### If Assessment/Diagnosis\n"
        "- Signs, symptoms, tests (bullets).\n"
        "### If Management/Treatment\n"
        "- First-line, adjuncts, monitoring (bullets).\n"
        "### If Prognosis/Complications\n"
        "- 1–3 concise lines.\n\n"
        "## Key Terms\n"
        "- Up to 8 terms, alphabetized, one-line each from <context> only.\n\n"
        "## Self-Quiz\n"
        "- 3–5 short, high-yield questions (no answers).\n\n"
        "## Conclusion\n"
        "- 1–2 sentence takeaway tying back to the question.\n\n"
        "STYLE:\n"
        "- No emojis. No fluff. No giant paragraphs.\n"
        "- Bullets ≤20 words. No repetition. Exam-style phrasing.\n"
    )

    layout_principles = (
        "FORMAT — headings EXACTLY as written:\n"
        "## Summary\n"
        "- One short paragraph stating you will list the principles succinctly.\n\n"
        "## Principles\n"
        "- **Exact Principle Name**: ≤18-word summary using textbook wording.\n\n"
        "## Conclusion\n"
        "- One sentence synthesizing the principles’ aim.\n"
    )

    system = base_rules + (layout_principles if intent == "principles" else layout_default)

    user = (
        f"<context>\n{context}\n</context>\n\n"
        f"Question: {question}\n"
        f"Intent: {intent}\n\n"
        "Return ALL mandatory sections (Summary, Definition, Key Points, Detailed Explanation, Conclusion). "
        "Optional subsections may be omitted only if no support exists in <context>."
    )

    return [{"role":"system","content":system},{"role":"user","content":user}]

# ---- Markdown post-processor to fix hierarchy/spacing ----
SECTION_ALIASES = {
    r'^\s*(?:#+\s*)?summary\s*:?\s*$':                  "## Summary",
    r'^\s*(?:#+\s*)?definition\s*:?\s*$':               "## Definition",
    r'^\s*(?:#+\s*)?key\s*points?\s*:?\s*$':            "## Key Points",
    r'^\s*(?:#+\s*)?detailed( explanation)?\s*:?\s*$':  "## Detailed Explanation",
    r'^\s*(?:#+\s*)?principles(.*)\s*$':                "## Principles",
    r'^\s*(?:#+\s*)?conclusion(s)?\s*:?\s*$':           "## Conclusion",
}

SUBHEAD_ALIASES = {
    r'^\s*(?:#+\s*)?pathophysiology\s*:?\s*$':              "### Pathophysiology",
    r'^\s*(?:#+\s*)?mechanism\s*:?\s*$':                    "### Mechanism",
    r'^\s*(?:#+\s*)?(types?|classification)\s*:?\s*$':      "### Types / Subtypes",
    r'^\s*(?:#+\s*)?(clinical\s*features?|signs\s*&\s*symptoms)\s*:?\s*$': "### Clinical Features",
    r'^\s*(?:#+\s*)?risk\s*factors?\s*:?\s*$':              "### Risk Factors",
    r'^\s*(?:#+\s*)?(assessment|diagnosis|diagnostic\s*tests?)\s*:?\s*$':  "### Diagnosis",
    r'^\s*(?:#+\s*)?(management|treatment|therapy)\s*:?\s*$':              "### Management / Treatment",
    r'^\s*(?:#+\s*)?(prognosis|complications?)\s*:?\s*$':                  "### Prognosis / Complications",
}

BULLET_CHARS = r"[\u2022•●‣∙·–—-]"

def polish_markdown(md: str) -> str:
    if not md: 
        return md

    lines = md.strip().splitlines()
    out = []
    for ln in lines:
        s = ln.rstrip()

        # Normalize bullets to '- '
        if re.match(rf'^{BULLET_CHARS}\s*', s):
            s = re.sub(rf'^{BULLET_CHARS}\s*', '- ', s)

        # H2 aliases
        matched = False
        for pat, repl in SECTION_ALIASES.items():
            if re.match(pat, s, flags=re.I):
                s = repl; matched = True; break

        # H3 aliases (only if not converted to H2)
        if not matched:
            for pat, repl in SUBHEAD_ALIASES.items():
                if re.match(pat, s, flags=re.I):
                    s = repl; break

        out.append(s)

    text = "\n".join(out)

    # Insert '## Detailed Explanation' if any H3 exists but H2 is missing
    if "## Detailed Explanation" not in text and re.search(r'(?m)^### ', text):
        parts = text.splitlines()
        for i, line in enumerate(parts):
            if line.startswith("### "):
                parts.insert(i, "")
                parts.insert(i, "## Detailed Explanation")
                break
        text = "\n".join(parts)

    # Spacing and tidy-up
    text = re.sub(r'(?m)(^## .+?$)', r'\n\1\n', text)
    text = re.sub(r'(?m)(^### .+?$)', r'\n\1\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    text = re.sub(r'(?m)(^[^\n-#].+)\n(- )', r'\1\n\n- ', text)
    return text


SECTION_CAPS = {
    "## Definition": 3,        # up to 3 bullets under the single-sentence definition
    "## Key Points": 8,
    "## Key Terms": 8,
    "## Self-Quiz": 5,
    "## Principles": 12,
}

def _dedupe_and_cap_list(lines, cap):
    seen = set()
    out = []
    for ln in lines:
        key = ln.strip()
        if key in seen: 
            continue
        seen.add(key)
        out.append(ln)
        if cap and len(out) >= cap:
            break
    return out

def limit_and_dedupe(md: str) -> str:
    if not md: return md
    lines = md.splitlines()
    result, buffer, current_cap = [], [], None

    def flush():
        nonlocal result, buffer, current_cap
        if buffer:
            result.extend(_dedupe_and_cap_list(buffer, current_cap))
            buffer = []

    # determine caps by heading
    current_h = None
    for ln in lines:
        if ln.startswith("## "):
            flush()
            current_h = ln.strip()
            current_cap = SECTION_CAPS.get(current_h, 12)  # default cap for any list in this section
            result.append(ln); continue
        if ln.startswith("### "):
            flush()
            # subheadings inherit current section cap
            result.append(ln); continue

        if ln.lstrip().startswith(("- ", "* ", "+ ", "1.", "2.", "3.")):
            buffer.append(ln)
        else:
            flush()
            result.append(ln)

    flush()
    return "\n".join(result)

# ---- packing ----
def pack_context(rows, tokenizer, question:str):
    budget = get_budget(tokenizer)
    packed, used = [], []
    for i, (_, text, chunk_idx, title, path) in enumerate(rows, 1):
        t = clean(text)[:CHUNK_CHAR_LIMIT]
        header = f"{title} · chunk {chunk_idx}"
        block = f"[{i}] ({header})\n{t}"
        trial = "\n\n---\n\n".join(packed + [block])
        msgs = build_messages(trial, question, intent="other")
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if len(tokenizer(prompt).input_ids) <= budget:
            packed.append(block); used.append(i)
        else:
            break
    return "\n\n---\n\n".join(packed), used

def generate_answer(tok, mdl, messages, max_new:int) -> Tuple[str, int]:
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,                 # keep deterministic
            repetition_penalty=1.18,         # 1.1–1.3 works well
            no_repeat_ngram_size=6,          # block repeated 6-grams
            eos_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
        )
    seq = out.sequences
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = seq[0, prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    text = re.sub(r'^(?:<\|assistant\|>|assistant|Assistant)\s*:?', '', text).lstrip()
    text = re.sub(r'^(?:Answer)\s*:?', '', text).lstrip()
    return text, int(inputs["input_ids"].shape[1])

def strip_context_sections(md: str) -> str:
    # Remove any "### Context-<n> ... (until next H2 or end)"
    return re.sub(r'(?s)\n###\s*Context-\d+.*?(?=\n##\s|\Z)', '', md)

# ---- API ----
@app.get("/api/health")
def health():
    ok = {"db": os.path.exists(DB_PATH), "index": os.path.exists(INDEX_PATH)}
    return {"ok": all(ok.values()), **ok}

@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        embedder = get_embedder()
        index = get_index()
        tok, mdl = get_generator(GEN_DEFAULT)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    conn = get_conn()

    # multi-query search + fusion
    cand_ids, q_vec_main, intent_mq = multi_query_search(index, embedder, tok, mdl, q, req.k)
    intent = intent_mq if intent_mq and intent_mq != "other" else infer_intent(q)

    if not cand_ids:
        raise HTTPException(status_code=404, detail="No relevant chunks. Re-ingest or check PDF extraction.")

    # fetch & embed candidates
    cand_rows = fetch_chunks(conn, cand_ids)
    cand_texts = [clean(r[1])[:CHUNK_CHAR_LIMIT] for r in cand_rows]
    c_vecs = embed(cand_texts, embedder)

    # MMR rerank
    sel_ids = mmr_select(q_vec_main, c_vecs, cand_ids, top_k=req.k, lamb=0.65) or cand_ids[:req.k]
    id2row = {row[0]: row for row in cand_rows}
    rows = [id2row[i] for i in sel_ids if i in id2row]

    # final packing with budget
    # inside ask(), after building `rows` from fetch_chunks:
    # rows now are tuples: (id, text, chunk_index, title, path)

    # final packing
    context_blocks, preview_rows, budget = [], [], get_budget(tok)
    for i, (_, text, chunk_idx, title, path) in enumerate(rows, 1):
        header = f"{title} · chunk {chunk_idx}"
        block = f"[{i}] ({header})\n{clean(text)[:CHUNK_CHAR_LIMIT]}"
        trial = "\n\n---\n\n".join(context_blocks + [block])
        msgs = build_messages(trial, q, intent=intent)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if len(tok(prompt).input_ids) <= budget:
            context_blocks.append(block); preview_rows.append(i)
        else:
            break

    if not context_blocks:
        _, text, chunk_idx, title, path = rows[0]
        header = f"{title} · chunk {chunk_idx}"
        context_blocks.append(f"[1] ({header})\n{clean(text)[:CHUNK_CHAR_LIMIT]}")
        preview_rows = [1]

    context = "\n\n---\n\n".join(context_blocks)
    messages = build_messages(context, q, intent=intent)

    answer, prompt_tokens = generate_answer(tok, mdl, messages, MAX_NEW_TOKENS)
    answer = polish_markdown(answer)
    answer = strip_context_sections(answer)
    answer = limit_and_dedupe(answer)

    # sources (no page)
    sources = []
    for i in preview_rows:
        _, _, chunk_idx, title, path = rows[i-1]
        sources.append({"n": i, "title": title, "chunk": int(chunk_idx), "file": os.path.basename(path)})

    return AskResponse(answer=answer, sources=sources, prompt_tokens=prompt_tokens, used_chunks=preview_rows)


# ---- static HTML ----
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
def index():
    return FileResponse("static/index.html")
