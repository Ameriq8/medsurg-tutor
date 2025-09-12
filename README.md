# Textbook RAG Chatbot (API + Chat UI)

Local Retrieval‑Augmented Generation (RAG) over your textbooks/notes. Ingest PDFs/TXT/MD, store text in SQLite, vectors in FAISS, and answer questions using a small open LLM. Includes a FastAPI backend and a ChatGPT‑style Tailwind UI.

---

## Features

* Ingest PDFs/TXT/MD with configurable chunk size and overlap.
* Text stored in **SQLite**; vector search with **FAISS** (cosine via normalized inner product).
* Generator model locked to server **GEN\_DEFAULT** for consistent behavior.
* ChatGPT‑like web UI (Tailwind) with Enter to send, Shift+Enter newline.
* Clean answers only (no token meta or sources shown in UI).

---

## Architecture

```
PDF/TXT/MD → ingest_textbook.py → SQLite (text) + FAISS (vectors)
                                     ↓
                                FastAPI (/api/ask)
                                     ↓
                             Generator (GEN_DEFAULT)
                                     ↓
                               Chat UI (Tailwind)
```

---

## Requirements

* Python ≥ 3.10 (tested on 3.12)
* OS: Linux/macOS/WSL (CPU‑only works; GPU if available)

### Python deps

```
fastapi
uvicorn[standard]
sentence-transformers
faiss-cpu
pypdf
transformers>=4.46
accelerate
torch
einops
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U fastapi "uvicorn[standard]" sentence-transformers faiss-cpu pypdf \
  "transformers>=4.46" accelerate torch einops
```

---

## Ingest textbooks

```bash
# example
python ingest_textbook.py --path ./Crisis.pdf --max_words 120 --overlap 30 --clear
```

**Flags**

* `--path`: files or folders (globs ok). Accepts `.pdf .txt .md`.
* `--max_words` (default 120)
* `--overlap` (default 30)
* `--clear`: delete existing chunks for the same docs before insert.
* `--embed_model` (default `all-MiniLM-L6-v2`).

The FAISS index is rebuilt from the DB on each ingest run (no drift).

---

## Run the API + Chat UI

```bash
# project tree should include:
# app_api.py, ingest_textbook.py, static/index.html, indexes/, textbook.db

# start the server
uvicorn app_api:app --host 0.0.0.0 --port 8000 --reload

# open the UI
xdg-open http://localhost:8000/  # or open in your browser
```

---

## Configuration (env vars)

* `RAG_DB` (default `textbook.db`)
* `RAG_INDEX` (default `indexes/textbook.faiss`)
* `RAG_EMB` (default `all-MiniLM-L6-v2`)
* `RAG_GEN` (**GEN\_DEFAULT**, default `Qwen/Qwen2.5-0.5B-Instruct`)
* `RAG_CHUNK_CHAR_LIMIT` (default `700`)
* `RAG_MAX_NEW_TOKENS` (default `192`)
* `RAG_SAFETY_MARGIN` (default `512`)
* `RAG_FORCE_CPU` (`1` to force CPU; default `0`)

Example:

```bash
export RAG_GEN=Qwen/Qwen2.5-0.5B-Instruct
export RAG_FORCE_CPU=1
uvicorn app_api:app --host 0.0.0.0 --port 8000
```

---

## Project structure

```
project/
├─ ingest_textbook.py          # ingest PDFs/TXT/MD → SQLite + FAISS
├─ query_bot.py                # CLI query tool (dev/debug)
├─ app_api.py                  # FastAPI backend (locks to GEN_DEFAULT)
├─ static/
│  └─ index.html               # Tailwind chat UI (ChatGPT‑like)
├─ indexes/
│  └─ textbook.faiss           # FAISS index (generated)
└─ textbook.db                 # SQLite DB (generated)
```

---

## API

### POST `/api/ask`

**Request**

```json
{ "question": "...", "k": 8 }
```

**Response**

```json
{ "answer": "...markdown..." }
```

Errors: `400` empty question, `404` no relevant chunks, `500` missing index/DB.

Health: `GET /api/health` → `{ ok: true/false, db: bool, index: bool }`.

---

## Troubleshooting

* **No answers / 404**: Re‑ingest with smaller chunks: `--max_words 100 --overlap 25 --clear`.
* **Slow on CPU**: set `RAG_FORCE_CPU=1` and keep `Qwen 0.5B`; reduce `RAG_MAX_NEW_TOKENS`.
* **PDF text looks garbled**: convert to text first or try a different extractor; ingestion cleans basic bullets.
* **OOM or stalls**: use the default 0.5B model; ensure swap is available; reduce `CHUNK_CHAR_LIMIT` and `k`.

---

## Maintenance

Free disk space from caches:

```bash
rm -rf ~/.cache/huggingface/{hub,transformers,datasets,accelerate}
rm -rf ~/.cache/torch
pip cache purge
```

Reset project artifacts:

```bash
rm -rf indexes textbook.db
```

---

## Security & privacy

* Runs locally. No external calls. Your PDFs never leave the machine unless you change the code.

---

## License

Choose one (MIT/Apache‑2.0/BSD‑3‑Clause) and add a LICENSE file.
