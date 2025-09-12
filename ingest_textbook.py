#!/usr/bin/env python3
# ingest_textbook.py — Ingest PDFs/TXT/MD → SQLite + FAISS (cosine via L2-normalized MiniLM)
import argparse, os, sqlite3, time, re, glob
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss

# Prefer PyMuPDF if available (better extraction), else fall back to pypdf
_HAS_PYMUPDF = False
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    from pypdf import PdfReader  # fallback

from sentence_transformers import SentenceTransformer

DB_PATH = "textbook.db"
INDEX_PATH = "indexes/textbook.faiss"

# ------------------------ DB ------------------------
def ensure_dirs():
    Path("indexes").mkdir(parents=True, exist_ok=True)

def _ensure_page_column(conn: sqlite3.Connection) -> None:
    # fetch columns from the *cursor*, then build a set of names
    rows = conn.execute("PRAGMA table_info(chunks)").fetchall()
    cols = {r[1] for r in rows}  # r[1] = column name
    if "page_start" not in cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN page_start INTEGER")
        conn.commit()

def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY,
        path TEXT UNIQUE,
        title TEXT,
        created_at REAL
    );""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY,
        document_id INTEGER,
        chunk_index INTEGER,
        page_start INTEGER,            -- <— include when creating new DBs
        text TEXT,
        n_words INTEGER,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    );""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id, chunk_index);")
    conn.commit()
    _ensure_page_column(conn)         # keeps old DBs compatible
    return conn

def upsert_document(conn: sqlite3.Connection, path: str, title: str) -> int:
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO documents(path, title, created_at) VALUES(?,?,?)",
                (path, title, time.time()))
    conn.commit()
    cur.execute("SELECT id FROM documents WHERE path=?", (path,))
    return cur.fetchone()[0]

def delete_doc_chunks(conn: sqlite3.Connection, doc_id: int):
    conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
    conn.commit()

def insert_chunks(conn: sqlite3.Connection, doc_id: int, chunks_with_page: List[Tuple[int, str]]) -> List[int]:
    """
    chunks_with_page: list of (page_number, text); page_number is 1-based.
    """
    cur = conn.cursor()
    rows = [(doc_id, idx, page, txt, len(txt.split()))
            for idx, (page, txt) in enumerate(chunks_with_page)]
    cur.executemany(
        "INSERT INTO chunks(document_id, chunk_index, page_start, text, n_words) VALUES(?,?,?,?,?)",
        rows
    )
    conn.commit()
    cur.execute("SELECT id FROM chunks WHERE document_id=? ORDER BY chunk_index ASC", (doc_id,))
    return [r[0] for r in cur.fetchall()]

# --------------------- Extraction --------------------
BULLETS = ["•", "", "◦", "●", "–", "—", "‣", "∙", "·", "", "\u2022"]

def _clean_text(s: str) -> str:
    # fix hyphenated line-breaks like "ther-\nmal"
    s = re.sub(r"-\n", "", s)
    # normalize bullets to "- "
    for b in BULLETS:
        s = s.replace(b, "- ")
    # collapse weird spaces and extra newlines
    s = s.replace("\t", " ")
    s = re.sub(r"[ \u00A0]+", " ", s)
    s = re.sub(r"\r\n|\r", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def extract_pages_from_pdf(pdf_path: str, extractor: str = "auto") -> List[str]:
    """Return a list of page texts (1-based order)."""
    extractor = extractor.lower()
    if extractor not in {"auto", "pymupdf", "pypdf"}:
        extractor = "auto"

    use_pymupdf = (_HAS_PYMUPDF and extractor in {"auto", "pymupdf"}) or (extractor == "pymupdf")

    pages: List[str] = []
    if use_pymupdf:
        doc = fitz.open(pdf_path)
        for p in doc:
            pages.append(_clean_text(p.get_text("text") or ""))
        doc.close()
    else:
        reader = PdfReader(pdf_path)
        for p in reader.pages:
            pages.append(_clean_text(p.extract_text() or ""))

    return pages

# ---------------------- Chunking ---------------------
def chunk_text(text: str, max_words: int, overlap: int, min_words: int = 10) -> List[str]:
    words = text.split()
    out, i = [], 0
    while i < len(words):
        j = min(i + max_words, len(words))
        chunk = " ".join(words[i:j]).strip()
        if chunk and len(chunk.split()) >= min_words:
            out.append(chunk)
        if j >= len(words): break
        i = max(0, j - overlap)
    return out

def chunk_pages(pages: List[str], max_words: int, overlap: int, min_words: int = 10) -> List[Tuple[int, str]]:
    """
    Chunk each page independently so we can keep the page number.
    Returns list of (page_number, chunk_text).
    """
    out: List[Tuple[int, str]] = []
    for page_idx, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        for ch in chunk_text(page_text, max_words=max_words, overlap=overlap, min_words=min_words):
            out.append((page_idx, ch))
    return out

# ---------------------- FAISS ------------------------
def _encode_batches(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        enc = model.encode(texts[i:i+batch_size], convert_to_numpy=True, normalize_embeddings=True)
        vecs.append(enc.astype("float32"))
    if not vecs:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype="float32")
    return np.vstack(vecs)

def rebuild_index_from_db(conn: sqlite3.Connection, model: SentenceTransformer) -> int:
    ensure_dirs()
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))  # cosine with normalized vectors
    cur = conn.cursor()
    cur.execute("SELECT id, text FROM chunks ORDER BY id ASC")
    rows: List[Tuple[int, str]] = cur.fetchall()
    if not rows:
        faiss.write_index(index, INDEX_PATH)
        return 0
    ids  = np.array([r[0] for r in rows], dtype=np.int64)
    texts = [r[1] for r in rows]
    vecs = _encode_batches(model, texts)
    index.add_with_ids(vecs, ids)
    faiss.write_index(index, INDEX_PATH)
    return index.ntotal

# ----------------------- I/O -------------------------
def expand_inputs(items: List[str]) -> List[str]:
    files = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            files += glob.glob(str(p / "**/*"), recursive=True)
        else:
            files += glob.glob(item)
    return [f for f in files if os.path.isfile(f) and f.lower().endswith((".pdf", ".txt", ".md"))]

# ---------------------- Ingest -----------------------
def ingest_paths(paths: List[str], max_words: int, overlap: int, clear_existing: bool,
                 embed_model: str, extractor: str):
    ensure_dirs()
    conn = connect_db()
    model = SentenceTransformer(embed_model)

    total_docs, added_chunks = 0, 0
    for p in paths:
        p_abs = str(Path(p).resolve())
        title = Path(p_abs).stem
        doc_id = upsert_document(conn, p_abs, title)
        if clear_existing:
            delete_doc_chunks(conn, doc_id)

        chunks_with_page: List[Tuple[int, str]] = []

        if p_abs.lower().endswith(".pdf"):
            pages = extract_pages_from_pdf(p_abs, extractor=extractor)
            chunks_with_page = chunk_pages(pages, max_words=max_words, overlap=overlap, min_words=10)
        else:
            # TXT/MD: treat whole file as "page 1"
            with open(p_abs, "r", encoding="utf-8", errors="ignore") as f:
                text = _clean_text(f.read())
            chunks = chunk_text(text, max_words=max_words, overlap=overlap, min_words=10)
            chunks_with_page = [(1, ch) for ch in chunks]

        if not chunks_with_page:
            print(f"[skip] Empty after extraction: {p_abs}")
            continue

        insert_chunks(conn, doc_id, chunks_with_page)
        total_docs += 1
        added_chunks += len(chunks_with_page)
        print(f"[ok] {title}: {len(chunks_with_page)} chunks")

    total_index = rebuild_index_from_db(conn, model)
    conn.close()
    print(f"\nDone. Docs ingested: {total_docs} | New chunks: {added_chunks} | Index size: {total_index}")

# ----------------------- CLI -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest textbooks into SQLite + FAISS (cosine)")
    ap.add_argument("--path", nargs="+", required=True, help="Files or folders (glob OK)")
    ap.add_argument("--max_words", type=int, default=100, help="Words per chunk (default 100)")
    ap.add_argument("--overlap", type=int, default=30, help="Overlap words (default 30)")
    ap.add_argument("--clear", action="store_true", help="Delete existing chunks for same docs before ingest")
    ap.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    ap.add_argument("--extractor", choices=["auto", "pymupdf", "pypdf"], default="auto",
                    help="PDF text extractor (default auto)")
    args = ap.parse_args()

    files = expand_inputs(args.path)
    if not files:
        raise SystemExit("No PDF/TXT/MD files found.")

    ingest_paths(files, args.max_words, args.overlap, args.clear, args.embed_model, args.extractor)
