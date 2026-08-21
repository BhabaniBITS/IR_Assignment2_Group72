"""
db.py
Storage layer for the IR system.

Design choice (requirement B):
    "Store extracted metadata separately from document contents."
We therefore use TWO tables:
    documents  -> doc_id, url, title, content, category, source, crawl_depth, crawled_at
    doc_meta   -> doc_id, char_count, word_count, content_hash, fetch_status, language

A third table `links` stores the hyperlink graph discovered while crawling
(used later for PageRank / HITS), and `interactions` stores a synthetic
user-item click/rating log used by the collaborative recommender.
"""

import sqlite3
import hashlib
import os
import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ir_system.db")
DB_PATH = os.path.abspath(DB_PATH)


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(reset=False):
    conn = get_conn()
    cur = conn.cursor()
    if reset:
        for t in ["documents", "doc_meta", "links", "interactions"]:
            cur.execute(f"DROP TABLE IF EXISTS {t}")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            content TEXT,
            category TEXT,
            source TEXT,
            crawl_depth INTEGER,
            crawled_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_meta (
            doc_id INTEGER PRIMARY KEY,
            char_count INTEGER,
            word_count INTEGER,
            content_hash TEXT,
            fetch_status TEXT,
            language TEXT,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS links (
            src_doc INTEGER,
            dst_doc INTEGER,
            PRIMARY KEY (src_doc, dst_doc)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            user_id INTEGER,
            doc_id INTEGER,
            rating REAL,
            PRIMARY KEY (user_id, doc_id)
        )
    """)
    conn.commit()
    conn.close()


def content_hash(text: str) -> str:
    return hashlib.md5((text or "").strip().lower().encode("utf-8", "ignore")).hexdigest()


def url_exists(url: str) -> bool:
    conn = get_conn()
    r = conn.execute("SELECT 1 FROM documents WHERE url = ?", (url,)).fetchone()
    conn.close()
    return r is not None


def content_duplicate(text: str):
    """Return doc_id of an existing document with identical content hash, else None."""
    h = content_hash(text)
    conn = get_conn()
    r = conn.execute("SELECT doc_id FROM doc_meta WHERE content_hash = ?", (h,)).fetchone()
    conn.close()
    return r[0] if r else None


def insert_document(url, title, content, category, source, depth, fetch_status="ok", language="en"):
    """Insert a document + its metadata as two separate writes. Returns doc_id or None if duplicate."""
    if url_exists(url):
        return None
    dup = content_duplicate(content)
    if dup is not None:
        return None

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (url, title, content, category, source, crawl_depth, crawled_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (url, title, content, category, source, depth, datetime.datetime.now().isoformat())
    )
    doc_id = cur.lastrowid
    words = (content or "").split()
    cur.execute(
        "INSERT INTO doc_meta (doc_id, char_count, word_count, content_hash, fetch_status, language) "
        "VALUES (?,?,?,?,?,?)",
        (doc_id, len(content or ""), len(words), content_hash(content), fetch_status, language)
    )
    conn.commit()
    conn.close()
    return doc_id


def add_link(src_doc, dst_doc):
    if src_doc == dst_doc:
        return
    conn = get_conn()
    try:
        conn.execute("INSERT OR IGNORE INTO links (src_doc, dst_doc) VALUES (?,?)", (src_doc, dst_doc))
        conn.commit()
    finally:
        conn.close()


def get_all_documents(as_dataframe=True):
    import pandas as pd
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT d.doc_id, d.url, d.title, d.content, d.category, d.source, d.crawl_depth, d.crawled_at,
               m.char_count, m.word_count, m.content_hash, m.fetch_status, m.language
        FROM documents d LEFT JOIN doc_meta m ON d.doc_id = m.doc_id
        ORDER BY d.doc_id
    """, conn)
    conn.close()
    return df if as_dataframe else df.to_dict("records")


def get_links():
    import pandas as pd
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM links", conn)
    conn.close()
    return df


def doc_count():
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    return n


def clear_all():
    init_db(reset=True)


def save_interactions(df):
    conn = get_conn()
    df.to_sql("interactions", conn, if_exists="replace", index=False)
    conn.close()


def get_interactions():
    import pandas as pd
    conn = get_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM interactions", conn)
    except Exception:
        df = pd.DataFrame(columns=["user_id", "doc_id", "rating"])
    conn.close()
    return df
