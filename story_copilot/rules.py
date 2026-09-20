import re
import json
import sqlite3
import subprocess
from pathlib import Path

from .store import digest, now, packed


EXTRACTOR_REVISION = "pdftotext-reading-order-v2"


def import_rules(store, path, title, edition="custom"):
    """Append a reading-order extraction; preserve every previous chunk and ID."""
    path = Path(path).expanduser().resolve(strict=True)
    source_hash = digest(path.read_bytes())
    version_result = subprocess.run(
        ["pdftotext", "-v"], check=True, capture_output=True, text=True
    )
    version = (version_result.stderr or version_result.stdout).splitlines()[0]
    # Default Poppler reading order handles columns and end-of-line hyphenation.
    # The older -layout text stays available under its original citation IDs.
    text = subprocess.run(
        ["pdftotext", str(path), "-"], check=True, capture_output=True, text=True
    ).stdout
    if not text.strip():
        raise ValueError(
            "The PDF has no extractable text; the previous index remains active."
        )
    extraction_id = digest(
        packed([source_hash, edition, EXTRACTOR_REVISION, version, digest(text)])
    )[:24]
    metadata = {
        "source_path": str(path),
        "source_sha256": source_hash,
        "text_sha256": digest(text),
        "extractor_revision": EXTRACTOR_REVISION,
        "extractor_version": version,
        "flags": [],
    }
    count = 0
    with store.db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS rule_extractions (
            id TEXT PRIMARY KEY,source_hash TEXT NOT NULL,edition TEXT NOT NULL,
            extractor_revision TEXT NOT NULL,extractor_version TEXT NOT NULL,
            metadata TEXT NOT NULL,created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rule_chunk_extractions (
            rule_id TEXT PRIMARY KEY REFERENCES rules(id),
            extraction_id TEXT NOT NULL REFERENCES rule_extractions(id));
        CREATE TABLE IF NOT EXISTS active_rule_extractions (
            source_hash TEXT NOT NULL,edition TEXT NOT NULL,
            extraction_id TEXT NOT NULL REFERENCES rule_extractions(id),
            PRIMARY KEY(source_hash,edition));
        """)
        db.execute(
            "INSERT OR IGNORE INTO rule_extractions VALUES(?,?,?,?,?,?,?)",
            (
                extraction_id,
                source_hash,
                edition,
                EXTRACTOR_REVISION,
                version,
                packed(metadata),
                now(),
            ),
        )
        for page, content in enumerate(text.split("\f"), 1):
            for start in range(0, len(content), 1600):
                chunk = content[start : start + 1800].strip()
                if not chunk:
                    continue
                rid = digest(packed([extraction_id, page, start, digest(chunk)]))[:24]
                if db.execute("SELECT 1 FROM rules WHERE id=?", (rid,)).fetchone():
                    continue
                db.execute(
                    "INSERT INTO rules VALUES(?,?,?,?,?,?)",
                    (rid, title, page, edition, chunk, source_hash),
                )
                db.execute("INSERT INTO rule_search VALUES(?,?)", (rid, chunk))
                db.execute(
                    "INSERT INTO rule_chunk_extractions VALUES(?,?)",
                    (rid, extraction_id),
                )
                count += 1
        db.execute(
            "INSERT INTO active_rule_extractions VALUES(?,?,?) ON CONFLICT(source_hash,edition) DO UPDATE SET extraction_id=excluded.extraction_id",
            (source_hash, edition, extraction_id),
        )
    return count


def normalize_rule_query(query, edition="custom"):
    return " ".join(query.split())


def rule_query_terms(query, edition="custom"):
    stop = {
        "the",
        "a",
        "an",
        "to",
        "is",
        "i",
        "you",
        "of",
        "and",
        "in",
        "it",
        "for",
        "that",
        "what",
        "how",
        "with",
        "my",
        "can",
        "could",
        "would",
        "should",
        "may",
        "does",
        "have",
        "has",
        "after",
        "before",
        "please",
        "from",
        "when",
        "with",
        "your",
        "than",
        "that",
        "this",
        "rule",
        "rules",
    }
    aliases = {}
    query = normalize_rule_query(query, edition)
    query = " ".join(aliases.get(w, w) for w in re.findall(r"\w+", query.lower()))
    words = [
        w for w in re.findall(r"\w+", query.lower()) if w not in stop and len(w) > 2
    ][:16]
    return list(dict.fromkeys(words))


def campaign_rule_chunks(snapshot):
    """Chunks are bound to documents in this exact frozen campaign snapshot."""
    result = []
    for document in snapshot.get("documents", []):
        metadata = document.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if metadata.get("kind") != "rules":
            continue
        for page, text in enumerate(document["text"].split("\f"), 1):
            for start in range(0, len(text), 1500):
                chunk = text[start : start + 1800].strip()
                if chunk:
                    result.append(
                        {
                            "id": digest(
                                packed(
                                    [document["id"], document["sha256"], page, start]
                                )
                            )[:24],
                            "title": document["title"],
                            "page": page,
                            "text": chunk,
                            "document_id": document["id"],
                            "source_hash": document["sha256"],
                            "visibility": document["visibility"],
                        }
                    )
    return result


def search_rules(store, query, limit=4, edition="custom", *, snapshot=None):
    if not 1 <= limit <= 12:
        raise ValueError("Retrieve one to twelve rule excerpts.")
    if snapshot is not None:
        aliases = snapshot.get("rule_profile", {}).get("query_aliases", {})
        for term, expanded in aliases.items():
            query = re.sub(
                r"(?<!\w)" + re.escape(term) + r"(?!\w)",
                lambda _: expanded,
                query,
                flags=re.I,
            )
        chunks = campaign_rule_chunks(snapshot)
        words = rule_query_terms(query)
        if not words or not chunks:
            return []
        lookup = {chunk["id"]: chunk for chunk in chunks}
        with sqlite3.connect(":memory:") as db:
            db.execute(
                "CREATE VIRTUAL TABLE source_search USING fts5(id UNINDEXED,text,tokenize='porter unicode61')"
            )
            db.executemany(
                "INSERT INTO source_search VALUES(?,?)",
                [(chunk["id"], chunk["text"]) for chunk in chunks],
            )
            match = " OR ".join('"' + word + '"' for word in words)
            ids = [
                row[0]
                for row in db.execute(
                    "SELECT id FROM source_search WHERE source_search MATCH ? ORDER BY bm25(source_search),id LIMIT ?",
                    (match, limit),
                )
            ]
        return [lookup[key] for key in ids]

    words = rule_query_terms(query, edition)
    if not words:
        return []
    match = " OR ".join('"' + w + '"' for w in words)
    with store.db() as db:
        versioned = (
            db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='active_rule_extractions'"
            ).fetchone()
            is not None
        )
        if versioned:
            query_sql = """SELECT r.*,p.extraction_id,e.extractor_revision,e.extractor_version
                FROM rule_search_stemmed f JOIN rules r ON r.id=f.id
                LEFT JOIN rule_chunk_extractions p ON p.rule_id=r.id
                LEFT JOIN rule_extractions e ON e.id=p.extraction_id
                LEFT JOIN active_rule_extractions a ON a.source_hash=r.source_hash AND a.edition=r.edition
                WHERE rule_search_stemmed MATCH ? AND r.edition=?
                AND (a.extraction_id IS NULL OR p.extraction_id=a.extraction_id)
                ORDER BY bm25(rule_search_stemmed) LIMIT ?"""
        else:
            query_sql = """SELECT r.* FROM rule_search_stemmed f JOIN rules r ON r.id=f.id
                WHERE rule_search_stemmed MATCH ? AND edition=? ORDER BY bm25(rule_search_stemmed) LIMIT ?"""
        return [dict(row) for row in db.execute(query_sql, (match, edition, limit))]
