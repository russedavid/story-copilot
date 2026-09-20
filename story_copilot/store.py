from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .schema import Event


def now():
    return datetime.now(timezone.utc).isoformat()


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(
        value if isinstance(value, bytes) else value.encode()
    ).hexdigest()


def source_quote(source, quote):
    """Recover the literal span while tolerating only whitespace normalization."""
    if quote in source:
        return quote
    pieces = quote.split()
    if not pieces:
        raise ValueError("Evidence cannot be empty.")
    match = re.search(r"\s+".join(re.escape(p) for p in pieces), source)
    if not match:
        raise ValueError(
            "Evidence is not an exact source quote (apart from whitespace)."
        )
    return match.group(0)


def data_home():
    return (
        Path(os.environ.get("STORY_DATA", Path.home() / ".local/share/story-copilot"))
        .expanduser()
        .resolve()
    )


class Store:
    def __init__(self, home=None):
        self.home = Path(home or data_home()).resolve()
        if any((p / ".git").exists() for p in [self.home, *self.home.parents]):
            raise ValueError("Keep the private workspace outside source repositories.")
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        identity = self.home / "workspace.json"
        database = self.home / "library.sqlite"
        if identity.exists():
            if json.loads(identity.read_text()).get("application") != "story-copilot":
                raise ValueError(
                    "This data directory belongs to another application. Choose a new directory."
                )
        elif database.exists() and database.stat().st_size:
            raise ValueError(
                "Choose a new Story Copilot data directory; this directory contains an unrecognized database."
            )
        else:
            identity.write_text(packed({"application": "story-copilot", "format": 1}))
            identity.chmod(0o600)
        self.path = self.home / "library.sqlite"
        with self.db() as db:
            needs_stemmed_index = (
                db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='rule_search_stemmed'"
                ).fetchone()
                is None
            )
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS stories (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, split TEXT NOT NULL
                CHECK(split IN ('train','validation','test','development')));
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL,
                body TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS collections (
                id TEXT PRIMARY KEY, story_id TEXT NOT NULL REFERENCES stories(id),
                source_id TEXT NOT NULL REFERENCES sources(id), title TEXT NOT NULL,
                audio_path TEXT, metadata TEXT NOT NULL, created TEXT NOT NULL,
                UNIQUE(story_id,source_id));
            CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY, collection_id TEXT NOT NULL REFERENCES collections(id),
                ordinal INTEGER NOT NULL, speaker TEXT NOT NULL, original TEXT NOT NULL,
                start REAL, end REAL, metadata TEXT NOT NULL,
                UNIQUE(collection_id,ordinal));
            CREATE TABLE IF NOT EXISTS revisions (
                id TEXT PRIMARY KEY, turn_id TEXT NOT NULL REFERENCES turns(id),
                text TEXT NOT NULL, role TEXT NOT NULL, character TEXT NOT NULL,
                status TEXT NOT NULL, category TEXT NOT NULL, reviewer TEXT NOT NULL,
                note TEXT NOT NULL, created TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS revision_turn ON revisions(turn_id);
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, collection_id TEXT NOT NULL REFERENCES collections(id),
                anchor INTEGER NOT NULL, payload TEXT NOT NULL, revisions TEXT NOT NULL,
                created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS event_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL REFERENCES events(id), status TEXT NOT NULL,
                reviewer TEXT NOT NULL, note TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, collection_id TEXT NOT NULL REFERENCES collections(id),
                kind TEXT NOT NULL, status TEXT NOT NULL, request TEXT NOT NULL,
                result TEXT NOT NULL, created TEXT NOT NULL, finished TEXT);
            CREATE TABLE IF NOT EXISTS rules (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, page INTEGER NOT NULL,
                edition TEXT NOT NULL, text TEXT NOT NULL, source_hash TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS rule_search USING fts5(id UNINDEXED,text);
            CREATE VIRTUAL TABLE IF NOT EXISTS rule_search_stemmed USING fts5(id UNINDEXED,text,tokenize='porter unicode61');
            CREATE TRIGGER IF NOT EXISTS rules_stem_insert AFTER INSERT ON rules BEGIN
                INSERT INTO rule_search_stemmed(id,text) VALUES(new.id,new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS rules_stem_delete AFTER DELETE ON rules BEGIN
                DELETE FROM rule_search_stemmed WHERE id=old.id;
            END;
            CREATE TRIGGER IF NOT EXISTS rules_stem_update AFTER UPDATE ON rules BEGIN
                DELETE FROM rule_search_stemmed WHERE id=old.id;
                INSERT INTO rule_search_stemmed(id,text) VALUES(new.id,new.text);
            END;
            CREATE TABLE IF NOT EXISTS split_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT, story_id TEXT NOT NULL REFERENCES stories(id),
                old_split TEXT NOT NULL,new_split TEXT NOT NULL,reviewer TEXT NOT NULL,note TEXT NOT NULL,created TEXT NOT NULL);
            """)
            if needs_stemmed_index:
                db.execute("INSERT INTO rule_search_stemmed SELECT id,text FROM rules")

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def collections(self):
        with self.db() as db:
            return [
                dict(r)
                for r in db.execute("""SELECT c.*,s.split,
                (SELECT count(*) FROM turns WHERE collection_id=c.id) AS turn_count
                FROM collections c JOIN stories s ON s.id=c.story_id ORDER BY c.created DESC""")
            ]

    def collection(self, cid):
        with self.db() as db:
            r = db.execute(
                """SELECT c.*,s.split FROM collections c JOIN stories s
                ON s.id=c.story_id WHERE c.id=?""",
                (cid,),
            ).fetchone()
            if not r:
                raise ValueError("Collection not found.")
            return dict(r)

    def turns(self, cid):
        with self.db() as db:
            rows = db.execute(
                """SELECT t.*,r.id AS revision,r.text,r.role,r.character,
                r.status,r.category,r.reviewer,r.note FROM turns t JOIN revisions r
                ON r.rowid=(SELECT max(rowid) FROM revisions WHERE turn_id=t.id)
                WHERE t.collection_id=? ORDER BY t.ordinal""",
                (cid,),
            ).fetchall()
            return [dict(r) for r in rows]

    def revise(
        self,
        tid,
        *,
        text,
        role,
        character="",
        status="pending",
        category="gameplay",
        reviewer="human",
        note="",
        expected_revision=None,
        _db=None,
    ):
        if role not in {"facilitator", "player", "unknown", "mixed"}:
            raise ValueError("Invalid speaker role.")
        if status not in {"pending", "approved", "screened", "excluded"}:
            raise ValueError("Invalid review status.")
        if category not in {
            "gameplay",
            "narration",
            "dialogue",
            "mechanics",
            "production",
            "chatter",
        }:
            raise ValueError("Invalid content category.")
        if not text.strip() or not reviewer.strip():
            raise ValueError("Text and reviewer are required.")
        if status in {"approved", "screened"} and role in {"unknown", "mixed"}:
            raise ValueError(
                "Resolve or split ambiguous speakers before approving a turn."
            )
        rid = uuid4().hex
        with nullcontext(_db) if _db is not None else self.db() as db:
            if _db is None:
                db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT * FROM revisions WHERE turn_id=? ORDER BY rowid DESC LIMIT 1",
                (tid,),
            ).fetchone()
            if not prior:
                raise ValueError("Turn not found.")
            if expected_revision and prior["id"] != expected_revision:
                raise ValueError("This turn changed; reload before saving your review.")
            db.execute(
                "INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    rid,
                    tid,
                    text,
                    role,
                    character,
                    status,
                    category,
                    reviewer,
                    note,
                    now(),
                ),
            )
        return rid

    def promote_development_story(self, story, reviewer, note):
        """Move a development story into training without touching held-out splits."""
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT split FROM stories WHERE id=?", (story,)
            ).fetchone()
            if row is None:
                raise ValueError("Story not found.")
            if row["split"] == "train":
                return
            if row["split"] != "development":
                raise ValueError(
                    "Held-out validation/test stories cannot be promoted into training."
                )
            if not reviewer.strip() or not note.strip():
                raise ValueError(
                    "Record why the development story is entering training."
                )
            db.execute("UPDATE stories SET split=? WHERE id=?", ("train", story))
            db.execute(
                "INSERT INTO split_reviews(story_id,old_split,new_split,reviewer,note,created) VALUES (?,?,?,?,?,?)",
                (story, "development", "train", reviewer, note, now()),
            )

    def add_event(self, cid, event: Event):
        event = Event.model_validate(event.model_dump())
        turns = {t["ordinal"]: t for t in self.turns(cid)}
        refs = {}
        resolved_evidence = []
        for evidence in event.evidence:
            t = turns.get(evidence.turn)
            if not t:
                raise ValueError(
                    f"Evidence for turn {evidence.turn} is not an exact source quote."
                )
            quote = source_quote(t["text"], evidence.quote)
            resolved_evidence.append(evidence.model_copy(update={"quote": quote}))
            refs[str(evidence.turn)] = t["revision"]
        event = event.model_copy(update={"evidence": resolved_evidence})
        anchor = max(e.turn for e in event.evidence)
        payload = event.model_dump()
        identity = {k: v for k, v in payload.items() if k != "rationale"}
        identity["evidence"] = sorted(
            identity["evidence"], key=lambda e: (e["turn"], e["quote"])
        )
        eid = digest(cid + packed(identity) + packed(refs))[:24]
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            for ordinal, revision in refs.items():
                current = db.execute(
                    "SELECT id FROM revisions WHERE turn_id=? ORDER BY rowid DESC LIMIT 1",
                    (f"{cid}:{ordinal}",),
                ).fetchone()
                if current is None or current["id"] != revision:
                    raise ValueError(
                        "Source changed while the event was being proposed. Retry with current evidence."
                    )
            for link in [event.resolves, event.supersedes]:
                if link:
                    linked = db.execute(
                        "SELECT * FROM events WHERE id=? AND collection_id=?",
                        (link, cid),
                    ).fetchone()
                    if not linked or linked["anchor"] > anchor:
                        raise ValueError(
                            "Event links must refer to earlier evidence in this collection."
                        )
                    if (
                        link == event.resolves
                        and json.loads(linked["payload"])["kind"] != "action"
                    ):
                        raise ValueError("A resolution must refer to an action event.")
            db.execute(
                "INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?)",
                (eid, cid, anchor, packed(payload), packed(refs), now()),
            )
        return eid

    def events(self, cid):
        turns = {str(t["ordinal"]): t["revision"] for t in self.turns(cid)}
        with self.db() as db:
            rows = db.execute(
                """SELECT e.*,coalesce(r.status,'pending') AS status,
                coalesce(r.reviewer,'') AS reviewer FROM events e LEFT JOIN event_reviews r
                ON r.id=(SELECT max(id) FROM event_reviews WHERE event_id=e.id)
                WHERE collection_id=? ORDER BY anchor,e.rowid""",
                (cid,),
            ).fetchall()
        result = []
        for row in rows:
            e = dict(row)
            e["payload"] = json.loads(e["payload"])
            e["revisions"] = json.loads(e["revisions"])
            e["stale"] = any(turns.get(k) != v for k, v in e["revisions"].items())
            result.append(e)
        return result

    def review_event(self, cid, eid, status, reviewer="human", note=""):
        if status not in {"accepted", "rejected", "pending"} or not reviewer.strip():
            raise ValueError("Invalid review.")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            events = self.events(cid)
            event = next((e for e in events if e["id"] == eid), None)
            if event is None:
                raise ValueError("Event not found.")
            if status == "accepted":
                Event.model_validate(event["payload"])
                if event["stale"]:
                    raise ValueError(
                        "Source changed. Propose a fresh event from the corrected turn."
                    )
                link = event["payload"].get("resolves")
                if link:
                    target = next(e for e in events if e["id"] == link)
                    if target["status"] != "accepted" or target["stale"]:
                        raise ValueError(
                            "Accept the supported action before its resolution."
                        )
                    if any(
                        e["id"] != eid
                        and e["status"] == "accepted"
                        and not e["stale"]
                        and e["payload"].get("resolves") == link
                        and event["payload"].get("supersedes") != e["id"]
                        for e in events
                    ):
                        raise ValueError(
                            "This action already has a resolution. Supersede it explicitly to correct it."
                        )
            db.execute(
                "INSERT INTO event_reviews(event_id,status,reviewer,note,created) VALUES (?,?,?,?,?)",
                (eid, status, reviewer, note, now()),
            )

    def start_run(self, cid, kind, request):
        rid = uuid4().hex
        with self.db() as db:
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                (rid, cid, kind, "running", packed(request), "{}", now(), None),
            )
        return rid

    def finish_run(self, rid, result, status="complete"):
        with self.db() as db:
            db.execute(
                "UPDATE runs SET status=?,result=?,finished=? WHERE id=?",
                (status, packed(result), now(), rid),
            )

    def runs(self, cid):
        with self.db() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM runs WHERE collection_id=? ORDER BY created DESC",
                    (cid,),
                )
            ]
