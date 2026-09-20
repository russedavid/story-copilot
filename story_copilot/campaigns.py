"""Conversation-driven campaigns with private point-in-time Facilitator guidance.

Source-backed observations build revisable working state. Generated guidance is
never transcript speech, official game history, or a publishable player message.
Legacy decisions remain readable; normal live play has no acceptance workflow.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .store import digest, now, packed, source_quote

VISIBILITIES = {"public", "private"}
ROLES = {"facilitator", "player", "unknown"}
KINDS = {"narration", "scene", "npc", "question", "rule", "action", "state", "note"}
MAX_IMPORT_BYTES = 10 * 1024 * 1024
MAX_TEXT = 250_000


def required(text, label="Text", limit=MAX_TEXT):
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{label} is required.")
    if len(text) > limit:
        raise ValueError(f"{label} is longer than {limit:,} characters.")
    return text.strip()


def check_visibility(value):
    if value not in VISIBILITIES:
        raise ValueError("Visibility must be public or story_copilot.")
    return value


def imported_text(filename, data):
    """Decode an uploaded file, never execute it or read a browser-supplied path."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_IMPORT_BYTES:
        raise ValueError("Choose a nonempty file no larger than 10 MB.")
    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".json", ".csv", ".pdf"}:
        raise ValueError("Use a text, Markdown, JSON, CSV, or PDF file.")
    if suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise ValueError("This file is not a PDF.")
        try:
            with tempfile.TemporaryDirectory(prefix="facilitator-import-") as directory:
                source = Path(directory) / "source.pdf"
                source.write_bytes(data)
                result = subprocess.run(
                    ["pdftotext", "-layout", str(source), "-"],
                    capture_output=True,
                    timeout=30,
                    check=True,
                )
                text = result.stdout.decode("utf-8")
        except FileNotFoundError as exc:
            raise ValueError(
                "PDF text import requires pdftotext. You can paste the text instead."
            ) from exc
        except (subprocess.SubprocessError, UnicodeError) as exc:
            raise ValueError(
                "Could not extract text from this PDF. Paste text or use a searchable PDF."
            ) from exc
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeError as exc:
            raise ValueError("Text files must use UTF-8 encoding.") from exc
        if suffix == ".json":
            try:
                value = json.loads(text)
                text = packed(value)
            except (ValueError, RecursionError) as exc:
                raise ValueError("The JSON file is invalid.") from exc
        elif suffix == ".csv":
            try:
                text = "\n".join(
                    " | ".join(row) for row in csv.reader(io.StringIO(text))
                )
            except csv.Error as exc:
                raise ValueError("The CSV file is invalid.") from exc
    if "\x00" in text:
        raise ValueError("The file contains binary data.")
    return required(text, "Imported text")


def _finite_time(value):
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _created_utc(message):
    try:
        value = datetime.fromisoformat(message["created"].replace("Z", "+00:00"))
        return value.timestamp() if value.tzinfo is not None else None
    except (ValueError, KeyError, AttributeError):
        return None


def order_conversation(messages):
    """Sort comparable capture times without rewriting ingestion order.

    Incomparable clocks retain their ingestion *slots*. Their relative positions
    are only deterministic display anchors, never asserted event chronology.
    Replay seconds cannot join the UTC bucket even if malformed input supplies
    a field called utc_start. Interval overlap is reported, not invented away.
    """
    rows, buckets, notes = [], {}, set()
    unknown = 0
    for message in messages:
        row = dict(message)
        source = message.get("source", {})
        timeline = source.get("timeline")
        timeline = timeline if isinstance(timeline, dict) else {}
        kind = timeline.get("kind")
        domain = timeline.get("domain")
        start, end, group, basis = None, None, None, "ingestion_only"
        if (
            kind == "recording"
            and isinstance(domain, str)
            and domain
            and _finite_time(timeline.get("start_seconds"))
        ):
            start, end, group, basis = (
                timeline["start_seconds"],
                timeline.get("end_seconds"),
                "recording:" + domain,
                "recording_relative",
            )
        elif kind == "live_capture" and isinstance(domain, str) and domain:
            if _finite_time(timeline.get("utc_start")):
                start, end, group, basis = (
                    timeline["utc_start"],
                    timeline.get("utc_end"),
                    "utc",
                    "estimated_capture_utc",
                )
            elif _finite_time(timeline.get("start_seconds")):
                start, end, group, basis = (
                    timeline["start_seconds"],
                    timeline.get("end_seconds"),
                    "capture:" + domain,
                    "shared_capture_relative",
                )
        elif kind == "manual_entry" and _finite_time(timeline.get("utc_start")):
            start, end, group, basis = (
                timeline["utc_start"],
                timeline.get("utc_end"),
                "utc",
                "manual_entry_utc",
            )
        elif (
            not timeline
            and source.get("kind") in {None, "manual"}
            and not source.get("channel")
        ):
            start = _created_utc(message)
            if start is not None:
                end, group, basis = start, "utc", "manual_entry_utc"
        if not _finite_time(end) or (start is not None and end < start):
            end = None
        if group is None:
            unknown += 1
            notes.add(
                "Untimed or uncertain-clock messages retain ingestion positions; their event order is not known."
            )
        else:
            buckets.setdefault(group, []).append(len(rows))
        if kind == "recording":
            notes.add(
                "Recording times are relative to that source only; their order against manual/live conversation is unknown."
            )
        if basis == "estimated_capture_utc":
            notes.add(
                "Live speech uses estimated capture UTC; cross-host synchronization and sub-second ordering are not calibrated."
            )
        if basis == "manual_entry_utc":
            notes.add(
                "Manual timestamps describe when text was entered, not when any recounted event occurred."
            )
        if isinstance(timeline.get("uncertainty"), str) and timeline["uncertainty"]:
            notes.add(timeline["uncertainty"])
        row["ingestion_ordinal"] = row["ordinal"]
        row["chronology"] = {
            "clock_group": group,
            "source_domain": domain,
            "basis": basis,
            "start": start,
            "end": end,
            "overlaps_prior": False,
        }
        rows.append(row)
    overlaps = 0
    for indices in buckets.values():
        ordered = sorted(
            (rows[i] for i in indices),
            key=lambda m: (m["chronology"]["start"], m["ordinal"]),
        )
        end_seen = None
        for row in ordered:
            timing = row["chronology"]
            if end_seen is not None and timing["start"] < end_seen:
                timing["overlaps_prior"] = True
                overlaps += 1
            if timing["end"] is not None:
                end_seen = max(
                    end_seen if end_seen is not None else timing["end"], timing["end"]
                )
        for index, row in zip(indices, ordered, strict=True):
            rows[index] = row
    partial = unknown > 0 or len(buckets) > 1
    if partial:
        notes.add(
            "Only messages within the same clock group are chronologically ordered. Positions between incompatible groups are ingestion-based display anchors, not evidence of a response or causal sequence."
        )
    if overlaps:
        notes.add(
            "Some speech intervals overlap; ordering by start time does not establish who answered whom."
        )
    for i, row in enumerate(rows, 1):
        row["order_index"] = i
        row["chronology"]["cross_group_order_known"] = not partial
    return {
        "messages": rows,
        "chronology": {
            "policy_version": 1,
            "partial_order": partial,
            "clock_groups": sorted(buckets),
            "unknown_clock_messages": unknown,
            "overlapping_messages": overlaps,
            "notes": sorted(notes),
        },
    }


def snapshot_change(previous, current):
    """Allow private as-of drafts only for provably later append-only speech."""
    if previous.get("session_id") != current.get("session_id"):
        return "invalidated"
    if previous.get("evidence_hash") == current.get("evidence_hash"):
        return "unchanged"
    if previous.get("authority_hash") != current.get("authority_hash"):
        return "invalidated"
    old, new = previous.get("sources", []), current.get("sources", [])
    if len(new) <= len(old) or new[: len(old)] != old:
        return "invalidated"
    if previous.get("partial_order") or current.get("partial_order"):
        return "invalidated"
    groups = {source.get("clock_group") for source in new}
    if len(groups) != 1 or None in groups:
        return "invalidated"
    through = max(
        (
            source["end"] if source.get("end") is not None else source["start"]
            for source in old
        ),
        default=None,
    )
    if any(
        source.get("start") is None
        or (through is not None and source["start"] < through)
        for source in new[len(old) :]
    ):
        return "invalidated"
    return "append_only"


class Campaigns:
    def __init__(self, store):
        self.store = store
        with store.db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS play_campaigns (
                id TEXT PRIMARY KEY,title TEXT NOT NULL,system TEXT NOT NULL,
                direction TEXT NOT NULL,style TEXT NOT NULL,created TEXT NOT NULL,updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS play_rule_profiles (
                campaign_id TEXT PRIMARY KEY REFERENCES play_campaigns(id),
                body TEXT NOT NULL,revision INTEGER NOT NULL,updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS play_characters (
                id TEXT PRIMARY KEY,campaign_id TEXT NOT NULL REFERENCES play_campaigns(id),
                name TEXT NOT NULL,sheet TEXT NOT NULL,visibility TEXT NOT NULL,
                revision INTEGER NOT NULL,updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS play_participants (
                id TEXT PRIMARY KEY,campaign_id TEXT NOT NULL REFERENCES play_campaigns(id),
                name TEXT NOT NULL,role TEXT NOT NULL,speaker TEXT NOT NULL,character_id TEXT,
                UNIQUE(campaign_id,speaker));
            CREATE TABLE IF NOT EXISTS play_documents (
                id TEXT PRIMARY KEY,campaign_id TEXT NOT NULL REFERENCES play_campaigns(id),
                title TEXT NOT NULL,text TEXT NOT NULL,visibility TEXT NOT NULL,
                sha256 TEXT NOT NULL,metadata TEXT NOT NULL,created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS play_sessions (
                id TEXT PRIMARY KEY,campaign_id TEXT NOT NULL REFERENCES play_campaigns(id),
                ordinal INTEGER NOT NULL,title TEXT NOT NULL,parent_id TEXT,
                proactive INTEGER NOT NULL DEFAULT 1,created TEXT NOT NULL,
                UNIQUE(campaign_id,ordinal));
            CREATE TABLE IF NOT EXISTS play_session_kinds (
                session_id TEXT PRIMARY KEY REFERENCES play_sessions(id),
                kind TEXT NOT NULL CHECK(kind IN ('fresh','continue','branch')));
            CREATE TABLE IF NOT EXISTS play_campaign_progress (
                campaign_id TEXT PRIMARY KEY REFERENCES play_campaigns(id),
                current_session_id TEXT NOT NULL REFERENCES play_sessions(id));
            CREATE TABLE IF NOT EXISTS play_demo_imports (
                slug TEXT PRIMARY KEY,campaign_id TEXT NOT NULL REFERENCES play_campaigns(id),
                session_id TEXT NOT NULL REFERENCES play_sessions(id),created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS play_messages (
                id TEXT PRIMARY KEY,session_id TEXT NOT NULL REFERENCES play_sessions(id),
                ordinal INTEGER NOT NULL,speaker TEXT NOT NULL,role TEXT NOT NULL,character TEXT NOT NULL,
                text TEXT NOT NULL,visibility TEXT NOT NULL,source TEXT NOT NULL,external_id TEXT,
                created TEXT NOT NULL,UNIQUE(session_id,ordinal),UNIQUE(session_id,external_id));
            CREATE TABLE IF NOT EXISTS play_message_revisions (
                id TEXT PRIMARY KEY,message_id TEXT NOT NULL REFERENCES play_messages(id),
                speaker TEXT NOT NULL,role TEXT NOT NULL,character TEXT NOT NULL,
                text TEXT NOT NULL,visibility TEXT NOT NULL,note TEXT NOT NULL,created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS play_proposals (
                id TEXT PRIMARY KEY,session_id TEXT NOT NULL REFERENCES play_sessions(id),
                run_id TEXT,kind TEXT NOT NULL,title TEXT NOT NULL,text TEXT NOT NULL,
                visibility TEXT NOT NULL,payload TEXT NOT NULL,evidence TEXT NOT NULL,
                parent_id TEXT,created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS play_observations (
                proposal_id TEXT PRIMARY KEY REFERENCES play_proposals(id),created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS play_publications (
                id TEXT PRIMARY KEY,session_id TEXT NOT NULL REFERENCES play_sessions(id),
                proposal_id TEXT NOT NULL REFERENCES play_proposals(id),
                text TEXT NOT NULL,created TEXT NOT NULL,withdrawn TEXT);
            CREATE TABLE IF NOT EXISTS play_decisions (
                id TEXT PRIMARY KEY,session_id TEXT NOT NULL REFERENCES play_sessions(id),
                proposal_id TEXT,action TEXT NOT NULL,target_id TEXT,undone_by TEXT,note TEXT NOT NULL,
                created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS play_runs (
                id TEXT PRIMARY KEY,session_id TEXT NOT NULL REFERENCES play_sessions(id),
                context_hash TEXT NOT NULL,status TEXT NOT NULL,request TEXT NOT NULL,
                result TEXT NOT NULL,created TEXT NOT NULL,finished TEXT);
            CREATE TABLE IF NOT EXISTS play_edits (
                id TEXT PRIMARY KEY,campaign_id TEXT NOT NULL,entity_id TEXT NOT NULL,
                before_json TEXT NOT NULL,after_json TEXT NOT NULL,created TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS play_message_session ON play_messages(session_id,ordinal);
            CREATE INDEX IF NOT EXISTS play_run_session ON play_runs(session_id,created);
            """)

    def _one(self, table, identifier, db=None):
        if db is None:
            with self.store.db() as connection:
                return self._one(table, identifier, connection)
        row = db.execute(f"SELECT * FROM {table} WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("Campaign item not found.")
        return dict(row)

    def campaigns(self):
        with self.store.db() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM play_campaigns ORDER BY updated DESC,id"
                )
            ]

    def campaign(self, identifier):
        return self._one("play_campaigns", identifier)

    def rule_profile(self, identifier):
        from .profiles import DEFAULT
        from copy import deepcopy

        self.campaign(identifier)
        with self.store.db() as db:
            row = db.execute(
                "SELECT body FROM play_rule_profiles WHERE campaign_id=?", (identifier,)
            ).fetchone()
        return json.loads(row["body"]) if row else deepcopy(DEFAULT)

    def set_rule_profile(self, identifier, profile):
        from .profiles import validate_profile

        profile = validate_profile(profile)
        before = self.rule_profile(identifier)
        with self.store.db() as db:
            db.execute(
                "INSERT INTO play_rule_profiles VALUES(?,?,1,?) ON CONFLICT(campaign_id) DO UPDATE SET body=excluded.body,revision=play_rule_profiles.revision+1,updated=excluded.updated",
                (identifier, packed(profile), now()),
            )
            self._edit(db, identifier, identifier + ":rule-profile", before, profile)
        return profile

    def create(self, title, system="Custom rules", direction="", style=""):
        if len(direction) + len(style) > 20_000:
            raise ValueError(
                "Facilitator direction and style must fit within 20,000 characters."
            )
        identifier = uuid4().hex
        with self.store.db() as db:
            db.execute(
                "INSERT INTO play_campaigns VALUES(?,?,?,?,?,?,?)",
                (
                    identifier,
                    required(title, "Campaign title", 200),
                    required(system, "System", 120),
                    direction.strip(),
                    style.strip(),
                    now(),
                    now(),
                ),
            )
        return identifier

    def update(self, identifier, *, title, direction="", style=""):
        before = self.campaign(identifier)
        after = {
            **before,
            "title": required(title, "Campaign title", 200),
            "direction": direction.strip(),
            "style": style.strip(),
            "updated": now(),
        }
        if len(direction) + len(style) > 20_000:
            raise ValueError(
                "Facilitator direction and style must fit within 20,000 characters."
            )
        with self.store.db() as db:
            db.execute(
                "UPDATE play_campaigns SET title=?,direction=?,style=?,updated=? WHERE id=?",
                (
                    after["title"],
                    after["direction"],
                    after["style"],
                    after["updated"],
                    identifier,
                ),
            )
            self._edit(db, identifier, identifier, before, after)

    def _edit(self, db, campaign_id, entity_id, before, after):
        db.execute(
            "INSERT INTO play_edits VALUES(?,?,?,?,?,?)",
            (uuid4().hex, campaign_id, entity_id, packed(before), packed(after), now()),
        )

    def documents(self, campaign_id, public_only=False):
        with self.store.db() as db:
            rows = db.execute(
                "SELECT * FROM play_documents WHERE campaign_id=? ORDER BY rowid",
                (campaign_id,),
            )
            return [
                dict(r) for r in rows if not public_only or r["visibility"] == "public"
            ]

    def add_document(
        self, campaign_id, title, text, *, visibility="private", metadata=None
    ):
        self.campaign(campaign_id)
        check_visibility(visibility)
        text = required(text, "Scenario or reference text")
        identifier = uuid4().hex
        with self.store.db() as db:
            db.execute(
                "INSERT INTO play_documents VALUES(?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    campaign_id,
                    required(title, "Document title", 200),
                    text,
                    visibility,
                    digest(text),
                    packed(metadata or {}),
                    now(),
                ),
            )
        return identifier

    def delete_document(self, campaign_id, document_id):
        with self.store.db() as db:
            before = self._one("play_documents", document_id, db)
            if before["campaign_id"] != campaign_id:
                raise ValueError("This document belongs to a different campaign.")
            self._edit(db, campaign_id, document_id, before, {"removed": True})
            db.execute("DELETE FROM play_documents WHERE id=?", (document_id,))

    def characters(self, campaign_id, public_only=False):
        with self.store.db() as db:
            return [
                {**dict(r), "sheet": json.loads(r["sheet"])}
                for r in db.execute(
                    "SELECT * FROM play_characters WHERE campaign_id=? ORDER BY rowid",
                    (campaign_id,),
                )
                if not public_only or r["visibility"] == "public"
            ]

    def save_character(
        self,
        campaign_id,
        name,
        sheet,
        *,
        character_id=None,
        visibility="public",
        expected_revision=None,
    ):
        self.campaign(campaign_id)
        check_visibility(visibility)
        if not isinstance(sheet, dict) or len(packed(sheet)) > 30_000:
            raise ValueError(
                "A character sheet must be a JSON object of at most 30,000 characters."
            )
        resources = sheet.get("resources", {})
        if not isinstance(resources, dict) or any(
            not isinstance(k, str)
            or not k
            or (v is not None and (not isinstance(v, int) or isinstance(v, bool)))
            for k, v in resources.items()
        ):
            raise ValueError(
                "Resources must map names to integer totals, or null when unknown."
            )
        identifier = character_id or uuid4().hex
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            before = (
                self._one("play_characters", identifier, db) if character_id else {}
            )
            if before and before["campaign_id"] != campaign_id:
                raise ValueError("This character belongs to a different campaign.")
            if (
                before
                and expected_revision is not None
                and before["revision"] != int(expected_revision)
            ):
                raise ValueError("This sheet changed. Reload before saving.")
            name = required(name, "Character name", 120)
            duplicate = db.execute(
                "SELECT id FROM play_characters WHERE campaign_id=? AND name=? COLLATE NOCASE AND id!=?",
                (campaign_id, name, identifier),
            ).fetchone()
            if duplicate:
                raise ValueError(
                    "Character names must be distinct within this campaign."
                )
            revision = before.get("revision", 0) + 1
            after = {
                "name": required(name, "Character name", 120),
                "sheet": sheet,
                "visibility": visibility,
                "revision": revision,
            }
            db.execute(
                "INSERT INTO play_characters VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,sheet=excluded.sheet,visibility=excluded.visibility,revision=excluded.revision,updated=excluded.updated",
                (
                    identifier,
                    campaign_id,
                    after["name"],
                    packed(sheet),
                    visibility,
                    revision,
                    now(),
                ),
            )
            self._edit(db, campaign_id, identifier, before, after)
        return identifier

    def participants(self, campaign_id):
        with self.store.db() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM play_participants WHERE campaign_id=? ORDER BY rowid",
                    (campaign_id,),
                )
            ]

    def speaker_mappings(self, campaign_id):
        """Explicit user mappings, separate from immutable transcript attribution."""
        characters = {c["id"]: c["name"] for c in self.characters(campaign_id)}
        return {
            p["speaker"]: {
                "speaker": p["speaker"],
                "role": p["role"],
                "character_id": p["character_id"],
                "character": characters.get(p["character_id"], ""),
            }
            for p in self.participants(campaign_id)
        }

    def map_participant(
        self, campaign_id, name, speaker, *, role="player", character_id=None
    ):
        self.campaign(campaign_id)
        if role not in ROLES:
            raise ValueError("Choose facilitator, player, or unknown.")
        if character_id and not any(
            c["id"] == character_id for c in self.characters(campaign_id)
        ):
            raise ValueError("Select a character from this campaign.")
        with self.store.db() as db:
            db.execute(
                "INSERT INTO play_participants VALUES(?,?,?,?,?,?) ON CONFLICT(campaign_id,speaker) DO UPDATE SET name=excluded.name,role=excluded.role,character_id=excluded.character_id",
                (
                    uuid4().hex,
                    campaign_id,
                    required(name, "Participant name", 120),
                    role,
                    required(speaker, "Speaker label", 120),
                    character_id or None,
                ),
            )

    def sessions(self, campaign_id):
        with self.store.db() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT s.*,coalesce(k.kind,'legacy') kind FROM play_sessions s LEFT JOIN play_session_kinds k ON k.session_id=s.id WHERE s.campaign_id=? ORDER BY s.ordinal",
                    (campaign_id,),
                )
            ]

    def session(self, identifier):
        result = self._one("play_sessions", identifier)
        with self.store.db() as db:
            row = db.execute(
                "SELECT kind FROM play_session_kinds WHERE session_id=?", (identifier,)
            ).fetchone()
        result["kind"] = row["kind"] if row else "legacy"
        return result

    def current_session(self, campaign_id):
        """The explicitly chosen continuation, never simply the newest branch."""
        with self.store.db() as db:
            row = db.execute(
                "SELECT current_session_id FROM play_campaign_progress WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
        return self.session(row["current_session_id"]) if row else None

    def create_session(
        self, campaign_id, title, *, parent_id=None, kind="fresh", proactive=True
    ):
        self.campaign(campaign_id)
        if kind not in {"fresh", "continue", "branch"} or (kind == "fresh") != (
            parent_id is None
        ):
            raise ValueError(
                "Fresh sessions have no parent; continuations and branches need a source session."
            )
        if parent_id and self.session(parent_id)["campaign_id"] != campaign_id:
            raise ValueError("A copied session must belong to its parent's campaign.")
        identifier = uuid4().hex
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            ordinal = db.execute(
                "SELECT coalesce(max(ordinal),0)+1 FROM play_sessions WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()[0]
            db.execute(
                "INSERT INTO play_sessions VALUES(?,?,?,?,?,?,?)",
                (
                    identifier,
                    campaign_id,
                    ordinal,
                    required(title, "Session title", 200),
                    parent_id,
                    int(bool(proactive)),
                    now(),
                ),
            )
            db.execute("INSERT INTO play_session_kinds VALUES(?,?)", (identifier, kind))
            if kind == "fresh":
                db.execute(
                    "INSERT INTO play_campaign_progress VALUES(?,?) ON CONFLICT(campaign_id) DO UPDATE SET current_session_id=excluded.current_session_id",
                    (campaign_id, identifier),
                )
        return identifier

    def set_proactive(self, session_id, enabled):
        self.session(session_id)
        with self.store.db() as db:
            db.execute(
                "UPDATE play_sessions SET proactive=? WHERE id=?",
                (bool(enabled), session_id),
            )

    def add_message(
        self,
        session_id,
        speaker,
        text,
        *,
        role="unknown",
        visibility="public",
        source=None,
        external_id=None,
    ):
        session = self.session(session_id)
        check_visibility(visibility)
        if role not in ROLES:
            raise ValueError("Invalid speaker role.")
        speaker = required(speaker, "Speaker", 120)
        text = required(text, "Conversation text", 20_000)
        participant = next(
            (
                p
                for p in self.participants(session["campaign_id"])
                if p["speaker"] == speaker
            ),
            None,
        )
        character = ""
        if participant:
            if role == "unknown":
                role = participant["role"]
            if participant["character_id"]:
                character = next(
                    (
                        c["name"]
                        for c in self.characters(session["campaign_id"])
                        if c["id"] == participant["character_id"]
                    ),
                    "",
                )
        identifier = uuid4().hex
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if external_id is not None:
                found = db.execute(
                    "SELECT * FROM play_messages WHERE session_id=? AND external_id=?",
                    (session_id, external_id),
                ).fetchone()
                if found:
                    if (
                        found["text"] != text
                        or found["speaker"] != speaker
                        or json.loads(found["source"]) != (source or {})
                    ):
                        raise ValueError(
                            "This source ID was already ingested with different content."
                        )
                    return found["id"]
            ordinal = db.execute(
                "SELECT coalesce(max(ordinal),0)+1 FROM play_messages WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
            db.execute(
                "INSERT INTO play_messages VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    session_id,
                    ordinal,
                    speaker,
                    role,
                    character,
                    text,
                    visibility,
                    packed(source or {}),
                    external_id,
                    now(),
                ),
            )
        return identifier

    def messages(self, session_id, public_only=False):
        with self.store.db() as db:
            rows = list(
                db.execute(
                    "SELECT * FROM play_messages WHERE session_id=? ORDER BY ordinal",
                    (session_id,),
                )
            )
            result = []
            for row in rows:
                message = {
                    **dict(row),
                    "source": json.loads(row["source"]),
                    "revision": row["id"],
                    "original": row["text"],
                }
                latest = db.execute(
                    "SELECT * FROM play_message_revisions WHERE message_id=? ORDER BY rowid DESC LIMIT 1",
                    (row["id"],),
                ).fetchone()
                if latest:
                    for key in ("speaker", "role", "character", "text", "visibility"):
                        message[key] = latest[key]
                    message["revision"] = latest["id"]
                if not public_only or message["visibility"] == "public":
                    if public_only:
                        message.pop("original", None)
                        message.pop("source", None)
                    result.append(message)
            return result

    def conversation(self, session_id, public_only=False):
        messages = self.messages(session_id)
        if public_only:
            messages = [m for m in messages if m["visibility"] == "public"]
        result = order_conversation(messages)
        if public_only:
            for message in result["messages"]:
                message.pop("source", None)
                message.pop("original", None)
        return result

    def revise_message(
        self,
        session_id,
        message_id,
        *,
        text,
        speaker,
        role,
        character="",
        visibility="private",
        expected_revision=None,
        note="",
    ):
        original = self._one("play_messages", message_id)
        if original["session_id"] != session_id:
            raise ValueError("This message belongs to another session.")
        check_visibility(visibility)
        if role not in ROLES:
            raise ValueError("Invalid speaker role.")
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT id FROM play_message_revisions WHERE message_id=? ORDER BY rowid DESC LIMIT 1",
                (message_id,),
            ).fetchone()
            current = prior["id"] if prior else message_id
            if expected_revision and expected_revision != current:
                raise ValueError("This message changed. Reload before saving.")
            revision = uuid4().hex
            db.execute(
                "INSERT INTO play_message_revisions VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    revision,
                    message_id,
                    required(speaker, "Speaker", 120),
                    role,
                    character.strip(),
                    required(text, "Conversation text", 20_000),
                    visibility,
                    note.strip(),
                    now(),
                ),
            )
        return revision

    def proposals(self, session_id, public_only=False):
        messages = {m["id"]: m for m in self.messages(session_id)}
        mappings = self.speaker_mappings(self.session(session_id)["campaign_id"])
        with self.store.db() as db:
            rows = db.execute(
                """SELECT p.*,coalesce((SELECT action FROM play_decisions d
                WHERE d.proposal_id=p.id AND d.undone_by IS NULL AND d.action IN ('accepted','rejected','pending')
                ORDER BY d.rowid DESC LIMIT 1),'pending') status,
                coalesce((SELECT d.rowid FROM play_decisions d WHERE d.proposal_id=p.id AND d.undone_by IS NULL
                AND d.action IN ('accepted','rejected','pending') ORDER BY d.rowid DESC LIMIT 1),0) decision_order,
                coalesce((SELECT d.created FROM play_decisions d WHERE d.proposal_id=p.id AND d.undone_by IS NULL
                ORDER BY d.rowid DESC LIMIT 1),p.created) decision_time,
                EXISTS(SELECT 1 FROM play_observations o WHERE o.proposal_id=p.id) observed
                FROM play_proposals p WHERE p.session_id=? ORDER BY p.rowid""",
                (session_id,),
            )
            result = []
            for row in rows:
                proposal = {
                    **dict(row),
                    "payload": json.loads(row["payload"]),
                    "evidence": json.loads(row["evidence"]),
                }
                stale = False
                for evidence in proposal["evidence"]:
                    message = messages.get(evidence["message_id"])
                    if message is None or (
                        evidence.get("revision")
                        and evidence["revision"] != message["revision"]
                    ):
                        stale = True
                    elif evidence["quote"] not in message["text"]:
                        stale = True
                    if evidence.get("speaker_mapping"):
                        mapped = mappings.get(message["speaker"]) if message else None
                        if mapped != evidence["speaker_mapping"]:
                            stale = True
                    if public_only and message and message["visibility"] != "public":
                        stale = True
                proposal["stale"] = stale
                if not public_only or (
                    proposal["visibility"] == "public"
                    and (
                        proposal["status"] == "accepted"
                        or proposal["observed"]
                        and proposal["status"] != "rejected"
                    )
                    and not stale
                ):
                    result.append(proposal)
            return result

    def _validate_proposal(self, session_id, proposal):
        if not isinstance(proposal, dict) or proposal.get("kind") not in KINDS:
            raise ValueError("Invalid suggestion type.")
        result = {
            "kind": proposal["kind"],
            "title": required(
                proposal.get("title", "Suggestion"), "Suggestion title", 200
            ),
            "text": required(proposal.get("text", ""), "Suggestion", 20_000),
            "visibility": check_visibility(proposal.get("visibility", "private")),
            "payload": proposal.get("payload", {}),
            "evidence": proposal.get("evidence", []),
        }
        if not isinstance(result["payload"], dict) or not isinstance(
            result["evidence"], list
        ):
            raise ValueError("Suggestion payload and evidence are invalid.")
        messages = {m["id"]: m for m in self.messages(session_id)}
        result["evidence"] = [
            dict(e) if isinstance(e, dict) else e for e in result["evidence"]
        ]
        for evidence in result["evidence"]:
            if (
                not isinstance(evidence, dict)
                or evidence.get("message_id") not in messages
            ):
                raise ValueError(
                    "Evidence must reference a conversation message from this session."
                )
            message = messages[evidence["message_id"]]
            evidence["quote"] = source_quote(
                message["text"], required(evidence.get("quote", ""), "Evidence quote")
            )
            evidence["revision"] = message["revision"]
            if evidence.get("speaker_mapping"):
                current = self.speaker_mappings(
                    self.session(session_id)["campaign_id"]
                ).get(message["speaker"])
                if current != evidence["speaker_mapping"]:
                    raise ValueError(
                        "Speaker mapping changed. Reanalyze this source with the current user mapping."
                    )
            if message["visibility"] == "private" and result["visibility"] == "public":
                raise ValueError(
                    "A public suggestion cannot cite private conversation."
                )
        changes = result["payload"].get("state_changes", [])
        if not isinstance(changes, list) or len(changes) > 30:
            raise ValueError("State changes must be a list of at most 30 entries.")
        for change in changes:
            if not isinstance(change, dict) or change.get("kind") not in {
                "fact",
                "resource",
                "pending",
                "resolve",
                "knowledge",
                "claim",
            }:
                raise ValueError("Invalid proposed state change.")
            required(change.get("entity", ""), "State entity", 200)
            required(change.get("attribute", ""), "State attribute", 200)
            check_visibility(change.get("visibility", result["visibility"]))
            if (
                result["visibility"] == "public"
                and change.get("visibility") == "private"
            ):
                raise ValueError("Private state must be in a private suggestion.")
            if change["kind"] == "resource":
                total, delta = change.get("value"), change.get("delta")
                if (
                    (total is None) == (delta is None)
                    or not isinstance(total if total is not None else delta, int)
                    or isinstance(total if total is not None else delta, bool)
                ):
                    raise ValueError(
                        "A resource change requires exactly one integer total or delta."
                    )
            elif change.get("value") is None:
                raise ValueError("A state change requires a value.")
            if change["kind"] == "resolve" and not change.get("action_id"):
                raise ValueError("A resolution must identify the pending action.")
        return result

    def add_proposal(
        self, session_id, proposal, *, run_id=None, parent_id=None, _db=None
    ):
        self.session(session_id)
        proposal = self._validate_proposal(session_id, proposal)
        identifier = uuid4().hex
        values = (
            identifier,
            session_id,
            run_id,
            proposal["kind"],
            proposal["title"],
            proposal["text"],
            proposal["visibility"],
            packed(proposal["payload"]),
            packed(proposal["evidence"]),
            parent_id,
            now(),
        )
        if _db is not None:
            _db.execute(
                "INSERT INTO play_proposals VALUES(?,?,?,?,?,?,?,?,?,?,?)", values
            )
        else:
            with self.store.db() as db:
                db.execute(
                    "INSERT INTO play_proposals VALUES(?,?,?,?,?,?,?,?,?,?,?)", values
                )
        return identifier

    def is_observation(self, proposal):
        """Only validated, quoted extraction events can drive conversation state."""
        from .schema import Event

        payload = proposal.get("payload", {})
        if not payload.get("source_event") or not proposal.get("evidence"):
            return False
        try:
            event = Event.model_validate(payload["source_event"])
            changes = payload.get("state_changes", [])
            if len(changes) != 1:
                return False
            change = changes[0]
            kind = {
                "entity": "fact",
                "action": "claim" if event.stage == "hypothetical" else "pending",
            }.get(event.kind, event.kind)
            return (
                proposal.get("kind") == "state"
                and change.get("kind") == kind
                and all(
                    change.get(k) == getattr(event, k)
                    for k in (
                        "entity",
                        "attribute",
                        "value",
                        "delta",
                        "stage",
                        "supersedes",
                    )
                )
                and change.get("action_id") == event.resolves
                and bool(payload.get("source_event_sha256"))
            )
        except (ValueError, TypeError):
            return False

    def record_observation(self, proposal_id, *, _db=None):
        # Called only after the source snapshot and event have been validated.
        if _db is not None:
            _db.execute(
                "INSERT OR IGNORE INTO play_observations VALUES(?,?)",
                (proposal_id, now()),
            )
        else:
            with self.store.db() as db:
                db.execute(
                    "INSERT OR IGNORE INTO play_observations VALUES(?,?)",
                    (proposal_id, now()),
                )

    def decide(self, session_id, proposal_id, action, note=""):
        if action not in {"accepted", "rejected", "pending"}:
            raise ValueError("Choose accept, reject, or reset.")
        proposal = self._one("play_proposals", proposal_id)
        if proposal["session_id"] != session_id:
            raise ValueError("This suggestion belongs to a different session.")
        if action == "accepted":
            if next(p for p in self.proposals(session_id) if p["id"] == proposal_id)[
                "stale"
            ]:
                raise ValueError(
                    "Source changed. Create a corrected suggestion before accepting it."
                )
            pending = self.state(session_id)["pending"]
            for change in json.loads(proposal["payload"]).get("state_changes", []):
                if change["kind"] == "resolve" and change["action_id"] not in pending:
                    raise ValueError(
                        "The action to resolve is no longer pending. Review the current state."
                    )
        identifier = uuid4().hex
        with self.store.db() as db:
            db.execute(
                "INSERT INTO play_decisions VALUES(?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    session_id,
                    proposal_id,
                    action,
                    None,
                    None,
                    note.strip(),
                    now(),
                ),
            )
            if action != "accepted":
                db.execute(
                    "UPDATE play_publications SET withdrawn=? WHERE proposal_id=? AND withdrawn IS NULL",
                    (now(), proposal_id),
                )
        return identifier

    def publish(self, session_id, proposal_id, text):
        raise ValueError(
            "Facilitator suggestions are private, point-in-time guidance and cannot be published to players."
        )

    def publications(self, session_id):
        allowed = {
            p["id"]
            for p in self.proposals(session_id)
            if p["status"] == "accepted" and not p["stale"]
        }
        with self.store.db() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM play_publications WHERE session_id=? AND withdrawn IS NULL ORDER BY rowid",
                    (session_id,),
                )
                if r["proposal_id"] in allowed
            ]

    def withdraw(self, session_id, publication_id):
        with self.store.db() as db:
            found = db.execute(
                "SELECT id FROM play_publications WHERE id=? AND session_id=?",
                (publication_id, session_id),
            ).fetchone()
            if not found:
                raise ValueError("Published wording not found in this session.")
            db.execute(
                "UPDATE play_publications SET withdrawn=? WHERE id=?",
                (now(), publication_id),
            )

    def undo(self, session_id):
        self.session(session_id)
        identifier = uuid4().hex
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT * FROM play_decisions WHERE session_id=? AND action!='undo' AND undone_by IS NULL ORDER BY rowid DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if prior is None:
                raise ValueError("There is no suggestion decision to undo.")
            db.execute(
                "INSERT INTO play_decisions VALUES(?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    session_id,
                    prior["proposal_id"],
                    "undo",
                    prior["id"],
                    None,
                    "Undo last decision",
                    now(),
                ),
            )
            db.execute(
                "UPDATE play_publications SET withdrawn=? WHERE proposal_id=? AND withdrawn IS NULL",
                (now(), prior["proposal_id"]),
            )
            db.execute(
                "UPDATE play_decisions SET undone_by=? WHERE id=?",
                (identifier, prior["id"]),
            )
        return identifier

    def continue_session(self, campaign_id, title, *, parent_id=None):
        self.campaign(campaign_id)
        parent = (
            self.session(parent_id) if parent_id else self.current_session(campaign_id)
        )
        if parent is None:
            raise ValueError(
                "Select the session to continue; this campaign has no chosen continuation yet."
            )
        if parent["campaign_id"] != campaign_id:
            raise ValueError("Choose a source session from this campaign.")
        new = self._copy_session(parent["id"], title, kind="continue")
        with self.store.db() as db:
            db.execute(
                "INSERT INTO play_campaign_progress VALUES(?,?) ON CONFLICT(campaign_id) DO UPDATE SET current_session_id=excluded.current_session_id",
                (campaign_id, new),
            )
        return new

    def branch(self, session_id, title):
        return self._copy_session(session_id, title, kind="branch")

    def _copy_session(self, session_id, title, *, kind):
        session = self.session(session_id)
        snapshot = self.snapshot(session_id)
        processed = {}
        identity = digest(packed(self.speaker_mappings(session["campaign_id"])))
        for run in reversed(self.runs(session_id)):
            if (
                run["status"] in {"complete", "superseded"}
                and run["request"].get("work_protocol") == "facilitator-copilot-v2"
                and run["request"].get("snapshot_guard")
                and self.snapshot_change(run["request"]["snapshot_guard"], snapshot)
                in {"unchanged", "append_only"}
            ):
                analysis = run["result"].get("trace", {}).get("classification", {})
                if analysis.get("identity_hash") == identity:
                    processed.update(analysis.get("processed_source_revisions", {}))
        new = self.create_session(
            session["campaign_id"],
            title,
            parent_id=session_id,
            kind=kind,
            proactive=session["proactive"],
        )
        mapping = {}
        for message in self.messages(session_id):
            branch_source = dict(message["source"])
            if processed.get(message["id"]) == message["revision"]:
                branch_source["inherited_analysis"] = {
                    "content_hash": digest(
                        packed(
                            {
                                key: message.get(key)
                                for key in (
                                    "speaker",
                                    "role",
                                    "character",
                                    "text",
                                    "visibility",
                                )
                            }
                        )
                    ),
                    "identity_hash": identity,
                }
            if (
                not branch_source.get("timeline")
                and branch_source.get("kind") in {None, "manual"}
                and not branch_source.get("channel")
            ):
                stamp = _created_utc(message)
                if stamp is not None:
                    branch_source["timeline"] = {
                        "kind": "manual_entry",
                        "domain": "manual-entry-utc",
                        "utc_start": stamp,
                        "utc_end": stamp,
                        "quality": "original manual entry UTC",
                        "uncertainty": "Manual entry time is not the time of a recounted event.",
                    }
            mapping[message["id"]] = self.add_message(
                new,
                message["speaker"],
                message["text"],
                role=message["role"],
                visibility=message["visibility"],
                source={
                    **branch_source,
                    "branched_from_message": message["id"],
                    "branched_from_session": session_id,
                    "session_copy_kind": kind,
                },
                external_id="branch:" + message["id"],
            )
        proposal_mapping = {}
        applied = set(self.state(session_id)["applied_events"])
        for proposal in sorted(
            self.proposals(session_id), key=lambda p: p["decision_order"]
        ):
            if (
                proposal["status"] == "accepted"
                or proposal.get("observed")
                and proposal["status"] != "rejected"
            ) and not proposal["stale"]:
                if not proposal["payload"].get("state_changes"):
                    continue
                if proposal["payload"].get("state_changes") and not any(
                    event.startswith(proposal["id"] + ":") for event in applied
                ):
                    continue
                proposal["evidence"] = [
                    {**e, "message_id": mapping[e["message_id"]]}
                    for e in proposal["evidence"]
                ]
                for change in proposal["payload"].get("state_changes", []):
                    if change["kind"] == "resolve":
                        old, index = change["action_id"].rsplit(":", 1)
                        change["action_id"] = proposal_mapping[old] + ":" + index
                    if change.get("supersedes"):
                        # Only effective state is inherited. The superseded
                        # parent event remains in its original session history.
                        change.pop("supersedes")
                        if proposal["payload"].get("source_event"):
                            proposal["payload"]["source_event"]["supersedes"] = None
                new_id = self.add_proposal(new, proposal, parent_id=proposal["id"])
                proposal_mapping[proposal["id"]] = new_id
                if proposal.get("observed"):
                    self.record_observation(new_id)
                else:
                    self.decide(
                        new, new_id, "accepted", "Inherited from parent " + kind
                    )
        return new

    def state(
        self,
        session_id,
        public_only=False,
        *,
        include_observed=True,
        extra_proposals=(),
    ):
        session = self.session(session_id)
        state = {
            k: {} for k in ("entities", "facts", "resources", "knowledge", "pending")
        }
        state.update(claims=[], resolutions=[], applied_events=[], stale_events=[])
        for char in self.characters(session["campaign_id"], public_only):
            state["entities"][char["name"]] = {
                "sheet": {
                    "value": char["sheet"],
                    "visibility": char["visibility"],
                    "character_id": char["id"],
                }
            }
            for name, value in char["sheet"].get("resources", {}).items():
                state["resources"][char["name"] + ":" + name] = {
                    "value": value,
                    "known_delta": 0,
                    "visibility": char["visibility"],
                }
        messages = self.conversation(session_id)["messages"]
        order = {message["id"]: i for i, message in enumerate(messages, 1)}
        proposals = self.proposals(session_id, public_only)
        for i, proposal in enumerate(extra_proposals):
            proposals.append(
                {
                    **proposal,
                    "id": "preview:" + proposal["payload"]["source_event_sha256"],
                    "status": "pending",
                    "stale": False,
                    "observed": True,
                    "created": "",
                    "decision_order": i,
                    "decision_time": "",
                }
            )

        def sequence(proposal):
            if proposal.get("observed"):
                return (
                    max(
                        (order.get(e["message_id"], 0) for e in proposal["evidence"]),
                        default=0,
                    ),
                    0,
                    proposal["created"],
                    proposal["id"],
                )
            anchor = max(
                (
                    order[m["id"]]
                    for m in messages
                    if m["created"] <= proposal["decision_time"]
                ),
                default=0,
            )
            return anchor, 1, proposal["decision_time"], proposal["id"]

        eligible = [
            p
            for p in proposals
            if (
                include_observed
                and p.get("observed")
                and p["status"] != "rejected"
                or not p.get("observed")
                and p["status"] == "accepted"
            )
        ]
        eligible.sort(key=sequence)
        superseded = set()
        prior = {}
        invalid_replacements = set()
        for proposal in eligible:
            if proposal["stale"]:
                continue
            replacements = []
            for index, change in enumerate(
                proposal["payload"].get("state_changes", [])
            ):
                replacement = change.get("supersedes")
                if replacement:
                    old = prior.get(replacement)
                    if old is None or (old["entity"], old["attribute"]) != (
                        change["entity"],
                        change["attribute"],
                    ):
                        invalid_replacements.add(proposal["id"])
                        continue
                    replacements.append(replacement)
            if proposal["id"] not in invalid_replacements:
                superseded.update(replacements)
                for index, change in enumerate(
                    proposal["payload"].get("state_changes", [])
                ):
                    prior[proposal["id"] + ":" + str(index)] = change
        for proposal in eligible:
            if (
                proposal["stale"]
                or any(
                    change["kind"] == "resolve"
                    and change["action_id"] not in state["pending"]
                    for change in proposal["payload"].get("state_changes", [])
                )
                or proposal["id"] in invalid_replacements
            ):
                state["stale_events"].append(proposal["id"])
                continue
            for index, change in enumerate(
                proposal["payload"].get("state_changes", [])
            ):
                vis = change.get("visibility", proposal["visibility"])
                if public_only and vis != "public":
                    continue
                key = change["entity"] + ":" + change["attribute"]
                identifier = proposal["id"] + ":" + str(index)
                if identifier in superseded:
                    continue
                value = {
                    **change,
                    "source_proposal": proposal["id"],
                    "visibility": vis,
                    "id": identifier,
                    "basis": "conversation" if proposal.get("observed") else "manual",
                }
                kind = change["kind"]
                if kind == "resource":
                    prior = state["resources"].get(
                        key, {"value": None, "known_delta": 0}
                    )
                    delta = change.get("delta")
                    value["value"] = (
                        change.get("value")
                        if delta is None
                        else (
                            prior["value"] + delta
                            if prior["value"] is not None
                            else None
                        )
                    )
                    value["known_delta"] = (
                        0 if delta is None else prior.get("known_delta", 0) + delta
                    )
                    state["resources"][key] = value
                elif kind == "pending":
                    state["pending"][identifier] = value
                elif kind == "resolve":
                    if change["action_id"] not in state["pending"]:
                        state["stale_events"].append(identifier)
                        continue
                    state["pending"].pop(change["action_id"])
                    state["resolutions"].append(
                        {"action": change["action_id"], **value}
                    )
                elif kind == "claim":
                    state["claims"].append(value)
                elif kind == "knowledge":
                    state["knowledge"].setdefault(change["entity"], {})[
                        change["attribute"]
                    ] = value
                else:
                    state["facts"][key] = value
                state["applied_events"].append(identifier)
        return state

    def snapshot(self, session_id, *, public_only=False):
        session = self.session(session_id)
        campaign = self.campaign(session["campaign_id"])
        if public_only:
            campaign = {
                k: v for k, v in campaign.items() if k not in {"direction", "style"}
            }
        conversation = self.conversation(session_id, public_only)
        snapshot = {
            "campaign": campaign,
            "session": session,
            "characters": self.characters(session["campaign_id"], public_only),
            "documents": self.documents(session["campaign_id"], public_only),
            "messages": conversation["messages"],
            "chronology": conversation["chronology"],
            "state": self.state(session_id, public_only),
            "authority_state": self.state(
                session_id, public_only, include_observed=False
            ),
            "proposals": [] if public_only else self.proposals(session_id),
            "publications": [] if public_only else self.publications(session_id),
        }
        if not public_only:
            snapshot["rule_profile"] = self.rule_profile(session["campaign_id"])
            # Choosing or publishing a draft does not advance fictional time.
            # Explicit rejection requests an alternative; its marker stays stable
            # when another draft is accepted, including after a prior rejection.
            with self.store.db() as db:
                snapshot["alternative_request_revision"] = db.execute(
                    "SELECT coalesce(max(rowid),0) FROM play_decisions WHERE session_id=? AND action='rejected' AND undone_by IS NULL",
                    (session_id,),
                ).fetchone()[0]
            snapshot["participants"] = self.participants(session["campaign_id"])
            snapshot["speaker_mappings"] = self.speaker_mappings(session["campaign_id"])
            snapshot["feedback"] = [
                {
                    "proposal_id": p["id"],
                    "kind": p["kind"],
                    "text": p["text"],
                    "decision": p["status"],
                }
                for p in sorted(
                    snapshot["proposals"], key=lambda p: p["decision_order"]
                )
                if p["status"] == "rejected"
                and not p.get("observed")
                and p["kind"] != "state"
            ][-8:]
        return snapshot

    def context_hash(self, snapshot):
        """A conservative novelty gate: ignore only repeats and unambiguous fillers.

        Affirmative replies are substantive when answering a question or pending
        action. No fuzzy text similarity is allowed to erase a changed roll/number.
        """
        messages = []
        previous = None
        question_open = False
        for message in snapshot["messages"]:
            normalized = re.sub(r"\s+", " ", message["text"]).strip().casefold()
            fingerprint = (
                message["speaker"],
                message["role"],
                message["character"],
                normalized,
                message["visibility"],
            )
            filler = normalized.strip(".! ,") in {
                "um",
                "uh",
                "hmm",
                "okay",
                "ok",
                "yeah",
                "yes",
                "right",
            }
            if fingerprint != previous and not (
                filler
                and not question_open
                and not snapshot.get("authority_state", snapshot["state"])["pending"]
            ):
                messages.append(fingerprint)
                question_open = "?" in message["text"]
            previous = fingerprint
        campaign = {
            k: snapshot["campaign"].get(k)
            for k in ("title", "system", "direction", "style")
        }
        return digest(
            packed(
                {
                    "campaign": campaign,
                    "state": snapshot.get("authority_state", snapshot["state"]),
                    "observation_overrides": [
                        (p["id"], p["status"])
                        for p in snapshot.get("proposals", [])
                        if p.get("observed") and p["status"] != "pending"
                    ],
                    "characters": snapshot["characters"],
                    "participants": snapshot.get("participants", []),
                    "alternative_request_revision": snapshot.get(
                        "alternative_request_revision", 0
                    ),
                    "documents": snapshot["documents"],
                    "rule_profile": snapshot.get("rule_profile", {}),
                    "messages": messages,
                }
            )
        )

    def evidence_hash(self, snapshot):
        return digest(
            packed(
                {
                    "context_hash": self.context_hash(snapshot),
                    "sources": [
                        (
                            m["id"],
                            m["revision"],
                            m.get("order_index"),
                            m.get("chronology"),
                        )
                        for m in snapshot["messages"]
                    ],
                }
            )
        )

    def snapshot_guard(self, snapshot):
        sources = []
        for message in snapshot["messages"]:
            timing = message.get("chronology", {})
            sources.append(
                {
                    "id": message["id"],
                    "ordinal": message["ordinal"],
                    "revision": message["revision"],
                    "content_hash": digest(
                        packed(
                            {
                                k: message.get(k)
                                for k in (
                                    "speaker",
                                    "role",
                                    "character",
                                    "text",
                                    "visibility",
                                )
                            }
                        )
                    ),
                    "clock_group": timing.get("clock_group"),
                    "start": timing.get("start"),
                    "end": timing.get("end"),
                }
            )
        return {
            "session_id": snapshot["session"]["id"],
            "authority_hash": self.context_hash({**snapshot, "messages": []}),
            "evidence_hash": self.evidence_hash(snapshot),
            "sources": sources,
            "partial_order": snapshot.get("chronology", {}).get("partial_order", True),
        }

    def snapshot_change(self, previous_guard, current_snapshot):
        return snapshot_change(previous_guard, self.snapshot_guard(current_snapshot))

    def runs(self, session_id):
        with self.store.db() as db:
            return [
                {
                    **dict(r),
                    "request": json.loads(r["request"]),
                    "result": json.loads(r["result"]),
                }
                for r in db.execute(
                    "SELECT * FROM play_runs WHERE session_id=? ORDER BY rowid DESC",
                    (session_id,),
                )
            ]

    def recover_run(self, session_id, run_id):
        """Retain a failed trace after the caller establishes its worker is gone."""
        with self.store.db() as db:
            found = self._one("play_runs", run_id, db)
            if found["session_id"] != session_id or found["status"] != "running":
                raise ValueError("Select a running request from this session.")
            db.execute(
                "UPDATE play_runs SET status='failed',result=?,finished=? WHERE id=?",
                (
                    packed(
                        {
                            "error": "Interrupted request explicitly recovered by the Facilitator. No suggestion was applied."
                        }
                    ),
                    now(),
                    run_id,
                ),
            )

    def start_run(self, session_id, *, force=False, build_context=None):
        snapshot = self.snapshot(session_id)
        fingerprint = self.context_hash(snapshot)
        context = build_context(snapshot) if build_context else snapshot
        context["evidence_hash"] = self.evidence_hash(snapshot)
        context["snapshot_guard"] = self.snapshot_guard(snapshot)
        copilot = context.get("work_protocol") in {
            "facilitator-copilot-v1",
            "facilitator-copilot-v2",
        }
        if copilot:
            context["manual_request"] = bool(force)
            if force:
                context["narration_needed"] = True
                context["classification_only"] = False
        classification_only = (
            copilot
            and context.get("classification_only")
            and bool(
                context.get("target_messages") or context.get("cached_classification")
            )
        )
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute(
                "SELECT id FROM play_runs WHERE session_id=? AND status='running' ORDER BY rowid DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if active:
                return None
            prior = db.execute(
                "SELECT id FROM play_runs WHERE session_id=? AND context_hash=? AND status='complete' LIMIT 1",
                (session_id, fingerprint),
            ).fetchone()
            if prior and not force and not classification_only:
                return None
            identifier = uuid4().hex
            db.execute(
                "INSERT INTO play_runs VALUES(?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    session_id,
                    fingerprint,
                    "running",
                    packed(context),
                    "{}",
                    now(),
                    None,
                ),
            )
        return identifier, context

    def finish_run(self, run_id, result=None, *, error=None):
        result = result or {}
        if not isinstance(result, dict):
            raise ValueError(
                "Model output must contain a suggestions list and optional trace."
            )
        proposals = result.get("suggestions", [])
        if not isinstance(proposals, list) or len(proposals) > 20:
            raise ValueError("Generate at most 20 suggestions per run.")
        # Commit the snapshot check and resulting disposition together; ingestion
        # cannot race between validation and promotion into current proposals.
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            run = self._one("play_runs", run_id, db)
            if run["status"] != "running":
                raise ValueError("This run already finished.")
            current_snapshot = self.snapshot(run["session_id"])
            request = json.loads(run["request"])
            if request.get("snapshot_guard"):
                change = self.snapshot_change(
                    request["snapshot_guard"], current_snapshot
                )
            else:
                stale = (
                    self.evidence_hash(current_snapshot) != request["evidence_hash"]
                    if request.get("evidence_hash")
                    else self.context_hash(current_snapshot) != run["context_hash"]
                )
                change = "invalidated" if stale else "unchanged"
            cancelled = result.get("trace", {}).get("cancelled") is True
            status = (
                "failed"
                if error
                else "cancelled"
                if cancelled
                else "stale"
                if change == "invalidated"
                or change == "append_only"
                and request.get("work_protocol")
                not in {"facilitator-copilot-v1", "facilitator-copilot-v2"}
                else "superseded"
                if change == "append_only"
                else "complete"
            )
            output = {**result, "error": str(error)} if error else result.copy()
            output["snapshot_change"] = change
            output["proposal_ids"] = []
            if status == "complete":
                validated = [
                    self._validate_proposal(run["session_id"], p) for p in proposals
                ]
                output["proposal_ids"] = [
                    self.add_proposal(run["session_id"], p, run_id=run_id, _db=db)
                    for p in validated
                ]
                if request.get("work_protocol") == "facilitator-copilot-v2":
                    for identifier, proposal in zip(output["proposal_ids"], validated):
                        if self.is_observation(proposal):
                            self.record_observation(identifier, _db=db)
                            output.setdefault("observation_ids", []).append(identifier)
            elif status == "superseded":
                # Guidance remains private history. Independently validated
                # observations still update the working state during speech.
                items, rejected = [], []
                for proposal in proposals:
                    if request.get(
                        "work_protocol"
                    ) == "facilitator-copilot-v2" and self.is_observation(proposal):
                        valid = self._validate_proposal(run["session_id"], proposal)
                        identifier = self.add_proposal(
                            run["session_id"], valid, run_id=run_id, _db=db
                        )
                        self.record_observation(identifier, _db=db)
                        output.setdefault("observation_ids", []).append(identifier)
                        continue
                    if proposal.get("payload", {}).get("state_changes") or proposal.get(
                        "payload", {}
                    ).get("source_event"):
                        continue
                    if proposal.get("kind") not in {
                        "narration",
                        "scene",
                        "npc",
                        "question",
                        "action",
                        "note",
                        "rule",
                    }:
                        continue
                    try:
                        valid = self._validate_proposal(run["session_id"], proposal)
                        guidance_id = None
                        if request.get("work_protocol") == "facilitator-copilot-v2":
                            valid["payload"] = {
                                **valid["payload"],
                                "point_in_time": True,
                                "as_of_source": request["snapshot_guard"]["sources"][-1]
                                if request["snapshot_guard"]["sources"]
                                else None,
                            }
                            guidance_id = self.add_proposal(
                                run["session_id"], valid, run_id=run_id, _db=db
                            )
                            output["proposal_ids"].append(guidance_id)
                        items.append(
                            {
                                "proposal_id": guidance_id,
                                "kind": valid["kind"],
                                "title": valid["title"],
                                "text": valid["text"],
                                "visibility": "private",
                            }
                        )
                    except (ValueError, TypeError, KeyError) as exc:
                        rejected.append(str(exc))
                sources = request["snapshot_guard"]["sources"]
                output["historical_draft"] = {
                    "items": items,
                    "visibility": "private",
                    "current": False,
                    "as_of_source": sources[-1] if sources else None,
                    "source_count": len(sources),
                    "newer_source_count": len(current_snapshot["messages"])
                    - len(sources),
                    "state_changes_applied": False,
                    "published": False,
                    "rejected_items": rejected,
                }
            db.execute(
                "UPDATE play_runs SET status=?,result=?,finished=? WHERE id=?",
                (status, packed(output), now(), run_id),
            )
        return status

    def generate(self, session_id, generator, *, force=False, build_context=None):
        try:
            started = self.start_run(
                session_id, force=force, build_context=build_context
            )
        except Exception as exc:
            snapshot = self.snapshot(session_id)
            identifier = uuid4().hex
            with self.store.db() as db:
                db.execute(
                    "INSERT INTO play_runs VALUES(?,?,?,?,?,?,?,?)",
                    (
                        identifier,
                        session_id,
                        self.context_hash(snapshot),
                        "failed",
                        packed({"snapshot_before_context_build": snapshot}),
                        packed({"error": str(exc), "trace": getattr(exc, "trace", {})}),
                        now(),
                        now(),
                    ),
                )
            return identifier
        if started is None:
            return None
        run_id, context = started
        result = None
        try:
            result = generator(context)
            if self._one("play_runs", run_id)["status"] == "running":
                self.finish_run(run_id, result)
        except Exception as exc:
            if self._one("play_runs", run_id)["status"] == "running":
                self.finish_run(
                    run_id,
                    {"trace": getattr(exc, "trace", {}), "raw_result": result},
                    error=exc,
                )
        return run_id
