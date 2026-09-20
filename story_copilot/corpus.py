from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

from .store import Store, digest, packed, now


def parse_text(text):
    """Preserve every nonblank line, including unknown/unrecognized speaker labels."""
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = re.match(
            r"^(?:Speaker[ _])?(SPEAKER_\d+|Unknown|UNKNOWN)\s*:\s*(.*)$", line
        )
        yield dict(
            speaker=match[1] if match else "Unknown",
            text=match[2] if match else line,
            start=None,
            end=None,
            metadata={"source_line": line_no, "timing": "unavailable"},
        )


def parse_whisperx(document):
    """Keep word timing and split mixed segments at observed word-speaker changes."""
    for i, segment in enumerate(document["segments"]):
        words = segment.get("words") or []
        if not words:
            yield dict(
                speaker=segment.get("speaker", "Unknown"),
                text=segment["text"],
                start=segment.get("start"),
                end=segment.get("end"),
                metadata={"segment": i, "words": [], "timing": "segment"},
            )
            continue
        group = []
        speaker = None
        for word in words:
            current = word.get("speaker", segment.get("speaker", "Unknown"))
            if group and speaker != current:
                yield word_group(group, speaker, i)
                group = []
            speaker = current
            group.append(word)
        if group:
            yield word_group(group, speaker, i)


def word_group(words, speaker, segment):
    return dict(
        speaker=speaker or "Unknown",
        text=" ".join(w.get("word", "") for w in words).strip(),
        start=next((w["start"] for w in words if "start" in w), None),
        end=next((w["end"] for w in reversed(words) if "end" in w), None),
        metadata={"segment": segment, "words": words, "timing": "word"},
    )


def import_transcript(
    store: Store, path, *, story, title=None, split="development", audio=None
):
    path = Path(path).expanduser().resolve(strict=True)
    raw = path.read_bytes()
    source_id = digest(raw)
    text = raw.decode("utf-8-sig")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        doc = None
    if doc is not None:
        if not isinstance(doc, dict) or "segments" not in doc:
            raise ValueError(
                "Import the original speaker transcript or WhisperX JSON; chat exports have lost speaker provenance."
            )
        turns = list(parse_whisperx(doc))
    else:
        turns = list(parse_text(text))
    if not turns:
        raise ValueError("The source contains no turns.")
    if split not in {"train", "validation", "test", "development"}:
        raise ValueError("Invalid story split.")
    audio_path = str(Path(audio).expanduser().resolve(strict=True)) if audio else None
    cid = digest(story + ":" + source_id)[:24]
    with store.db() as db:
        existing_story = db.execute(
            "SELECT story_id FROM collections WHERE source_id=? LIMIT 1", (source_id,)
        ).fetchone()
        if existing_story and existing_story["story_id"] != story:
            raise ValueError(
                "This source already belongs to a story; preserve its identity and split."
            )
        prior = db.execute("SELECT split FROM stories WHERE id=?", (story,)).fetchone()
        if prior and prior["split"] != split:
            raise ValueError("All variants of a story must share its existing split.")
        db.execute(
            "INSERT OR IGNORE INTO stories VALUES (?,?,?)", (story, story, split)
        )
        db.execute(
            "INSERT OR IGNORE INTO sources VALUES (?,?,?,?,?)",
            (source_id, str(path), source_id, text, now()),
        )
        exists = db.execute("SELECT id FROM collections WHERE id=?", (cid,)).fetchone()
        if exists:
            return cid
        metadata = {
            "format": "whisperx" if doc else "legacy-text",
            "source_sha256": source_id,
        }
        db.execute(
            "INSERT INTO collections VALUES (?,?,?,?,?,?,?)",
            (
                cid,
                story,
                source_id,
                title or path.stem,
                audio_path,
                packed(metadata),
                now(),
            ),
        )
        for i, turn in enumerate(turns, 1):
            tid = f"{cid}:{i}"
            db.execute(
                "INSERT INTO turns VALUES (?,?,?,?,?,?,?,?)",
                (
                    tid,
                    cid,
                    i,
                    turn["speaker"],
                    turn["text"],
                    turn["start"],
                    turn["end"],
                    packed(turn["metadata"]),
                ),
            )
            db.execute(
                "INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    uuid4().hex,
                    tid,
                    turn["text"],
                    "unknown",
                    "",
                    "pending",
                    "gameplay",
                    "import",
                    "",
                    now(),
                ),
            )
    return cid
