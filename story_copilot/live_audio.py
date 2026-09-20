"""Private, bounded audio spool and silent replay; no hardware or model imports.

Chunks have disjoint commit intervals with overlapping recognition context. A
consumer acknowledges only after its idempotent external IDs have committed.
Queue expiry retries unfinished work; it never silently drops captured speech.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import select
import shlex
import sys
import subprocess
import time
import wave
from uuid import uuid4

RATE = 16000
CHANNELS = {"mic", "system", "mixed"}


def packed(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class QueueFull(RuntimeError):
    """Stop capture or wait for capacity; no chunks were silently discarded."""


class AudioQueue:
    def __init__(self, home, *, max_pending=32, max_audio_bytes=512 * 1024**2):
        self.home = Path(home).expanduser().resolve()
        if any((p / ".git").exists() for p in [self.home, *self.home.parents]):
            raise ValueError("Keep the audio queue outside source repositories.")
        if max_pending < 1 or max_audio_bytes < RATE * 2:
            raise ValueError("Audio queue limits must be positive.")
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.home / "audio.sqlite"
        self.max_pending = max_pending
        self.max_audio_bytes = max_audio_bytes
        with self.db() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, session TEXT NOT NULL, channel TEXT NOT NULL,
                sequence INTEGER NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
                commit_start REAL NOT NULL, commit_end REAL NOT NULL,
                audio_sha256 TEXT NOT NULL, pcm BLOB, source TEXT NOT NULL,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                lease_token TEXT, lease_until REAL, error TEXT,
                created REAL NOT NULL, finished REAL, result TEXT,
                UNIQUE(session,channel,sequence));
            CREATE TABLE IF NOT EXISTS receipts (
                consumer TEXT NOT NULL, chunk_id TEXT NOT NULL REFERENCES chunks(id),
                created REAL NOT NULL, PRIMARY KEY(consumer,chunk_id));
            CREATE TABLE IF NOT EXISTS capture_clocks (
                session TEXT NOT NULL, boot_id TEXT NOT NULL, domain TEXT NOT NULL,
                origin_utc REAL NOT NULL,origin_monotonic REAL NOT NULL,
                PRIMARY KEY(session,boot_id));
            CREATE TABLE IF NOT EXISTS voices (
                session TEXT NOT NULL, channel TEXT NOT NULL, backend TEXT NOT NULL,
                speaker TEXT NOT NULL, embedding TEXT NOT NULL, observations INTEGER NOT NULL,
                PRIMARY KEY(session,channel,backend,speaker));
            CREATE INDEX IF NOT EXISTS chunk_work ON chunks(status,created);
            """)
        os.chmod(self.path, 0o600)

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

    def capture_clock(self, session_id, *, boot_id, utc_now=None, monotonic_now=None):
        """One shared origin per session/host boot, retained across capture restarts."""
        utc_now = time.time() if utc_now is None else utc_now
        monotonic_now = time.monotonic() if monotonic_now is None else monotonic_now
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT OR IGNORE INTO capture_clocks VALUES(?,?,?,?,?)",
                (session_id, boot_id, "capture:" + uuid4().hex, utc_now, monotonic_now),
            )
            row = dict(
                db.execute(
                    "SELECT * FROM capture_clocks WHERE session=? AND boot_id=?",
                    (session_id, boot_id),
                ).fetchone()
            )
        error = (utc_now - row["origin_utc"]) - (
            monotonic_now - row["origin_monotonic"]
        )
        return {
            "domain": row["domain"],
            "kind": "live_capture",
            "origin_utc": row["origin_utc"],
            "origin_monotonic": row["origin_monotonic"],
            "boot_id": boot_id,
            "utc_anchor_valid": abs(error) <= 0.5,
            "wall_clock_offset_change_seconds": round(error, 6),
            "utc_quality": "host wall-clock anchor; cross-host synchronization is not calibrated",
        }

    def enqueue_pcm(
        self,
        session_id,
        channel,
        sequence,
        start_seconds,
        pcm,
        *,
        commit_start=None,
        commit_end=None,
        source=None,
    ):
        """Enqueue 16 kHz mono signed little-endian PCM16; return stable chunk ID."""
        if not session_id or len(session_id) > 200 or channel not in CHANNELS:
            raise ValueError("A session ID and mic/system/mixed channel are required.")
        if not isinstance(sequence, int) or sequence < 0:
            raise ValueError("Sequence must be a nonnegative integer.")
        if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
            raise ValueError("Audio must be nonempty PCM16 bytes.")
        end = start_seconds + len(pcm) / (RATE * 2)
        commit_start = start_seconds if commit_start is None else commit_start
        commit_end = end if commit_end is None else commit_end
        if not all(
            math.isfinite(v) for v in (start_seconds, end, commit_start, commit_end)
        ):
            raise ValueError("Timing must be finite.")
        if not 0 <= start_seconds <= commit_start < commit_end <= end + 1e-7:
            raise ValueError("Commit interval must be inside the audio interval.")
        if end - start_seconds > 120:
            raise ValueError("Audio chunks are limited to 120 seconds.")
        metadata = packed(source or {})
        audio_hash = sha(pcm)
        cid = sha(packed([session_id, channel, sequence]).encode())[:32]
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM chunks WHERE id=?", (cid,)).fetchone()
            if existing:
                identity = (
                    audio_hash,
                    start_seconds,
                    commit_start,
                    commit_end,
                    metadata,
                )
                prior = tuple(
                    existing[k]
                    for k in (
                        "audio_sha256",
                        "start",
                        "commit_start",
                        "commit_end",
                        "source",
                    )
                )
                if identity != prior:
                    raise ValueError(
                        "Chunk identity was already used for different audio or timing."
                    )
                return cid
            count = db.execute(
                "SELECT count(*) FROM chunks WHERE status NOT IN ('complete','forwarded')"
            ).fetchone()[0]
            retained = db.execute(
                "SELECT coalesce(sum(length(pcm)),0) FROM chunks"
            ).fetchone()[0]
            if count >= self.max_pending or retained + len(pcm) > self.max_audio_bytes:
                raise QueueFull(
                    "Audio spool is full. Process queued audio or explicitly prune acknowledged recordings."
                )
            previous = db.execute(
                "SELECT * FROM chunks WHERE session=? AND channel=? ORDER BY sequence DESC LIMIT 1",
                (session_id, channel),
            ).fetchone()
            previous_source = json.loads(previous["source"]) if previous else {}
            old_clock = previous_source.get("clock")
            new_clock = (source or {}).get("clock")
            old_domain = (
                old_clock.get("domain") if isinstance(old_clock, dict) else None
            )
            new_domain = (
                new_clock.get("domain") if isinstance(new_clock, dict) else None
            )
            comparable = old_domain == new_domain
            if previous and (
                sequence <= previous["sequence"]
                or (comparable and commit_start < previous["commit_end"] - 1e-7)
            ):
                raise ValueError(
                    "Sequences must increase and committed audio intervals cannot overlap."
                )
            db.execute(
                "INSERT INTO chunks(id,session,channel,sequence,start,end,commit_start,commit_end,audio_sha256,pcm,source,status,created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cid,
                    session_id,
                    channel,
                    sequence,
                    start_seconds,
                    end,
                    commit_start,
                    commit_end,
                    audio_hash,
                    pcm,
                    metadata,
                    "pending",
                    time.time(),
                ),
            )
        return cid

    def claim(self, *, lease_seconds=300):
        if lease_seconds <= 0:
            raise ValueError("Lease must be positive.")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            stamp = time.time()
            db.execute(
                "UPDATE chunks SET status='pending',lease_token=NULL,lease_until=NULL WHERE status='processing' AND lease_until<?",
                (stamp,),
            )
            row = db.execute("""SELECT c.* FROM chunks c WHERE c.status='pending'
                AND NOT EXISTS(SELECT 1 FROM chunks older WHERE older.session=c.session
                  AND older.channel=c.channel AND older.sequence<c.sequence AND older.status NOT IN ('complete','forwarded'))
                ORDER BY c.created,c.channel LIMIT 1""").fetchone()
            if row is None:
                return None
            token = uuid4().hex
            db.execute(
                "UPDATE chunks SET status='processing',attempts=attempts+1,lease_token=?,lease_until=?,error=NULL WHERE id=?",
                (token, stamp + lease_seconds, row["id"]),
            )
            result = dict(row)
            result.update(
                lease_token=token, status="processing", attempts=row["attempts"] + 1
            )
            result["source"] = json.loads(result["source"])
            return result

    def heartbeat(self, chunk, *, lease_seconds=300):
        with self.db() as db:
            changed = db.execute(
                "UPDATE chunks SET lease_until=? WHERE id=? AND status='processing' AND lease_token=?",
                (time.time() + lease_seconds, chunk["id"], chunk["lease_token"]),
            ).rowcount
            if not changed:
                raise RuntimeError(
                    "Audio processing lease has expired or been replaced."
                )

    def fail(self, chunk, error):
        with self.db() as db:
            changed = db.execute(
                "UPDATE chunks SET status='failed',error=?,lease_token=NULL,lease_until=NULL WHERE id=? AND lease_token=?",
                (str(error)[:1000], chunk["id"], chunk["lease_token"]),
            ).rowcount
            if not changed:
                raise RuntimeError("Cannot fail an obsolete audio lease.")

    def retry(self, chunk_id):
        with self.db() as db:
            if not db.execute(
                "UPDATE chunks SET status='pending',error=NULL WHERE id=? AND status='failed'",
                (chunk_id,),
            ).rowcount:
                raise ValueError("Only a failed chunk can be retried.")

    def complete(self, chunk, document, *, voices=(), backend=""):
        payload = packed(document)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                "UPDATE chunks SET status='complete',result=?,finished=?,lease_token=NULL,lease_until=NULL WHERE id=? AND status='processing' AND lease_token=?",
                (payload, time.time(), chunk["id"], chunk["lease_token"]),
            ).rowcount
            if not changed:
                raise RuntimeError("Cannot finish an obsolete audio lease.")
            for voice in voices:
                db.execute(
                    "INSERT OR REPLACE INTO voices VALUES(?,?,?,?,?,?)",
                    (
                        chunk["session"],
                        chunk["channel"],
                        backend,
                        voice["speaker"],
                        packed(voice["embedding"]),
                        voice["observations"],
                    ),
                )

    def voices(self, session, channel, backend):
        with self.db() as db:
            return [
                {**dict(r), "embedding": json.loads(r["embedding"])}
                for r in db.execute(
                    "SELECT * FROM voices WHERE session=? AND channel=? AND backend=? ORDER BY speaker",
                    (session, channel, backend),
                )
            ]

    def ready(self, consumer, *, session=None, limit=100):
        with self.db() as db:
            rows = db.execute(
                """SELECT c.id,c.session,c.channel,c.sequence,c.start,c.end,c.commit_start,c.commit_end,c.audio_sha256,c.source,c.result
                FROM chunks c WHERE c.status='complete' AND (? IS NULL OR c.session=?)
                AND NOT EXISTS(SELECT 1 FROM receipts r WHERE r.consumer=? AND r.chunk_id=c.id)
                ORDER BY c.commit_start,c.channel,c.sequence LIMIT ?""",
                (session, session, consumer, limit),
            )
            return [
                {
                    **dict(r),
                    "source": json.loads(r["source"]),
                    "document": json.loads(r["result"]),
                }
                for r in rows
            ]

    def acknowledge(self, consumer, chunk_id):
        if not consumer:
            raise ValueError("Consumer ID is required.")
        with self.db() as db:
            if not db.execute(
                "SELECT 1 FROM chunks WHERE id=? AND status='complete'", (chunk_id,)
            ).fetchone():
                raise ValueError("Only a completed chunk can be acknowledged.")
            db.execute(
                "INSERT OR IGNORE INTO receipts VALUES(?,?,?)",
                (consumer, chunk_id, time.time()),
            )

    def prune_audio(self, consumer, *, before):
        """Explicitly discard old acknowledged PCM; retain result, hash and receipt."""
        with self.db() as db:
            return db.execute(
                "UPDATE chunks SET pcm=NULL WHERE status IN ('complete','forwarded') AND finished<? AND EXISTS(SELECT 1 FROM receipts r WHERE r.chunk_id=chunks.id AND r.consumer=?)",
                (before, consumer),
            ).rowcount

    def audio_clip(self, chunk_id, session_id):
        """Return one verified bounded WAV from this queue, never an arbitrary path.

        Callers must authorize the campaign message first. The audio session is
        the original capture/replay session, which can differ from campaign ID.
        """
        with self.db() as db:
            row = db.execute(
                "SELECT id,session,channel,start,end,commit_start,commit_end,audio_sha256,pcm,status FROM chunks WHERE id=? AND session=?",
                (chunk_id, session_id),
            ).fetchone()
        if row is None or row["status"] != "complete":
            raise ValueError("Completed audio chunk not found for this session.")
        pcm = row["pcm"]
        if pcm is None:
            raise FileNotFoundError(
                "This recording was pruned; its transcript and provenance remain available."
            )
        if (
            not pcm
            or len(pcm) % 2
            or len(pcm) > RATE * 2 * 120
            or sha(pcm) != row["audio_sha256"]
        ):
            raise ValueError("Stored audio is invalid or differs from its source hash.")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes(pcm)
        return {
            **{k: row[k] for k in row.keys() if k != "pcm"},
            "sample_rate": RATE,
            "wav_bytes": output.getvalue(),
            "media_type": "audio/wav",
        }

    def status(self, *, session=None):
        with self.db() as db:
            rows = db.execute(
                "SELECT status,count(*) n,coalesce(sum(length(pcm)),0) bytes FROM chunks WHERE (? IS NULL OR session=?) GROUP BY status",
                (session, session),
            ).fetchall()
            errors = db.execute(
                "SELECT id,channel,sequence,error FROM chunks WHERE status='failed' AND (? IS NULL OR session=?) ORDER BY created",
                (session, session),
            ).fetchall()
            return {
                "counts": {r["status"]: r["n"] for r in rows},
                "retained_audio_bytes": sum(r["bytes"] for r in rows),
                "errors": [dict(r) for r in errors],
            }


class PCMChunker:
    """Incremental overlap windows whose committed central regions never overlap."""

    def __init__(self, *, chunk_seconds=30, overlap_seconds=2, start_seconds=0):
        if not 1 <= chunk_seconds <= 90 or not 0 <= overlap_seconds <= min(
            10, chunk_seconds
        ):
            raise ValueError(
                "Use 1–90 second chunks and at most 10 seconds of context."
            )
        if not math.isfinite(start_seconds) or start_seconds < 0:
            raise ValueError("Invalid stream origin.")
        self.core = round(chunk_seconds * RATE)
        self.overlap = round(overlap_seconds * RATE)
        self.origin = start_seconds
        self.buffer = b""
        self.base = self.cursor = self.sequence = 0
        self.closed = False

    def feed(self, pcm, *, final=False):
        if self.closed:
            raise RuntimeError("This audio stream was already finalized.")
        if not isinstance(pcm, bytes) or len(pcm) % 2:
            raise ValueError("Expected PCM16 bytes.")
        self.buffer += pcm
        outputs = []
        available = self.base + len(self.buffer) // 2
        while available >= self.cursor + self.core + self.overlap or (
            final and available > self.cursor
        ):
            finish = min(self.cursor + self.core, available)
            window_start = max(0, self.cursor - self.overlap)
            window_end = min(finish + self.overlap, available)
            a, b = (window_start - self.base) * 2, (window_end - self.base) * 2
            outputs.append(
                dict(
                    sequence=self.sequence,
                    start_seconds=self.origin + window_start / RATE,
                    commit_start=self.origin + self.cursor / RATE,
                    commit_end=self.origin + finish / RATE,
                    pcm=self.buffer[a:b],
                )
            )
            self.cursor, self.sequence = finish, self.sequence + 1
            retained_start = max(0, self.cursor - self.overlap)
            self.buffer = self.buffer[(retained_start - self.base) * 2 :]
            self.base = retained_start
        self.closed = final
        return outputs


class TimedPCMChunker:
    """Window timestamped packets without erasing pauses or channel start offsets."""

    def __init__(
        self, *, chunk_seconds=30, overlap_seconds=2, sequence=0, tolerance_seconds=0.02
    ):
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.sequence = sequence
        self.tolerance = tolerance_seconds
        self.chunker = None
        self.expected = None
        self.closed = False

    def _number(self, parts):
        for part in parts:
            part["sequence"] = self.sequence
            self.sequence += 1
        return parts

    def feed(self, pcm, *, start_seconds=None, final=False):
        if self.closed:
            raise RuntimeError("Audio stream is already finalized.")
        parts = []
        if pcm:
            if (
                start_seconds is None
                or not math.isfinite(start_seconds)
                or start_seconds < 0
            ):
                raise ValueError("A finite common-clock packet timestamp is required.")
            if (
                self.expected is not None
                and start_seconds < self.expected - self.tolerance
            ):
                raise ValueError(
                    "Capture clock moved backwards; do not invent chronological order."
                )
            if (
                self.chunker is not None
                and start_seconds > self.expected + self.tolerance
            ):
                parts += self._number(self.chunker.feed(b"", final=True))
                self.chunker = None
            if self.chunker is None:
                self.chunker = PCMChunker(
                    chunk_seconds=self.chunk_seconds,
                    overlap_seconds=self.overlap_seconds,
                    start_seconds=start_seconds,
                )
                self.expected = start_seconds
            parts += self._number(self.chunker.feed(pcm))
            self.expected += len(pcm) / (RATE * 2)
        if final:
            if self.chunker is not None:
                parts += self._number(self.chunker.feed(b"", final=True))
            self.closed = True
        return parts

    def flush_idle(self, current_seconds, *, idle_seconds=2.0):
        """Publish a stopped source without inventing silent samples to fill a window."""
        if (
            self.closed
            or self.chunker is None
            or current_seconds - self.expected < idle_seconds
        ):
            return []
        parts = self._number(self.chunker.feed(b"", final=True))
        self.chunker = None
        self.expected = None
        return parts


def replay_file(
    queue,
    audio_path,
    *,
    session_id,
    channel="mixed",
    chunk_seconds=30,
    overlap_seconds=2,
    start_seconds=0,
    duration=None,
    wait_for_capacity=False,
):
    """Decode with ffmpeg to the private queue; never routes audio to a device."""
    path = Path(audio_path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    source = {
        "kind": "file_replay",
        "path": str(path),
        "sha256": digest.hexdigest(),
        "sample_rate": RATE,
        "source_offset_seconds": start_seconds,
    }
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        str(start_seconds),
        "-i",
        str(path),
    ]
    if duration is not None:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Duration must be positive.")
        command.extend(["-t", str(duration)])
    command += ["-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"]
    chunker = PCMChunker(chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    count = 0
    try:
        while True:
            block = process.stdout.read(RATE * 2)
            for part in chunker.feed(block, final=not block):
                while True:
                    try:
                        queue.enqueue_pcm(session_id, channel, source=source, **part)
                        count += 1
                        break
                    except QueueFull:
                        if not wait_for_capacity:
                            raise
                        time.sleep(0.25)
            if not block:
                break
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed to decode the source audio.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()
    return count


def timed_turns(document, *, pause_seconds=0.8):
    """Preserve source sentences and split same-speaker speech across real pauses."""
    from .corpus import parse_whisperx, word_group

    for turn in parse_whisperx(document):
        words = turn["metadata"]["words"]
        if not words:
            yield turn
            continue
        group = []
        for word in words:
            if (
                group
                and group[-1].get("end") is not None
                and word.get("start") is not None
                and word["start"] - group[-1]["end"] > pause_seconds
            ):
                yield word_group(group, turn["speaker"], turn["metadata"]["segment"])
                group = []
            group.append(word)
        if group:
            yield word_group(group, turn["speaker"], turn["metadata"]["segment"])


def audio_timeline(source, start, end, *, session, channel):
    """Describe compatible clocks explicitly; replay seconds are never UTC."""
    clock = source.get("clock")
    result = {
        "domain": None,
        "kind": "legacy_uncertain",
        "start_seconds": start,
        "end_seconds": end,
        "utc_start": None,
        "utc_end": None,
        "quality": "unknown clock relation",
        "uncertainty": "No common clock anchor; cross-channel/manual chronology is unknown.",
    }
    if source.get("kind") == "file_replay" and source.get("sha256"):
        offset = source.get("source_offset_seconds", 0)
        result.update(
            domain="recording:" + source["sha256"],
            kind="recording",
            start_seconds=start + offset if start is not None else None,
            end_seconds=end + offset if end is not None else None,
            quality="recording-relative aligned speech",
            uncertainty="No calendar/playback UTC anchor; ordering against manual messages is unknown.",
        )
    elif isinstance(clock, dict) and clock.get("kind") == "live_capture":
        result.update(
            domain=clock.get("domain"),
            kind="live_capture",
            quality=clock.get("timestamp_quality", "capture callback timing"),
            uncertainty=clock.get("utc_quality", "Host UTC alignment is approximate."),
        )
        if clock.get("utc_anchor_valid") and clock.get("origin_utc") is not None:
            result["utc_start"] = (
                clock["origin_utc"] + start if start is not None else None
            )
            result["utc_end"] = clock["origin_utc"] + end if end is not None else None
        else:
            result["uncertainty"] = (
                "Host UTC anchor changed or is absent; use only this clock domain."
            )
    else:
        result["domain"] = (
            f"legacy:{session}:{channel}:{source.get('capture_id', 'unknown')}"
        )
    if start is None or end is None:
        result["quality"] += "; word/utterance timing incomplete"
        result["uncertainty"] += " Missing speech timestamps are not reconstructed."
    return result


def deliver_to_campaign(
    queue,
    campaigns,
    audio_session_id,
    campaign_session_id,
    *,
    consumer_id=None,
    visibility="private",
):
    """Crash-safe via campaign external IDs, not an unsafe exactly-once promise."""
    if visibility not in {"private", "public"}:
        raise ValueError("Audio visibility must be facilitator or public.")
    consumer = consumer_id or f"campaign:{campaign_session_id}"
    delivered = 0
    for item in queue.ready(consumer, session=audio_session_id):
        for ordinal, turn in enumerate(timed_turns(item["document"])):
            if not turn["text"].strip():
                continue
            source = {
                "kind": "live_audio",
                "chunk_id": item["id"],
                "channel": item["channel"],
                "audio_sha256": item["audio_sha256"],
                "start": turn["start"],
                "end": turn["end"],
                "recording": item["source"],
                "timeline": audio_timeline(
                    item["source"],
                    turn["start"],
                    turn["end"],
                    session=audio_session_id,
                    channel=item["channel"],
                ),
                "transcription": item["document"].get("provenance", {}),
                "words": turn["metadata"]["words"],
                "review_status": "unreviewed",
            }
            campaigns.add_message(
                campaign_session_id,
                turn["speaker"],
                turn["text"],
                role="unknown",
                visibility=visibility,
                source=source,
                external_id=f"audio:{item['id']}:{ordinal}",
            )
        queue.acknowledge(consumer, item["id"])
        delivered += 1
    return delivered


def recover_captures(queue):
    """Requeue retained full/partial captures after backlog or transport failure."""
    pending = []
    for path in queue.home.glob("capture-recovery-*.json"):
        entry = json.loads(path.read_text())
        pcm_path = Path(entry.pop("pcm_file")).resolve()
        if pcm_path != path.with_suffix(".pcm").resolve():
            raise ValueError("Recovery audio must be adjacent to its manifest.")
        data = pcm_path.read_bytes()
        expected = entry.pop("pcm_sha256")
        if sha(data) != expected:
            raise ValueError("Recovery audio differs from its recorded hash.")
        pending.append((entry, data))
    pending.sort(
        key=lambda pair: (
            pair[0]["session_id"],
            pair[0]["channel"],
            pair[0]["sequence"],
        )
    )
    return [queue.enqueue_pcm(pcm=data, **entry) for entry, data in pending]


class SSHQueueTransport:
    """One authenticated SSH process, chunk acknowledgments and resumable delivery.

    SSH's existing agent/config/control socket supplies credentials. No passwords or
    API keys enter this protocol. The receiver writes only to its private audio queue.
    """

    def __init__(
        self,
        target,
        remote_queue,
        *,
        remote_python="python",
        control_path=None,
        timeout=30,
    ):
        if not target or target.startswith("-"):
            raise ValueError("Invalid SSH destination.")
        command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        if control_path:
            command.extend(["-S", str(Path(control_path).expanduser())])
        command.extend(
            [
                target,
                shlex.join(
                    [
                        remote_python,
                        "-m",
                        "story_copilot.live_audio",
                        "--queue",
                        remote_queue,
                        "receive",
                    ]
                ),
            ]
        )
        self.command, self.timeout = command, timeout
        self.consumer = "ssh:" + sha(packed([target, remote_queue]).encode())[:24]
        self.process = None
        self.cancel = None

    def start(self):
        if self.process is not None and self.process.poll() is not None:
            self.close()
        if self.process is None:
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
            )
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)

    def _exchange(self, payload):
        deadline = time.monotonic() + self.timeout
        sent = 0
        reply = bytearray()
        try:
            while True:
                if self.cancel is not None and self.cancel.is_set():
                    raise InterruptedError(
                        "Audio transfer cancelled; local chunks remain available."
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Audio SSH transfer timed out; local chunks remain available."
                    )
                if sent < len(payload):
                    _, ready, _ = select.select(
                        [], [self.process.stdin], [], min(remaining, 0.1)
                    )
                    if ready:
                        try:
                            sent += os.write(
                                self.process.stdin.fileno(),
                                payload[sent : sent + 65536],
                            )
                        except BlockingIOError:
                            pass
                else:
                    ready, _, _ = select.select(
                        [self.process.stdout], [], [], min(remaining, 0.1)
                    )
                    if ready:
                        block = os.read(self.process.stdout.fileno(), 4096 - len(reply))
                        if not block:
                            raise RuntimeError(
                                "Audio SSH receiver exited; local chunks remain available."
                            )
                        reply.extend(block)
                        if b"\n" in reply:
                            return json.loads(bytes(reply))
                        if len(reply) >= 4096:
                            raise ValueError("Audio SSH reply exceeds protocol limit.")
        except BaseException:
            self.close()
            raise

    def transfer(self, queue):
        self.start()
        with queue.db() as db:
            rows = db.execute(
                "SELECT * FROM chunks WHERE status='pending' ORDER BY created"
            ).fetchall()
        sent = 0
        for row in rows:
            if row["pcm"] is None:
                raise RuntimeError("Audio for a pending transfer is missing.")
            request = {
                "session_id": row["session"],
                "channel": row["channel"],
                "sequence": row["sequence"],
                "start_seconds": row["start"],
                "commit_start": row["commit_start"],
                "commit_end": row["commit_end"],
                "source": json.loads(row["source"]),
                "pcm_base64": base64.b64encode(row["pcm"]).decode("ascii"),
            }
            reply = self._exchange(packed(request).encode() + b"\n")
            if reply.get("error"):
                raise RuntimeError("Audio SSH receiver: " + reply["error"])
            if reply.get("chunk_id") != row["id"]:
                raise RuntimeError("Audio SSH receiver returned a mismatched chunk ID.")
            with queue.db() as db:
                db.execute(
                    "UPDATE chunks SET status='forwarded',finished=? WHERE id=? AND status='pending'",
                    (time.time(), row["id"]),
                )
                db.execute(
                    "INSERT OR IGNORE INTO receipts VALUES(?,?,?)",
                    (self.consumer, row["id"], time.time()),
                )
            sent += 1
        return sent

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            process.stdout.close()


def receive_stdio(queue):
    """Private SSH stdin protocol, never an unauthenticated network listener."""
    # A maximum 120-second PCM16 chunk plus metadata, base64 expansion and newline.
    limit = RATE * 2 * 120 * 4 // 3 + 65536
    while line := sys.stdin.buffer.readline(limit + 1):
        if len(line) > limit or not line.endswith(b"\n"):
            raise ValueError("Oversized or incomplete audio transfer frame.")
        try:
            request = json.loads(line)
            pcm = base64.b64decode(request.pop("pcm_base64"), validate=True)
            cid = queue.enqueue_pcm(pcm=pcm, **request)
            reply = {"chunk_id": cid}
        except Exception as exc:
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        print(packed(reply), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("audio")
    replay.add_argument("--session", required=True)
    replay.add_argument("--channel", choices=sorted(CHANNELS), default="mixed")
    replay.add_argument("--chunk-seconds", type=float, default=30)
    replay.add_argument("--overlap-seconds", type=float, default=2)
    replay.add_argument("--start-seconds", type=float, default=0)
    replay.add_argument("--duration", type=float)
    replay.add_argument(
        "--wait",
        action="store_true",
        help="Wait for a separate worker when backlog is full.",
    )
    sub.add_parser("status")
    sub.add_parser("receive")
    sub.add_parser("recover")
    ship = sub.add_parser("ship")
    ship.add_argument("--ssh", required=True)
    ship.add_argument("--remote-queue", required=True)
    ship.add_argument("--remote-python", default="python")
    ship.add_argument("--ssh-control")
    retry = sub.add_parser("retry")
    retry.add_argument("chunk_id")
    args = parser.parse_args()
    queue = AudioQueue(args.queue)
    if args.command == "replay":
        print(
            packed(
                {
                    "enqueued": replay_file(
                        queue,
                        args.audio,
                        session_id=args.session,
                        channel=args.channel,
                        chunk_seconds=args.chunk_seconds,
                        overlap_seconds=args.overlap_seconds,
                        start_seconds=args.start_seconds,
                        duration=args.duration,
                        wait_for_capacity=args.wait,
                    )
                }
            )
        )
    elif args.command == "recover":
        print(packed({"recovered": recover_captures(queue)}))
    elif args.command == "receive":
        receive_stdio(queue)
    elif args.command == "ship":
        transport = SSHQueueTransport(
            args.ssh,
            args.remote_queue,
            remote_python=args.remote_python,
            control_path=args.ssh_control,
        )
        try:
            print(packed({"forwarded": transport.transfer(queue)}))
        finally:
            transport.close()
    elif args.command == "retry":
        queue.retry(args.chunk_id)
    else:
        print(packed(queue.status()))


if __name__ == "__main__":
    main()
