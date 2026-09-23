"""Private replay of retained natural audio, with prior ASR kept as an unreviewed comparison.

This tool never records or plays audio. It does not call a previous model transcript
acoustic gold, infer speaker identity, or change the source queue.
"""

import argparse
from hashlib import sha256
import json
import sqlite3
from pathlib import Path
import time
import wave
from html import escape

from .live_audio import AudioQueue, RATE


def private_output(path):
    path = Path(path).expanduser().resolve()
    if path.exists() or any((p / ".git").exists() for p in (path, *path.parents)):
        raise ValueError("Choose a new private directory outside repositories.")
    path.mkdir(parents=True, mode=0o700)
    return path


def prepare(source_queue, session, output):
    source_queue = Path(source_queue).expanduser().resolve()
    if not (source_queue / "audio.sqlite").is_file():
        raise ValueError("The retained source queue does not exist.")
    db = sqlite3.connect(
        (source_queue / "audio.sqlite").as_uri() + "?mode=ro", uri=True
    )
    db.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in db.execute(
                "SELECT * FROM chunks WHERE session=? AND status='complete' ORDER BY sequence,channel",
                (session,),
            )
        ]
    finally:
        db.close()
    if not rows or any(row["pcm"] is None for row in rows):
        raise ValueError(
            "The selected completed session needs retained audio for every chunk."
        )
    output = private_output(output)
    manifest = {
        "kind": "retained_natural_audio",
        "capture": False,
        "playback": False,
        "baseline_is_gold": False,
        "speaker_identity": "unverified",
        "chunks": [],
    }
    for index, row in enumerate(rows):
        pcm = row.pop("pcm")
        if sha256(pcm).hexdigest() != row["audio_sha256"]:
            raise ValueError("Retained audio differs from the source hash.")
        path = output / f"chunk-{index:03d}.wav"
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(RATE)
            stream.writeframes(pcm)
        manifest["chunks"].append(
            {
                "file": path.name,
                "original_chunk_id": row["id"],
                "session": row["session"],
                "channel": row["channel"],
                "sequence": row["sequence"],
                "start": row["start"],
                "commit_start": row["commit_start"],
                "commit_end": row["commit_end"],
                "pcm_sha256": row["audio_sha256"],
                "source": json.loads(row["source"]),
                "baseline_asr_unreviewed": json.loads(row["result"]),
            }
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def transcribe(fixtures, output, *, device_index=1):
    from .audio_worker import PersistentAudioWorker, WhisperXDiarizer

    fixtures = Path(fixtures).expanduser().resolve()
    raw = (fixtures / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("kind") != "retained_natural_audio":
        raise ValueError("Expected a retained-recording manifest.")
    output = private_output(output)
    queue = AudioQueue(output / "queue")
    # No one-speaker prior: this is natural mixed speech.
    worker = PersistentAudioWorker(queue, WhisperXDiarizer(device_index=device_index))
    report = {
        "kind": "natural_audio_replay",
        "manifest_sha256": sha256(raw).hexdigest(),
        "acoustic_gold": False,
        "speaker_count_prior": None,
        "limits": "Prior ASR is a comparison only. No WER/DER or verified identity claim without an independent reference.",
        "chunks": [],
    }
    for item in manifest["chunks"]:
        path = (fixtures / item["file"]).resolve()
        if path.parent != fixtures:
            raise ValueError("Chunk files must be adjacent to the manifest.")
        with wave.open(str(path), "rb") as stream:
            if (
                stream.getnchannels(),
                stream.getsampwidth(),
                stream.getframerate(),
            ) != (1, 2, RATE):
                raise ValueError("Expected mono PCM16 at 16 kHz.")
            pcm = stream.readframes(stream.getnframes())
        if sha256(pcm).hexdigest() != item["pcm_sha256"]:
            raise ValueError("Replay audio changed after preparation.")
        chunk = queue.enqueue_pcm(
            item["session"],
            item["channel"],
            item["sequence"],
            item["start"],
            pcm,
            commit_start=item["commit_start"],
            commit_end=item["commit_end"],
            source=item["source"],
        )
        started = time.monotonic()
        metrics = worker.run_once()
        done = next(x for x in queue.ready("review") if x["id"] == chunk)
        report["chunks"].append(
            {
                "source": item,
                "fresh": done,
                "metrics": metrics,
                "seconds": time.monotonic() - started,
                "review": "pending",
            }
        )
        queue.acknowledge("review", chunk)
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps({"chunk": chunk, "seconds": report["chunks"][-1]["seconds"]}),
            flush=True,
        )
    return report


def assist(fixtures, transcribed, home, context, output):
    """Use actual recognized turns, with optional explicitly reviewed speaker mappings."""
    from .campaigns import Campaigns
    from .copilot import make_copilot
    from .live_audio import deliver_to_campaign
    from .settings import load, save
    from .store import Store

    fixtures = Path(fixtures).expanduser().resolve()
    raw = (fixtures / "manifest.json").read_bytes()
    transcription_bytes = Path(transcribed).read_bytes()
    acoustic = json.loads(transcription_bytes)
    if acoustic.get("manifest_sha256") != sha256(raw).hexdigest():
        raise ValueError(
            "The fresh transcription must match the frozen audio manifest."
        )
    setup = json.loads(Path(context).read_text())
    output = private_output(output)
    store = Store(output / "workspace")
    save(store.home, load(home))
    campaigns = Campaigns(store)
    cid = campaigns.create(
        "Recorded conversation evaluation",
        direction=setup.get(
            "direction",
            "Preserve uncertain identity and incomplete speech. Do not invent player choices.",
        ),
    )
    characters = {
        c["name"]: campaigns.save_character(cid, c["name"], c.get("sheet", {}))
        for c in setup.get("characters", [])
    }
    for document in setup.get("documents", []):
        campaigns.add_document(
            cid,
            document["title"],
            document["text"],
            visibility=document.get("visibility", "private"),
            metadata=document.get("metadata", {}),
        )
    mappings = setup.get("speaker_mappings", [])
    if (
        mappings
        and setup.get("mapping_transcription_sha256")
        != sha256(transcription_bytes).hexdigest()
    ):
        raise ValueError(
            "Speaker mappings must be reviewed for this exact transcription, not a prior clustering of the same audio."
        )
    for mapping in mappings:
        campaigns.map_participant(
            cid,
            mapping["name"],
            mapping["speaker"],
            role=mapping["role"],
            character_id=characters.get(mapping.get("character")),
        )
    sid = campaigns.create_session(cid, "Natural audio replay", proactive=False)
    build, generate = make_copilot(store)
    queue = AudioQueue(store.home / "live-audio")
    report = {
        "kind": "natural_audio_assistance",
        "status": "running",
        "campaign_id": cid,
        "acoustic_gold": False,
        "context_and_mapping_provenance": setup.get("provenance", "unreviewed"),
        "model_settings": load(home),
        "transcription_sha256": sha256(transcription_bytes).hexdigest(),
        "application_sources": {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "cases": [],
    }
    for item in acoustic["chunks"]:
        source, fresh = item["source"], item["fresh"]
        path = (fixtures / source["file"]).resolve()
        if path.parent != fixtures:
            raise ValueError("Audio must be adjacent to its manifest.")
        with wave.open(str(path), "rb") as stream:
            if (
                stream.getnchannels(),
                stream.getsampwidth(),
                stream.getframerate(),
            ) != (1, 2, RATE):
                raise ValueError("Expected mono PCM16 at 16 kHz.")
            pcm = stream.readframes(stream.getnframes())
        if (
            sha256(pcm).hexdigest() != source["pcm_sha256"]
            or fresh["audio_sha256"] != source["pcm_sha256"]
        ):
            raise ValueError("Audio or transcription provenance changed.")
        chunk = queue.enqueue_pcm(
            source["session"],
            source["channel"],
            source["sequence"],
            source["start"],
            pcm,
            commit_start=source["commit_start"],
            commit_end=source["commit_end"],
            source=source["source"],
        )
        claim = queue.claim()
        if claim["id"] != chunk or chunk != fresh["id"]:
            raise ValueError("Chunk identity changed during replay.")
        queue.complete(claim, fresh["document"])
        deliver_to_campaign(queue, campaigns, source["session"], sid)
        count = len(campaigns.messages(sid))
        started = time.monotonic()
        rid = campaigns.generate(sid, generate, build_context=build, force=True)
        run = next(r for r in campaigns.runs(sid) if r["id"] == rid)
        row = {
            "id": str(source["sequence"]),
            "seconds": time.monotonic() - started,
            "audio": item,
            "run": run,
            "quality_review": "pending",
            "recognized": " ".join(s["text"] for s in fresh["document"]["segments"]),
            "proposals": [
                p for p in campaigns.proposals(sid) if p.get("run_id") == rid
            ],
            "checks": {
                "run_completed": run["status"] == "complete",
                "guidance_did_not_become_speech": len(campaigns.messages(sid)) == count,
                "no_publication": not campaigns.publications(sid),
                "repeat_import_empty": deliver_to_campaign(
                    queue, campaigns, source["session"], sid
                )
                == 0,
            },
        }
        report["cases"].append(row)
        (output / "report.json").write_text(json.dumps(report, indent=2))
        (output / "review.html").write_text(
            '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            "<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:auto;padding:24px;background:#f4eee3;color:#352a20}section{border-top:1px solid #bca68e}pre{white-space:pre-wrap;overflow-wrap:anywhere}details pre{max-height:34rem;overflow:auto}</style>"
            "<h1>Recorded conversation → assistance</h1><p>Actual retained audio and fresh transcription. Prior ASR is not acoustic gold; speaker identities require explicit review. Model proposals remain private and are not reference answers.</p>"
            + "".join(
                f"<section><h2>Chunk {escape(x['id'])}</h2><h3>Recognized speech</h3><pre>{escape(x['recognized'])}</pre>"
                + "".join(
                    f"<h3>{escape(p['title'])}</h3><pre>{escape(p['text'])}</pre>"
                    for p in x["proposals"]
                    if p["kind"] != "state"
                )
                + f"<details><summary>Audio, source, state, model calls and timing</summary><pre>{escape(json.dumps(x, indent=2))}</pre></details></section>"
                for x in report["cases"]
            )
        )
        print(
            json.dumps(
                {"chunk": chunk, "seconds": row["seconds"], "status": run["status"]}
            ),
            flush=True,
        )
    report["status"] = "complete"
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source-queue", required=True)
    prep.add_argument("--session", required=True)
    prep.add_argument("--output", required=True)
    run = commands.add_parser("transcribe")
    run.add_argument("--fixtures", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device-index", type=int, default=1)
    assistance = commands.add_parser("assist")
    for name in ("fixtures", "transcribed", "home", "context", "output"):
        assistance.add_argument("--" + name, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare": prepare, "transcribe": transcribe, "assist": assist}[command](**args)
