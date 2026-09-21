"""Replay real ASR results through the live copilot; keep artifacts in a private directory."""

import argparse
from copy import deepcopy
import hashlib
from html import escape
import json
from pathlib import Path
import time

from .campaigns import Campaigns
from .copilot import make_copilot
from .live_audio import AudioQueue, deliver_to_campaign
from .settings import load, save
from .store import Store, now
from .transcribe_workflow import fixture_pcm
from .workflow_fixtures import seed


def render(report, output):
    cards = []
    for row in report["cases"]:
        trace = row["run"]["result"].get("trace", {})
        panels = [
            ("Audio, reference script and recognition", row["asr"]),
            ("Observed state and source classification", {"state": row["state"], "classification": trace.get("classification")}),
            ("Evidence decisions and verified calculations", {k: trace.get(k) for k in ("decision", "rules", "verified_answer")}),
            ("Writing and editing", {k: trace.get(k) for k in ("storyteller", "response_review")}),
            ("Complete source, state, model calls and timing", row),
        ]
        cards.append(f'<section id="{escape(row["id"])}"><h2>{escape(row["id"])}</h2>'
                     f'<p>{escape(row["expect"])}</p><h3>Recognized speech</h3>'
                     f'<p>{escape(row["recognized_speech"])}</p>'
                     f'<p>ASR: {row["asr_seconds"]:.2f}s · assistance: {row["assistance_seconds"]:.2f}s</p>'
                     f'<p>Mechanical checks: {escape(str(row["checks"]))}</p>'
                     f'<p>Semantic review: {escape(str(row.get("quality_review", "pending")))}</p>'
                     + ''.join(f'<h3>{escape(p["title"])}</h3><pre>{escape(p["text"])}</pre>'
                               for p in row["proposals"] if p["kind"] != "state")
                     + ''.join(f'<details><summary>{escape(title)}</summary><pre>{escape(json.dumps(value, indent=2))}</pre></details>'
                               for title, value in panels) + '</section>')
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Audio-to-assistance evaluation</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:auto;padding:24px;background:#f4eee3;color:#352a20}section{padding:20px;background:#fffaf3;border:1px solid #d4c4b0;margin:18px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#764e2e}details pre{max-height:34rem;overflow:auto;font:13px/1.4 ui-monospace,monospace;background:#efe5d5;padding:1rem}</style>'
        '<h1>Audio → assistance</h1><p>Authored synthesized voices, actual local transcription and model calls. '
        'Known fixture identities are assigned explicitly after transcription. No recording or playback. '
        'Timing sums exclude queueing and the time spent speaking; this is sequential replay, not live-latency or natural-speech accuracy.</p>'
        '<nav>' + ' · '.join(f'<a href="#{escape(x["id"])}">{escape(x["id"])}</a>' for x in report["cases"]) + '</nav>'
        + ''.join(cards))


def evaluate(home, fixtures, transcribed, output, *, progress=print):
    fixtures, output = Path(fixtures), Path(output).resolve()
    if output.exists() or any((p / ".git").exists() for p in [output, *output.parents]):
        raise ValueError("Choose a new private evaluation directory.")
    raw = (fixtures / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    acoustic = json.loads(Path(transcribed).read_text())
    if acoustic["manifest_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Transcriptions do not match the fixture manifest.")
    store = Store(output / "workspace")
    save(store.home, load(home))
    c = Campaigns(store)
    ids = seed(c)
    cid, sid = ids["campaign_id"], ids["session_id"]
    build, generate = make_copilot(store)
    queue = AudioQueue(store.home / "live-audio")
    report = {"created": now(), "kind": "authored_audio_workflow", "campaign_id": cid,
              "model_settings": load(home), "quality_review": "pending",
              "application_sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in Path(__file__).parent.glob('*.py')},
              "cases": [], "extra_checks": {}}

    def persist():
        (output / "report.json").write_text(json.dumps(report, indent=2))
        render(report, output)

    for scene, audio in zip(manifest["scenes"], acoustic["rows"], strict=True):
        if scene["id"] != audio["scene"]:
            raise ValueError("Transcription order differs from fixtures.")
        item = audio["item"]
        chunk = queue.enqueue_pcm(item["session"], item["channel"], item["sequence"], item["start"],
                                  fixture_pcm(fixtures, scene), source=item["source"])
        claim = queue.claim()
        if claim["id"] != item["id"] or claim["audio_sha256"] != item["audio_sha256"]:
            raise ValueError("Audio provenance changed during replay.")
        queue.complete(claim, deepcopy(item["document"]))
        before_ids = {m["id"] for m in c.messages(sid)}
        delivered = deliver_to_campaign(queue, c, item["session"], sid, visibility=scene["visibility"])
        imported = [m for m in c.messages(sid) if m["id"] not in before_ids]
        # This is an explicit fixture/operator identity assignment, not a claim
        # that acoustic diarization knows a person's role or character.
        for message in imported:
            c.revise_message(sid, message["id"], text=message["text"], speaker=scene["speaker"],
                             role=scene["role"], character="Nora" if scene["role"] == "player" else "",
                             visibility=scene["visibility"], expected_revision=message["revision"],
                             recipient=ids["character_id"] if scene["id"] == "private-clue" else None,
                             note="Evaluation fixture provides identity and audience; ASR text is unedited.")
        if scene.get("replace_rule"):
            ids["rule_id"] = c.add_document(cid, "Revised winch operating rule", scene["replace_rule"],
                                            visibility="public", metadata={"kind": "rules", "supersedes": ids["rule_id"]})
        before_count = len(c.messages(sid))
        started = time.monotonic()
        rid = c.generate(sid, generate, build_context=build)
        elapsed = time.monotonic() - started
        run = next(r for r in c.runs(sid) if r["id"] == rid)
        proposals = [p for p in c.proposals(sid) if p.get("run_id") == rid]
        duplicate = c.generate(sid, generate, build_context=build)
        state = c.state(sid)
        balance = state["resources"].get("Nora:power_cells", {}).get("value")
        trace = run["result"].get("trace", {})
        checks = {"run_completed": run["status"] == "complete", "one_audio_chunk_imported": delivered == 1,
                  "import_is_idempotent": deliver_to_campaign(queue, c, item["session"], sid) == 0,
                  "no_suggested_speech_recorded": len(c.messages(sid)) == before_count,
                  "duplicate_suppressed": duplicate is None, "nothing_published": not c.publications(sid),
                  "expected_observed_balance": balance == (6 if len(report["cases"]) < 2 else 2)}
        if scene["id"] in {"unaffordable", "affordable"}:
            calculated = trace.get("rules", {}).get("calculation") or {}
            checks["verified_cost_outcome"] = calculated.get("result") == (
                {"allowed": False, "remaining": 2} if scene["id"] == "unaffordable" else {"allowed": True, "remaining": 1})
        row = {"id": scene["id"], "expect": scene["expect"], "recognized_speech": audio["hypothesis"],
               "asr_seconds": audio["seconds"], "asr": audio, "assistance_seconds": elapsed,
               "checks": checks, "state": state, "run": run, "proposals": proposals}
        report["cases"].append(row)
        persist()
        progress(f'{scene["id"]}: {elapsed:.1f}s {checks}')

    # Actually generate with the real model, then revise the source before the
    # transaction commits. This exercises the same stale-write guard as a live edit.
    def revise_before_commit(context):
        result = generate(context)
        last = c.messages(sid)[-1]
        c.revise_message(sid, last["id"], text=last["text"] + " Correction: wait for Mara before leaving.",
                         speaker=last["speaker"], role=last["role"], character=last["character"],
                         visibility=last["visibility"], expected_revision=last["revision"])
        return result
    rid = c.generate(sid, revise_before_commit, build_context=build, force=True)
    stale = next(r for r in c.runs(sid) if r["id"] == rid)
    report["extra_checks"]["changed_source_rejects_inflight_draft"] = stale["status"] == "stale"
    report["stale_source_run"] = stale
    report["status"] = "complete"
    persist()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("home", "fixtures", "transcribed", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    evaluate(**vars(args), progress=lambda line: print(line, flush=True))
