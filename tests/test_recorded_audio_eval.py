from hashlib import sha256
import json

import pytest

from story_copilot.live_audio import AudioQueue, RATE
from story_copilot.recorded_audio_eval import prepare


def test_export_retains_exact_audio_without_mutating_source_or_promoting_asr_to_gold(
    tmp_path,
):
    queue = AudioQueue(tmp_path / "source")
    pcm = b"\0\0" * RATE * 2  # Silent unit fixture, not an acoustic evaluation.
    queue.enqueue_pcm(
        "session",
        "mixed",
        0,
        0,
        pcm,
        source={"kind": "file_replay", "sha256": "original"},
    )
    claim = queue.claim()
    queue.complete(
        claim,
        {
            "segments": [{"text": "Unreviewed machine text."}],
            "provenance": {"review_status": "unreviewed"},
        },
    )
    before = sha256(queue.path.read_bytes()).hexdigest()
    result = prepare(queue.home, "session", tmp_path / "export")
    assert sha256(queue.path.read_bytes()).hexdigest() == before
    assert (
        result["baseline_is_gold"] is False
        and result["speaker_identity"] == "unverified"
    )
    assert result["chunks"][0]["pcm_sha256"] == sha256(pcm).hexdigest()
    assert (
        result["chunks"][0]["baseline_asr_unreviewed"]["segments"][0]["text"]
        == "Unreviewed machine text."
    )
    assert (
        json.loads((tmp_path / "export/manifest.json").read_text())["capture"] is False
    )
    with pytest.raises(ValueError, match="new private"):
        prepare(queue.home, "session", tmp_path / "export")


def test_assistance_imports_fresh_words_and_keeps_unmapped_speakers_unknown(
    tmp_path, monkeypatch
):
    from story_copilot.recorded_audio_eval import assist
    from story_copilot.settings import save

    queue = AudioQueue(tmp_path / "source")
    queue.enqueue_pcm(
        "session",
        "mixed",
        0,
        0,
        b"\0\0" * RATE * 2,
        source={"kind": "file_replay", "sha256": "original"},
    )
    queue.complete(queue.claim(), {"segments": [{"text": "Earlier unreviewed ASR."}]})
    fixtures = tmp_path / "fixtures"
    manifest = prepare(queue.home, "session", fixtures)
    item = queue.ready("comparison")[0]
    item["document"] = {
        "segments": [
            {
                "text": "Fresh words.",
                "start": 0.1,
                "end": 1.0,
                "words": [
                    {"word": "Fresh", "start": 0.1, "end": 0.5, "speaker": "voice-1"},
                    {"word": "words.", "start": 0.6, "end": 1.0, "speaker": "voice-1"},
                ],
            }
        ]
    }
    report = {
        "manifest_sha256": sha256(
            (fixtures / "manifest.json").read_bytes()
        ).hexdigest(),
        "chunks": [
            {
                "source": manifest["chunks"][0],
                "fresh": item,
                "metrics": {},
                "seconds": 0.01,
            }
        ],
    }
    path = tmp_path / "transcribed.json"
    path.write_text(json.dumps(report))
    context = tmp_path / "context.json"
    context.write_text("{}")
    home = tmp_path / "home"
    home.mkdir()
    save(home, {})
    seen = []
    manual = []

    def build(snapshot):
        seen.extend(snapshot["messages"])
        return {**snapshot, "work_protocol": "facilitator-copilot-v2"}

    def generate(context):
        manual.append(context.get("manual_request"))
        return {"suggestions": []}

    monkeypatch.setattr(
        "story_copilot.copilot.make_copilot",
        lambda store: (build, generate),
    )
    result = assist(fixtures, path, home, context, tmp_path / "results")
    assert result["status"] == "complete" and result["acoustic_gold"] is False
    assert result["cases"][0]["recognized"] == "Fresh words."
    assert seen and all(m["role"] == "unknown" for m in seen)
    assert manual == [True]
    assert all(result["cases"][0]["checks"].values())
