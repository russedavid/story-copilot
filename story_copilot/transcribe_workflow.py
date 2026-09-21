"""Transcribe authored WAV fixtures through the real audio queue, without playback."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import wave

from .live_audio import AudioQueue, RATE, packed


def word_errors(reference, hypothesis):
    """Case/punctuation-insensitive edit distance; names and number words stay intact."""
    words = lambda value: re.findall(r"[\w']+", value.casefold())
    left, right = words(reference), words(hypothesis)
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return {"edits": previous[-1], "reference_words": len(left),
            "wer": previous[-1] / max(1, len(left))}


def fixture_pcm(root, scene):
    path = (Path(root) / scene["file"]).resolve()
    if path.parent != Path(root).resolve():
        raise ValueError("Fixture WAV must be adjacent to its manifest.")
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, RATE):
            raise ValueError("Expected mono PCM16 at 16 kHz.")
        pcm = stream.readframes(stream.getnframes())
    if hashlib.sha256(pcm).hexdigest() != scene["pcm_sha256"]:
        raise ValueError("Fixture audio differs from its manifest.")
    return pcm


def transcribe(fixtures, output, *, device_index=1, planner_home=None, speakers=1):
    from .audio_worker import PersistentAudioWorker, WhisperXDiarizer

    fixtures, output = Path(fixtures), Path(output).resolve()
    if output.exists() or any((p / ".git").exists() for p in [output, *output.parents]):
        raise ValueError("Choose a new private output directory.")
    output.mkdir(parents=True, mode=0o700)
    raw = (fixtures / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    queue = AudioQueue(output / "queue")
    backend = WhisperXDiarizer(device_index=device_index, num_speakers=speakers)
    worker = PersistentAudioWorker(queue, backend)
    def planner_probe():
        from .live_planner import LiveDecision, SYSTEM
        from .model import LocalModel
        from .settings import load
        start = time.monotonic()
        result, metrics = LocalModel(task="planner", configuration=load(planner_home)).complete(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": packed({"task": "How many cells does Neri currently have?",
                "characters": ["Neri"], "visible_evidence": [{"id": "current-note", "text": "Neri now has 3 cells, not 7."}],
                "current_state": {}, "rules_available": []})}],
            LiveDecision, max_tokens=320, temperature=0, timeout=30, constrain=False)
        return {"started": start, "finished": time.monotonic(), "answer": result.model_dump(), "model": metrics}

    rows = []
    for index, scene in enumerate(manifest["scenes"]):
        source = {"kind": "scheduled_replay", "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                  "fixture_id": scene["id"], "synthetic": True, "known_speakers_in_chunk": speakers}
        chunk = queue.enqueue_pcm("authored-workflow", scene["channel"], index, scene["start"],
                                  fixture_pcm(fixtures, scene), source=source)
        started = time.monotonic()
        # This optional probe measures genuine overlapping inference on the
        # configured endpoint. It does not feed reference text into ASR.
        if planner_home:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(planner_probe)
                asr_started = time.monotonic()
                result = worker.run_once()
                asr_finished = time.monotonic()
                probe = future.result()
                probe["overlap_seconds"] = max(0, min(asr_finished, probe["finished"]) - max(asr_started, probe["started"]))
        else:
            result = worker.run_once()
            probe = None
        item = next(x for x in queue.ready("evaluation") if x["id"] == chunk)
        hypothesis = " ".join(s["text"] for s in item["document"]["segments"])
        row = {"scene": scene["id"], "item": item, "worker": result,
               "seconds": time.monotonic() - started, "hypothesis": hypothesis,
               "reference": scene["text"], "word_errors": word_errors(scene["text"], hypothesis)}
        if probe:
            row["concurrent_planner"] = probe
        rows.append(row)
        queue.acknowledge("evaluation", chunk)
        report = {"kind": "authored_synthetic_audio", "manifest_sha256": source["manifest_sha256"],
                  "speaker_count_prior": speakers,
                  "limitations": "Clean synthesized voices; explicit speaker-count prior. Not natural speech, overlap, or diarization accuracy evidence.",
                  "rows": rows}
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(packed({k: row[k] for k in ("scene", "seconds", "hypothesis", "word_errors")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--planner-home", help="Optional configured planner for overlapping ASR/policy inference")
    parser.add_argument("--speakers", type=int, default=1, help="Known synthetic speaker count per chunk (default 1)")
    args = parser.parse_args()
    transcribe(args.fixtures, args.output, device_index=args.device_index, planner_home=args.planner_home, speakers=args.speakers)
