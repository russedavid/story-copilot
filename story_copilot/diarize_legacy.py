"""Reuse aligned words with the pyannote 3.x runtime for an authorized 3.1 model.

Run in the optional legacy diarization environment. Speaker ambiguities remain
explicit; this does not validate speaker identities or overwrite aligned input.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def attribute_words(result, intervals, label_key="speaker", include_candidates=True):
    for segment in result["segments"]:
        for word in segment.get("words", []):
            start, end = word.get("start"), word.get("end")
            if start is None or end is None or end <= start:
                word[label_key] = "Unknown"
                continue
            scores = {}
            for a, b, speaker in intervals:
                if a >= end:
                    break
                overlap = float(max(0, min(end, b) - max(start, a)))
                if overlap:
                    scores[speaker] = scores.get(speaker, 0) + overlap
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            word[label_key] = ranked[0][0] if ranked else "Unknown"
            if include_candidates:
                word["speaker_candidates"] = [
                    {"speaker": s, "overlap_seconds": round(v, 4)} for s, v in ranked
                ]
                word["overlap_ambiguous"] = bool(
                    len(ranked) > 1 and ranked[1][1] / (end - start) > 0.2
                )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("aligned")
    p.add_argument("output")
    p.add_argument("--speakers", type=int, default=4)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Choose a new output path.")
    result = json.loads(Path(args.aligned).read_text())
    meta = result["provenance"]
    audio_path = Path(meta["audio"])
    if hashlib.sha256(audio_path.read_bytes()).hexdigest() != meta["audio_sha256"]:
        raise ValueError("Audio differs from the alignment source.")
    import numpy as np
    import torch

    # Older official checkpoints store this metadata class; retain weights-only loading.
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    from huggingface_hub import get_token
    from pyannote.audio import Pipeline
    from pyannote.audio.core.task import Specifications, Problem, Resolution

    torch.serialization.add_safe_globals([Specifications, Problem, Resolution])
    started = time.monotonic()
    print("Loading authorized pyannote 3.1 pipeline", flush=True)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=get_token()
    )
    if pipeline is None:
        raise RuntimeError("The diarization pipeline could not be loaded.")
    pipeline.to(torch.device("cuda"))
    audio = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-t",
            str(meta["seconds_requested"]),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    waveform = torch.from_numpy(
        np.frombuffer(audio, dtype=np.float32).copy()
    ).unsqueeze(0)
    print("Diarizing aligned audio", flush=True)
    diarization = pipeline(
        {"waveform": waveform, "sample_rate": 16000}, num_speakers=args.speakers
    )
    intervals = sorted(
        (s.start, s.end, speaker)
        for s, _, speaker in diarization.itertracks(yield_label=True)
    )
    attribute_words(result, intervals)
    result["diarization_segments"] = [
        {"start": a, "end": b, "speaker": s} for a, b, s in intervals
    ]
    result["provenance"].update(
        {
            "diarization": "complete",
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "diarization_runtime": "pyannote.audio 3.3.2",
            "diarization_seconds": round(time.monotonic() - started, 2),
            "alignment_reused_from": str(Path(args.aligned).resolve()),
        }
    )
    result["provenance"].pop("diarization_error", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
