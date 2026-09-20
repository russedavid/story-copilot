"""Run in a dedicated WhisperX environment; never modify the source audio."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("output")
    p.add_argument("--seconds", type=int, default=5400)
    p.add_argument("--device-index", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--speakers", type=int)
    p.add_argument(
        "--aligned", help="Reuse an existing aligned JSON instead of repeating ASR."
    )
    p.add_argument(
        "--diarization-model", default="pyannote/speaker-diarization-community-1"
    )
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Output already exists; choose a new filename.")
    output.parent.mkdir(parents=True, exist_ok=True)
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_index)
    import torch
    import whisperx

    started = time.monotonic()
    print("Loading audio", flush=True)
    audio = whisperx.load_audio(args.audio)[: args.seconds * 16000]
    source_hash = hashlib.sha256(Path(args.audio).read_bytes()).hexdigest()
    if args.aligned:
        result = json.loads(Path(args.aligned).read_text())
        if (
            result["provenance"]["audio_sha256"] != source_hash
            or result["provenance"]["seconds_requested"] != args.seconds
        ):
            raise ValueError(
                "Aligned data must match the same source audio and duration."
            )
    else:
        print("Transcribing", flush=True)
        model = whisperx.load_model(
            "large-v3", "cuda", compute_type="float16", language="en"
        )
        result = model.transcribe(audio, batch_size=args.batch_size, language="en")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("Aligning words", flush=True)
        aligner, metadata = whisperx.load_align_model(language_code="en", device="cuda")
        result = whisperx.align(
            result["segments"],
            aligner,
            metadata,
            audio,
            "cuda",
            return_char_alignments=False,
        )
        del aligner
        gc.collect()
        torch.cuda.empty_cache()
    result["provenance"] = {
        "audio": str(Path(args.audio).resolve()),
        "audio_sha256": source_hash,
        "seconds_requested": args.seconds,
        "whisperx": importlib.metadata.version("whisperx"),
        "asr_model": "large-v3",
        "review_status": "unreviewed",
        "diarization": "pending",
        "alignment_reused_from": args.aligned,
        "diarization_model": args.diarization_model,
    }
    # Persist ASR before optional gated-model access, so a diarization failure loses no transcription.
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    try:
        from pyannote.audio import Pipeline
        from huggingface_hub import get_token
        from .diarize_legacy import attribute_words

        print("Diarizing speakers", flush=True)
        diarizer = Pipeline.from_pretrained(args.diarization_model, token=get_token())
        diarizer.to(torch.device("cuda"))
        sample = {
            "waveform": torch.from_numpy(audio).unsqueeze(0),
            "sample_rate": 16000,
        }
        output_diarization = (
            diarizer(sample, num_speakers=args.speakers)
            if args.speakers
            else diarizer(sample)
        )
        regular = sorted(
            (seg.start, seg.end, speaker)
            for seg, _, speaker in output_diarization.speaker_diarization.itertracks(
                yield_label=True
            )
        )
        exclusive = sorted(
            (seg.start, seg.end, speaker)
            for seg, _, speaker in output_diarization.exclusive_speaker_diarization.itertracks(
                yield_label=True
            )
        )
        attribute_words(result, regular, label_key="overlap_speaker")
        attribute_words(result, exclusive, include_candidates=False)
        result["diarization_segments"] = [
            {"start": a, "end": b, "speaker": s} for a, b, s in regular
        ]
        result["exclusive_diarization_segments"] = [
            {"start": a, "end": b, "speaker": s} for a, b, s in exclusive
        ]
        embeddings = output_diarization.speaker_embeddings
        if embeddings is not None:
            result["speaker_embeddings"] = {
                speaker: vector.tolist()
                for speaker, vector in zip(
                    output_diarization.speaker_diarization.labels(), embeddings
                )
            }
        result["provenance"]["pyannote_audio"] = importlib.metadata.version(
            "pyannote.audio"
        )
        result["provenance"]["num_speakers"] = args.speakers
        result["provenance"]["word_speaker_assignment"] = (
            "exclusive; overlapping alternatives retained"
        )
        result["provenance"]["diarization"] = "complete"
    except Exception as exc:
        result["provenance"]["diarization"] = "failed"
        result["provenance"]["diarization_error"] = (
            f"{type(exc).__name__}: {str(exc)[:700]}"
        )
        print("Diarization unavailable; aligned transcript retained.", flush=True)
    result["provenance"]["elapsed_seconds"] = round(time.monotonic() - started, 2)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
