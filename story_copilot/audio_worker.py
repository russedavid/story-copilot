"""Persistent local ASR/alignment/diarization worker, isolated from the web process.

Run inside the WhisperX environment. Models load on first queued work, stay resident,
and use one explicit GPU. No recording, model download or CUDA initialization occurs
on import. Speaker matches are provisional acoustic clusters, not verified identities.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import threading
import time

from .live_audio import AudioQueue, RATE, packed, sha


def normalized(vector):
    if not vector or not all(math.isfinite(float(v)) for v in vector):
        return None
    magnitude = math.sqrt(sum(float(v) ** 2 for v in vector))
    return [float(v) / magnitude for v in vector] if magnitude > 1e-9 else None


def cosine(a, b):
    if len(a) != len(b):
        return -1.0
    return sum(x * y for x, y in zip(a, b))


class SpeakerTracker:
    """Conservative matching, scoped to a session/channel/model embedding space.

    Thresholds are policy defaults, not calibrated accuracy claims. Borderline,
    short, missing and colliding embeddings remain Unknown. No cross-channel voice
    identity is inferred; a microphone can contain more than one person.
    """

    def __init__(
        self,
        profiles=(),
        *,
        match_threshold=0.85,
        novelty_threshold=0.55,
        margin=0.12,
        min_seconds=2,
    ):
        if not -1 <= novelty_threshold < match_threshold <= 1 or not 0 <= margin <= 2:
            raise ValueError("Invalid speaker similarity thresholds.")
        self.profiles = copy.deepcopy(list(profiles))
        self.match_threshold, self.novelty_threshold = (
            match_threshold,
            novelty_threshold,
        )
        self.margin, self.min_seconds = margin, min_seconds

    def assign(self, embeddings, durations, *, prefix):
        matches, used = {}, set()
        # Long clean samples are most useful; local diarization labels are never
        # assumed to correspond to the same person in the next window.
        for label in sorted(embeddings, key=lambda s: (-durations.get(s, 0), s)):
            vector = normalized(embeddings[label])
            decision = {
                "speaker": "Unknown",
                "local_speaker": label,
                "status": "uncertain",
            }
            if vector is None or durations.get(label, 0) < self.min_seconds:
                decision["reason"] = (
                    "missing/invalid embedding or too little clean speech"
                )
                matches[label] = decision
                continue
            ranked = sorted(
                (
                    (cosine(vector, normalized(p["embedding"]) or []), p)
                    for p in self.profiles
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
            best = ranked[0][0] if ranked else -1.0
            second = ranked[1][0] if len(ranked) > 1 else -1.0
            decision["similarity"] = round(best, 5) if ranked else None
            if (
                ranked
                and best >= self.match_threshold
                and best - second >= self.margin
                and ranked[0][1]["speaker"] not in used
            ):
                profile = ranked[0][1]
                decision.update(
                    speaker=profile["speaker"],
                    status="matched",
                    reason="unique acoustic match; identity remains unverified",
                )
                # Do not move a centroid toward weak matches and accumulate drift.
                if best >= max(self.match_threshold, 0.92):
                    count = min(profile["observations"], 20)
                    profile["embedding"] = normalized(
                        [a * count + b for a, b in zip(profile["embedding"], vector)]
                    )
                    profile["observations"] += 1
            elif not ranked or best < self.novelty_threshold:
                name = f"{prefix}:voice-{len(self.profiles) + 1:03d}"
                self.profiles.append(
                    {"speaker": name, "embedding": vector, "observations": 1}
                )
                decision.update(
                    speaker=name,
                    status="new",
                    reason="new provisional acoustic cluster",
                )
            else:
                decision["reason"] = (
                    "ambiguous, borderline, or conflicting acoustic match"
                )
            if decision["speaker"] != "Unknown":
                used.add(decision["speaker"])
            matches[label] = decision
        return matches


def diarization_durations(document):
    """Use non-overlapping speech only as support for cross-chunk matching."""
    intervals = document.get("diarization_segments", [])
    boundaries = sorted({float(x[k]) for x in intervals for k in ("start", "end")})
    durations = {}
    for a, b in zip(boundaries, boundaries[1:]):
        active = {x["speaker"] for x in intervals if x["start"] < b and x["end"] > a}
        if len(active) == 1:
            speaker = next(iter(active))
            durations[speaker] = durations.get(speaker, 0) + b - a
    return durations


def commit_document(raw, chunk, matches):
    """Keep original raw trace; publish only words owned by this chunk's interval."""
    result = copy.deepcopy(raw)
    result["raw_segments"] = result.pop("segments", [])
    result["segments"] = []
    unlocated = []
    for segment in result["raw_segments"]:
        words = []
        originals = segment.get("words", [])
        for index, original in enumerate(originals):
            word = copy.deepcopy(original)
            a, b = word.get("start"), word.get("end")
            located = (
                a is not None
                and b is not None
                and math.isfinite(a)
                and math.isfinite(b)
                and b >= a
            )
            if located:
                word["chunk_start_seconds"], word["chunk_end_seconds"] = a, b
                a, b = a + chunk["start"], b + chunk["start"]
                middle = (a + b) / 2
            else:
                # Forced alignment often cannot locate digits/names. Keep their
                # text using a nearby timed word only for interval ownership;
                # never fabricate the missing word's timestamps or speaker.
                unlocated.append(original)
                neighbors = [
                    (abs(i - index), w)
                    for i, w in enumerate(originals)
                    if w.get("start") is not None
                    and w.get("end") is not None
                    and math.isfinite(w["start"])
                    and math.isfinite(w["end"])
                ]
                if neighbors:
                    neighbor = min(neighbors, key=lambda pair: pair[0])[1]
                    middle = chunk["start"] + (neighbor["start"] + neighbor["end"]) / 2
                elif (
                    segment.get("start") is not None and segment.get("end") is not None
                ):
                    middle = chunk["start"] + (segment["start"] + segment["end"]) / 2
                else:
                    continue  # Available in raw trace, but cannot own an interval.
                word.pop("start", None)
                word.pop("end", None)
                word["ownership_anchor_seconds"] = round(middle, 6)
                word["alignment"] = (
                    "unlocated; neighboring word/segment used only for chunk ownership"
                )
            if not chunk["commit_start"] <= middle < chunk["commit_end"]:
                continue
            local = word.get("speaker", "Unknown")
            mapping = matches.get(local, {})
            # An exclusive diarization timeline assigns one label during overlap;
            # that convenience must not turn ambiguous speech into known identity.
            speaker = (
                "Unknown"
                if not located or word.get("overlap_ambiguous")
                else mapping.get("speaker", "Unknown")
            )
            if located:
                word.update(start=round(a, 6), end=round(b, 6))
            word.update(
                speaker=speaker,
                local_speaker=local,
                speaker_match=mapping.get("status", "uncertain"),
            )
            words.append(word)
        if words:
            result["segments"].append(
                {
                    "start": next((w["start"] for w in words if "start" in w), None),
                    "end": next(
                        (w["end"] for w in reversed(words) if "end" in w), None
                    ),
                    "text": " ".join(w.get("word", "") for w in words).strip(),
                    "words": words,
                }
            )
        elif not segment.get("words") and segment.get("text", "").strip():
            unlocated.append(
                {"segment_text": segment["text"], "reason": "no aligned words"}
            )
            a, b = segment.get("start"), segment.get("end")
            if (
                a is not None
                and b is not None
                and math.isfinite(a)
                and math.isfinite(b)
            ):
                a, b = a + chunk["start"], b + chunk["start"]
                if chunk["commit_start"] <= (a + b) / 2 < chunk["commit_end"]:
                    result["segments"].append(
                        {
                            "start": a,
                            "end": b,
                            "text": segment["text"],
                            "speaker": "Unknown",
                            "words": [],
                            "alignment": "segment only",
                        }
                    )
    result["unlocated_words"] = unlocated
    result["speaker_matches"] = matches
    result.setdefault("provenance", {}).update(
        {
            "chunk_id": chunk["id"],
            "session": chunk["session"],
            "channel": chunk["channel"],
            "sequence": chunk["sequence"],
            "pcm_sha256": chunk["audio_sha256"],
            "sample_rate": RATE,
            "audio_interval": [chunk["start"], chunk["end"]],
            "commit_interval": [chunk["commit_start"], chunk["commit_end"]],
            "source": chunk["source"],
            "review_status": "unreviewed",
            "timing": "published segments use session seconds; raw segments/diarization use chunk seconds",
            "speaker_identity": "provisional acoustic clusters; no automatic Facilitator/player assignment",
            "boundary_policy": "word midpoint in half-open nonoverlapping commit interval",
        }
    )
    return result


class WhisperXDiarizer:
    def __init__(
        self,
        *,
        device_index=1,
        device="cuda",
        batch_size=4,
        language="en",
        diarization_model="pyannote/speaker-diarization-community-1",
        num_speakers=None,
        compute_type="float16",
    ):
        if device not in {"cpu", "cuda"} or device_index < 0 or batch_size < 1:
            raise ValueError("Invalid audio worker device/batch configuration.")
        self.device, self.device_index = device, device_index
        self.batch_size, self.language = batch_size, language
        self.diarization_model, self.num_speakers = diarization_model, num_speakers
        self.compute_type = (
            "float32" if device == "cpu" and compute_type == "float16" else compute_type
        )
        self.loaded = False
        self.load_seconds = 0.0
        self.backend_key = diarization_model

    def load(self):
        if self.loaded:
            return
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
        started = time.monotonic()
        import torch
        import whisperx
        from huggingface_hub import get_token, try_to_load_from_cache
        from pyannote.audio import Pipeline

        self.torch, self.whisperx = torch, whisperx
        self.target_device = (
            f"cuda:{self.device_index}" if self.device == "cuda" else "cpu"
        )
        if self.device == "cuda":
            torch.cuda.set_device(self.device_index)
        self.asr = whisperx.load_model(
            "large-v3",
            self.device,
            device_index=self.device_index,
            compute_type=self.compute_type,
            language=self.language,
        )
        self.aligner, self.align_metadata = whisperx.load_align_model(
            language_code=self.language, device=self.target_device
        )
        self.diarizer = Pipeline.from_pretrained(
            self.diarization_model, token=get_token()
        )
        self.diarizer.to(torch.device(self.target_device))
        config = try_to_load_from_cache(self.diarization_model, "config.yaml")
        self.diarization_revision = None
        if isinstance(config, str) and "/snapshots/" in config:
            self.diarization_revision = config.split("/snapshots/", 1)[1].split("/", 1)[
                0
            ]
            self.backend_key = f"{self.diarization_model}@{self.diarization_revision}"
        self.loaded = True
        self.load_seconds = time.monotonic() - started
        self.memory_after_load = self.memory()

    def memory(self):
        if self.device != "cuda":
            return {}
        free, total = self.torch.cuda.mem_get_info(self.device_index)
        return {
            "device_used_mib": round((total - free) / 1024**2, 2),
            "device_free_mib": round(free / 1024**2, 2),
            "torch_reserved_mib": round(
                self.torch.cuda.memory_reserved(self.device_index) / 1024**2, 2
            ),
            "scope": "device totals include other processes; CTranslate2 is outside torch allocator",
        }

    def process(self, pcm):
        cold = not self.loaded
        self.load()
        import numpy as np
        from .diarize_legacy import attribute_words

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        started = time.monotonic()
        transcribed = self.asr.transcribe(
            audio, batch_size=self.batch_size, language=self.language
        )
        transcribe_finished = time.monotonic()
        if not transcribed["segments"]:
            result = {
                "segments": [],
                "speaker_embeddings": {},
                "diarization_segments": [],
                "exclusive_diarization_segments": [],
            }
            aligned_finished = diarized_finished = transcribe_finished
        else:
            result = self.whisperx.align(
                transcribed["segments"],
                self.aligner,
                self.align_metadata,
                audio,
                self.target_device,
                return_char_alignments=False,
            )
            aligned_finished = time.monotonic()
            sample = {
                "waveform": self.torch.from_numpy(audio).unsqueeze(0),
                "sample_rate": RATE,
            }
            diarized = (
                self.diarizer(sample, num_speakers=self.num_speakers)
                if self.num_speakers
                else self.diarizer(sample)
            )
            diarized_finished = time.monotonic()
            regular = sorted(
                (s.start, s.end, speaker)
                for s, _, speaker in diarized.speaker_diarization.itertracks(
                    yield_label=True
                )
            )
            exclusive = sorted(
                (s.start, s.end, speaker)
                for s, _, speaker in diarized.exclusive_speaker_diarization.itertracks(
                    yield_label=True
                )
            )
            attribute_words(result, regular, label_key="overlap_speaker")
            attribute_words(result, exclusive, include_candidates=False)
            result["diarization_segments"] = [
                {"start": float(a), "end": float(b), "speaker": s}
                for a, b, s in regular
            ]
            result["exclusive_diarization_segments"] = [
                {"start": float(a), "end": float(b), "speaker": s}
                for a, b, s in exclusive
            ]
            result["speaker_embeddings"] = (
                {}
                if diarized.speaker_embeddings is None
                else {
                    label: vector.tolist()
                    for label, vector in zip(
                        diarized.speaker_diarization.labels(),
                        diarized.speaker_embeddings,
                    )
                }
            )
        stages = {
            "load_seconds": self.load_seconds if cold else 0.0,
            "asr_seconds": transcribe_finished - started,
            "alignment_seconds": aligned_finished - transcribe_finished,
            "diarization_seconds": diarized_finished - aligned_finished,
            "warm_processing_seconds": diarized_finished - started,
        }
        result["provenance"] = {
            "asr_model": "large-v3",
            "diarization_model": self.diarization_model,
            "diarization_revision": self.diarization_revision,
            "device": self.target_device,
            "compute_type": self.compute_type,
            "batch_size": self.batch_size,
            "language": self.language,
            "num_speakers": self.num_speakers,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("whisperx", "torch", "pyannote.audio", "faster-whisper")
            },
            "timings": {k: round(v, 5) for k, v in stages.items()},
            "diarization": "complete",
            "gpu_memory_after_load": self.memory_after_load,
            "gpu_memory_after_chunk": self.memory(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        return result


class PersistentAudioWorker:
    def __init__(self, queue, backend, *, lease_seconds=300):
        self.queue, self.backend, self.lease_seconds = queue, backend, lease_seconds

    def run_once(self):
        chunk = self.queue.claim(lease_seconds=self.lease_seconds)
        if chunk is None:
            return None
        started = time.monotonic()
        quit_heartbeat = threading.Event()
        lease_errors = []

        def heartbeat():
            while not quit_heartbeat.wait(max(0.05, self.lease_seconds / 3)):
                try:
                    self.queue.heartbeat(chunk, lease_seconds=self.lease_seconds)
                except Exception as exc:
                    lease_errors.append(exc)
                    return

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            if sha(chunk["pcm"]) != chunk["audio_sha256"]:
                raise ValueError("Audio spool payload differs from its source hash.")
            raw = self.backend.process(chunk["pcm"])
            if lease_errors:
                raise lease_errors[0]
            key = self.backend.backend_key
            tracker = SpeakerTracker(
                self.queue.voices(chunk["session"], chunk["channel"], key)
            )
            matches = tracker.assign(
                raw.get("speaker_embeddings", {}),
                diarization_durations(raw),
                prefix=f"{chunk['channel']}:{sha(chunk['session'].encode())[:8]}",
            )
            document = commit_document(raw, chunk, matches)
            timings = document["provenance"].setdefault("timings", {})
            timings["worker_seconds"] = round(time.monotonic() - started, 5)
            timings["queue_wait_seconds"] = round(
                max(0, time.time() - chunk["created"] - timings["worker_seconds"]), 5
            )
            duration = chunk["commit_end"] - chunk["commit_start"]
            timings["committed_audio_seconds"] = round(duration, 5)
            timings["real_time_factor"] = round(
                timings.get("warm_processing_seconds", timings["worker_seconds"])
                / duration,
                5,
            )
            self.queue.complete(chunk, document, voices=tracker.profiles, backend=key)
            return {
                "chunk_id": chunk["id"],
                "channel": chunk["channel"],
                "timings": timings,
                "gpu_memory": document["provenance"].get("gpu_memory_after_chunk", {}),
                "published_words": sum(
                    len(s.get("words", [])) for s in document["segments"]
                ),
                "unlocated_words": len(document["unlocated_words"]),
            }
        except Exception as exc:
            try:
                self.queue.fail(chunk, f"{type(exc).__name__}: {exc}")
            except RuntimeError:
                pass  # A new lease must remain owned by the new worker.
            raise
        finally:
            quit_heartbeat.set()
            thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--speakers", type=int)
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load speech models before waiting for audio, avoiding first-chunk startup latency.",
    )
    parser.add_argument(
        "--drain", action="store_true", help="Process current backlog, then exit."
    )
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--report", help="Write timings to a new private JSON file.")
    args = parser.parse_args()
    report = Path(args.report).expanduser().resolve() if args.report else None
    if report and report.exists():
        raise SystemExit(
            "Choose a new report filename; previous benchmarks are immutable."
        )
    queue = AudioQueue(args.queue)
    backend = WhisperXDiarizer(
        device_index=args.device_index,
        device=args.device,
        batch_size=args.batch_size,
        num_speakers=args.speakers,
    )
    worker = PersistentAudioWorker(queue, backend)
    if args.preload:
        backend.load()
        print(
            packed(
                {
                    "status": "ready",
                    "model_load_seconds": backend.load_seconds,
                    "device_index": args.device_index,
                }
            ),
            flush=True,
        )
    results = []
    try:
        while args.max_chunks is None or len(results) < args.max_chunks:
            result = worker.run_once()
            if result is None:
                if args.drain:
                    break
                time.sleep(0.25)
                continue
            results.append(result)
            print(packed(result), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if report:
            report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with report.open("x") as stream:
                json.dump(
                    {
                        "model_load_seconds": backend.load_seconds,
                        "device_index": args.device_index,
                        "chunks": results,
                        "queue": queue.status(),
                        "limits": "Latency/RTF are performance measures, not transcript or diarization accuracy.",
                    },
                    stream,
                    indent=2,
                )
            os.chmod(report, 0o600)


if __name__ == "__main__":
    main()
