"""Silent fixtures validate transport/attribution policy, not acoustic accuracy."""

import copy
import math
import shutil
import subprocess
import sys
import wave

import pytest

from story_copilot.audio_worker import (
    PersistentAudioWorker,
    SpeakerTracker,
    commit_document,
    diarization_durations,
)
from story_copilot.live_audio import (
    AudioQueue,
    PCMChunker,
    QueueFull,
    RATE,
    deliver_to_campaign,
    replay_file,
)


def pcm(seconds=4):
    return b"\x01\x00" * round(RATE * seconds)


def enqueue(queue, *, session="session-a", channel="system", sequence=0, **kwargs):
    return queue.enqueue_pcm(session, channel, sequence, sequence * 4, pcm(), **kwargs)


class FixtureBackend:
    backend_key = "synthetic-v1"

    def __init__(self):
        self.calls = 0

    def process(self, data):
        self.calls += 1
        return {
            "segments": [
                {
                    "text": "Can I search?",
                    "start": 0,
                    "end": 3,
                    "words": [
                        {"word": "Can", "start": 0.1, "end": 1, "speaker": "LOCAL_0"},
                        {"word": "I", "start": 1, "end": 2, "speaker": "LOCAL_0"},
                        {"word": "search?", "start": 2, "end": 3, "speaker": "LOCAL_0"},
                    ],
                }
            ],
            "speaker_embeddings": {"LOCAL_0": [1, 0, 0]},
            "diarization_segments": [{"start": 0, "end": 4, "speaker": "LOCAL_0"}],
            "provenance": {
                "fixture": True,
                "timings": {"warm_processing_seconds": 0.2},
            },
        }


def test_audio_imports_do_not_load_models_or_native_capture():
    source = "import sys; import story_copilot.live_audio, story_copilot.audio_worker, story_copilot.mac_audio; assert not any(x in sys.modules for x in ['torch','whisperx','CoreAudio','sounddevice','numpy'])"
    subprocess.run([sys.executable, "-c", source], check=True)


def test_queue_idempotence_conflicts_bounds_and_channel_independence(tmp_path):
    queue = AudioQueue(tmp_path, max_pending=2)
    first = enqueue(queue)
    assert enqueue(queue) == first
    with pytest.raises(ValueError, match="different audio"):
        queue.enqueue_pcm("session-a", "system", 0, 0, b"\x02\x00" * RATE * 4)
    second = enqueue(queue, channel="mic")
    assert first != second
    with pytest.raises(QueueFull):
        enqueue(queue, sequence=1)
    assert queue.status()["counts"] == {"pending": 2}
    assert queue.path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="finite"):
        queue.enqueue_pcm("a", "system", 0, float("nan"), pcm())


def test_no_recordings_inside_repository(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="outside source"):
        AudioQueue(tmp_path / "recordings")


def test_expired_lease_cannot_overwrite_new_owner_and_order_preserved(tmp_path):
    queue = AudioQueue(tmp_path)
    first = enqueue(queue)
    second = enqueue(queue, sequence=1)
    claim = queue.claim()
    assert claim["id"] == first
    assert queue.claim() is None
    with queue.db() as db:
        db.execute("UPDATE chunks SET lease_until=0 WHERE id=?", (first,))
    retry = queue.claim()
    with pytest.raises(RuntimeError, match="obsolete"):
        queue.complete(claim, {"segments": []})
    queue.complete(retry, {"segments": []})
    assert queue.claim()["id"] == second


def test_failure_requires_explicit_retry_and_preserves_audio(tmp_path):
    queue = AudioQueue(tmp_path)
    cid = enqueue(queue)

    class Failing:
        backend_key = "failing"

        def process(self, data):
            raise RuntimeError("synthetic model failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        PersistentAudioWorker(queue, Failing()).run_once()
    assert queue.status()["counts"] == {"failed": 1}
    assert queue.claim() is None
    queue.retry(cid)
    assert queue.claim()["pcm"] == pcm()


def test_chunker_overlap_ownership_and_restart_safe_replay(tmp_path):
    chunker = PCMChunker(chunk_seconds=3, overlap_seconds=1)
    parts = []
    for _ in range(8):
        parts += chunker.feed(pcm(1))
    parts += chunker.feed(b"", final=True)
    assert [
        (p["start_seconds"], p["commit_start"], p["commit_end"]) for p in parts
    ] == [(0, 0, 3), (2, 3, 6), (5, 6, 8)]
    assert [len(p["pcm"]) / (RATE * 2) for p in parts] == [4, 5, 3]
    queue = AudioQueue(tmp_path)
    ids = [queue.enqueue_pcm("a", "system", **part) for part in parts]
    assert [queue.enqueue_pcm("a", "system", **part) for part in parts] == ids
    with pytest.raises(RuntimeError):
        chunker.feed(b"")


def test_diarization_tracker_ignores_arbitrary_local_labels_and_uncertainty():
    tracker = SpeakerTracker()
    first = tracker.assign(
        {"LOCAL_0": [1, 0], "LOCAL_1": [0, 1]},
        {"LOCAL_0": 3, "LOCAL_1": 4},
        prefix="system:s",
    )
    profiles = copy.deepcopy(tracker.profiles)
    restored = SpeakerTracker(profiles)
    swapped = restored.assign(
        {"LOCAL_1": [1, 0], "LOCAL_0": [0, 1]},
        {"LOCAL_0": 3, "LOCAL_1": 4},
        prefix="system:s",
    )
    assert swapped["LOCAL_1"]["speaker"] == first["LOCAL_0"]["speaker"]
    assert swapped["LOCAL_0"]["speaker"] == first["LOCAL_1"]["speaker"]
    uncertain = restored.assign(
        {"a": [0.7, 0.7], "short": [1, 0], "nan": [math.nan, 1]},
        {"a": 3, "short": 0.4, "nan": 5},
        prefix="system:s",
    )
    assert all(m["speaker"] == "Unknown" for m in uncertain.values())
    collision = restored.assign(
        {"a": [1, 0], "b": [1, 0]}, {"a": 3, "b": 3}, prefix="system:s"
    )
    assert collision["a"]["speaker"] != "Unknown"
    assert collision["b"]["speaker"] == "Unknown"


def test_overlap_speech_not_used_for_identification():
    raw = {
        "diarization_segments": [
            {"start": 0, "end": 3, "speaker": "a"},
            {"start": 2, "end": 4, "speaker": "b"},
        ]
    }
    assert diarization_durations(raw) == {"a": 2, "b": 1}


def test_publish_only_owned_words_and_leave_overlap_unidentified(tmp_path):
    queue = AudioQueue(tmp_path)
    queue.enqueue_pcm("s", "system", 0, 2, pcm(), commit_start=3, commit_end=5)
    chunk = queue.claim()
    raw = {
        "segments": [
            {
                "text": "prior hello shared tail missing",
                "words": [
                    {"word": "prior", "start": 0, "end": 0.5, "speaker": "a"},
                    {"word": "hello", "start": 1, "end": 1.4, "speaker": "a"},
                    {
                        "word": "shared",
                        "start": 1.4,
                        "end": 2,
                        "speaker": "a",
                        "overlap_ambiguous": True,
                    },
                    {"word": "tail", "start": 3, "end": 3.5, "speaker": "a"},
                    {"word": "missing"},
                ],
            }
        ]
    }
    saved = copy.deepcopy(raw)
    doc = commit_document(
        raw, chunk, {"a": {"speaker": "system:voice1", "status": "matched"}}
    )
    words = doc["segments"][0]["words"]
    assert [w["word"] for w in words] == ["hello", "shared"]
    assert [w["speaker"] for w in words] == ["system:voice1", "Unknown"]
    assert words[0]["start"] == 3
    assert doc["unlocated_words"] == [{"word": "missing"}]
    assert raw == saved


def test_worker_reuses_backend_persists_voice_mapping_and_reports_rtf(tmp_path):
    queue = AudioQueue(tmp_path)
    enqueue(queue)
    enqueue(queue, sequence=1)
    backend = FixtureBackend()
    worker = PersistentAudioWorker(queue, backend)
    first = worker.run_once()
    worker.run_once()
    assert first["timings"]["real_time_factor"] == 0.05
    assert backend.calls == 2
    assert worker.run_once() is None
    results = AudioQueue(tmp_path).ready("reviewer")
    assert len(results) == 2
    assert (
        results[0]["document"]["segments"][0]["words"][0]["speaker"]
        == results[1]["document"]["segments"][0]["words"][0]["speaker"]
    )
    assert results[1]["document"]["segments"][0]["start"] == 4.1


def test_campaign_crash_redelivery_is_idempotent_and_pruning_requires_ack(tmp_path):
    queue = AudioQueue(tmp_path)
    enqueue(queue)
    PersistentAudioWorker(queue, FixtureBackend()).run_once()

    class Campaign:
        def __init__(self):
            self.messages = {}
            self.fail = True

        def add_message(self, session, speaker, text, **kwargs):
            self.messages.setdefault(
                kwargs["external_id"],
                dict(session=session, speaker=speaker, text=text, **kwargs),
            )
            if self.fail:
                self.fail = False
                raise RuntimeError("crash after commit before queue ack")

    campaign = Campaign()
    assert queue.prune_audio("campaign:play", before=float("inf")) == 0
    with pytest.raises(RuntimeError):
        deliver_to_campaign(queue, campaign, "session-a", "play")
    assert deliver_to_campaign(queue, campaign, "session-a", "play") == 1
    assert deliver_to_campaign(queue, campaign, "session-a", "play") == 0
    assert len(campaign.messages) == 1
    turn = next(iter(campaign.messages.values()))
    assert turn["role"] == "unknown"
    assert turn["visibility"] == "private"
    assert turn["source"]["channel"] == "system"
    assert queue.prune_audio("campaign:play", before=float("inf")) == 1
    assert queue.status()["retained_audio_bytes"] == 0


@pytest.mark.skipif(
    not shutil.which("ffmpeg"), reason="ffmpeg optional replay dependency"
)
def test_file_replay_is_silent_resumable_and_source_linked(tmp_path):
    source = tmp_path / "fixture.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm(7))
    queue = AudioQueue(tmp_path / "queue")
    assert (
        replay_file(
            queue, source, session_id="replay", chunk_seconds=3, overlap_seconds=1
        )
        == 3
    )
    assert (
        replay_file(
            queue, source, session_id="replay", chunk_seconds=3, overlap_seconds=1
        )
        == 3
    )
    assert queue.status()["counts"] == {"pending": 3}
    claim = queue.claim()
    assert claim["source"]["kind"] == "file_replay"
    assert len(claim["source"]["sha256"]) == 64
    assert claim["commit_end"] == 3


def test_persistent_stdio_transfer_keeps_channels_and_resumes(tmp_path):
    from story_copilot.live_audio import SSHQueueTransport

    local = AudioQueue(tmp_path / "local")
    remote = AudioQueue(tmp_path / "remote")
    enqueue(local, channel="mic")
    enqueue(local, channel="system")
    transport = SSHQueueTransport("fixture-host", str(remote.home))
    # Exercise the actual receiver/protocol without network/credentials/hardware.
    transport.command = [
        sys.executable,
        "-m",
        "story_copilot.live_audio",
        "--queue",
        str(remote.home),
        "receive",
    ]
    try:
        assert transport.transfer(local) == 2
        process_id = transport.process.pid
        enqueue(local, channel="mic", sequence=1)
        assert transport.transfer(local) == 1
        assert transport.process.pid == process_id
        assert transport.transfer(local) == 0
    finally:
        transport.close()
    assert local.status()["counts"] == {"forwarded": 3}
    assert remote.status()["counts"] == {"pending": 3}
    # Emulate lost acknowledgment: resend identical source, not a new recording.
    with local.db() as db:
        db.execute("UPDATE chunks SET status='pending'")
    try:
        assert transport.transfer(local) == 3
    finally:
        transport.close()
    assert remote.status()["counts"] == {"pending": 3}


def test_capture_is_opt_in_and_preserves_partial_audio_on_transfer_failure(
    tmp_path, monkeypatch
):
    import threading
    import story_copilot.mac_audio as module
    from story_copilot.live_audio import recover_captures

    queue = AudioQueue(tmp_path)
    stop = threading.Event()
    instances = []

    class MicrophoneFixture:
        def __init__(self, device):
            self.stopped = False
            self.delivered = False
            instances.append(self)

        def start(self):
            pass

        def drain(self):
            if self.delivered:
                return b""
            self.delivered = True
            return pcm(1)

        def stop(self):
            self.stopped = True

    class FailedTransport:
        def transfer(self, queue):
            raise RuntimeError("fixture disconnect")

    monkeypatch.setattr(module, "Microphone", MicrophoneFixture)
    with pytest.raises(ValueError, match="Explicitly enable"):
        module.capture(queue, session_id="live")
    with pytest.raises(RuntimeError, match="fixture disconnect"):
        module.capture(
            queue, session_id="live", mic=True, stop=stop, transport=FailedTransport()
        )
    assert all(i.stopped for i in instances)
    manifests = list(tmp_path.glob("capture-recovery-*.json"))
    assert len(manifests) == 0
    assert queue.status()["counts"] == {"pending": 1}
    assert recover_captures(queue) == []
    claim = queue.claim()
    assert claim["pcm"] == pcm(1)
    assert claim["channel"] == "mic"
    assert claim["commit_end"] - claim["commit_start"] == pytest.approx(1)


def test_unaligned_roll_numbers_are_retained_without_inventing_timestamps(tmp_path):
    queue = AudioQueue(tmp_path)
    enqueue(queue)
    chunk = queue.claim()
    raw = {
        "segments": [
            {
                "text": "I rolled 37",
                "start": 0,
                "end": 2,
                "words": [
                    {"word": "I", "start": 0, "end": 0.5, "speaker": "a"},
                    {"word": "rolled", "start": 0.5, "end": 1.1, "speaker": "a"},
                    {"word": "37"},
                ],
            }
        ]
    }
    doc = commit_document(raw, chunk, {"a": {"speaker": "voice1", "status": "matched"}})
    word = doc["segments"][0]["words"][-1]
    assert word["word"] == "37"
    assert word["speaker"] == "Unknown"
    assert "start" not in word and "end" not in word
    assert word["ownership_anchor_seconds"] == 0.8


def test_corrupted_spool_is_rejected_before_model_call(tmp_path):
    queue = AudioQueue(tmp_path)
    cid = enqueue(queue)
    with queue.db() as db:
        db.execute("UPDATE chunks SET pcm=? WHERE id=?", (b"\x02\x00" * RATE * 4, cid))
    backend = FixtureBackend()
    with pytest.raises(ValueError, match="source hash"):
        PersistentAudioWorker(queue, backend).run_once()
    assert backend.calls == 0
    assert queue.status()["counts"] == {"failed": 1}


def test_microphone_closes_stream_when_start_or_stop_fails(monkeypatch):
    from types import SimpleNamespace
    from story_copilot.mac_audio import Microphone

    class Stream:
        def __init__(self, fail_start=False, fail_stop=False):
            self.fail_start = fail_start
            self.fail_stop = fail_stop
            self.closed = False

        def start(self):
            if self.fail_start:
                raise RuntimeError("input start failed")

        def stop(self):
            if self.fail_stop:
                raise RuntimeError("input stop failed")

        def close(self):
            self.closed = True

    failed_start = Stream(fail_start=True)
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(RawInputStream=lambda **kwargs: failed_start),
    )
    mic = Microphone()
    with pytest.raises(RuntimeError, match="start failed"):
        mic.start()
    assert failed_start.closed and mic.stream is None

    failed_stop = Stream(fail_stop=True)
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(RawInputStream=lambda **kwargs: failed_stop),
    )
    mic.start()
    with pytest.raises(RuntimeError, match="already open"):
        mic.start()
    with pytest.raises(RuntimeError, match="stop failed"):
        mic.stop()
    assert failed_stop.closed and mic.stream is None
    mic.stop()


def test_shared_capture_origin_survives_restart_and_retains_gap(tmp_path):
    from story_copilot.live_audio import TimedPCMChunker, audio_timeline

    q = AudioQueue(tmp_path)
    clock = q.capture_clock("play", boot_id="boot-a", utc_now=1000, monotonic_now=200)
    restarted = q.capture_clock(
        "play", boot_id="boot-a", utc_now=1060, monotonic_now=260
    )
    assert clock["domain"] == restarted["domain"]
    assert restarted["origin_monotonic"] == 200
    assert restarted["origin_utc"] == 1000
    assert restarted["utc_anchor_valid"]
    mic = TimedPCMChunker(chunk_seconds=3, overlap_seconds=0)
    system = TimedPCMChunker(chunk_seconds=3, overlap_seconds=0)
    mic_parts = mic.feed(pcm(1), start_seconds=10)
    mic_parts += mic.feed(pcm(1), start_seconds=70)
    mic_parts += mic.feed(b"", final=True)
    system_parts = system.feed(pcm(1), start_seconds=10.5)
    system_parts += system.feed(b"", final=True)
    assert [(p["commit_start"], p["commit_end"]) for p in mic_parts] == [
        (10, 11),
        (70, 71),
    ]
    assert system_parts[0]["commit_start"] == 10.5
    for channel, parts in [("mic", mic_parts), ("system", system_parts)]:
        for part in parts:
            q.enqueue_pcm(
                "play", channel, source={"kind": "live_capture", "clock": clock}, **part
            )
    timeline = audio_timeline(
        {"kind": "live_capture", "clock": clock},
        10.5,
        11.5,
        session="play",
        channel="system",
    )
    assert timeline["utc_start"] == 1010.5
    assert timeline["domain"] == clock["domain"]
    jumped = q.capture_clock("play", boot_id="boot-a", utc_now=1200, monotonic_now=270)
    assert not jumped["utc_anchor_valid"]
    assert (
        audio_timeline(
            {"kind": "live_capture", "clock": jumped},
            70,
            71,
            session="play",
            channel="mic",
        )["utc_start"]
        is None
    )
    rebooted = q.capture_clock("play", boot_id="boot-b", utc_now=2000, monotonic_now=5)
    assert rebooted["domain"] != clock["domain"]


def test_recording_and_legacy_clock_domains_are_not_calendar_timestamps():
    from story_copilot.live_audio import audio_timeline

    result = audio_timeline(
        {
            "kind": "file_replay",
            "sha256": "recording-sha",
            "source_offset_seconds": 240,
        },
        5,
        7,
        session="s",
        channel="mixed",
    )
    assert result["domain"] == "recording:recording-sha"
    assert result["start_seconds"] == 245
    assert result["utc_start"] is None
    legacy = audio_timeline(
        {"kind": "live_capture", "clock": "per-channel sample time"},
        5,
        7,
        session="s",
        channel="mic",
    )
    assert legacy["kind"] == "legacy_uncertain" and legacy["utc_start"] is None
    assert "unknown" in legacy["uncertainty"]


def test_timed_turns_split_real_pause_without_modifying_source():
    from story_copilot.live_audio import timed_turns

    document = {
        "segments": [
            {
                "text": "First. Later.",
                "words": [
                    {"word": "First.", "start": 1, "end": 2, "speaker": "mic:voice1"},
                    {"word": "Later.", "start": 20, "end": 21, "speaker": "mic:voice1"},
                ],
            }
        ]
    }
    previous = copy.deepcopy(document)
    turns = list(timed_turns(document))
    assert [t["text"] for t in turns] == ["First.", "Later."]
    assert [t["start"] for t in turns] == [1, 20]
    assert document == previous


def test_device_clock_conversion_uses_first_sample_not_callback_arrival():
    import ctypes as ct
    from types import SimpleNamespace
    from story_copilot.mac_audio import (
        AudioTimeStamp,
        coreaudio_timestamp,
        portaudio_timestamp,
    )

    assert ct.sizeof(AudioTimeStamp) == 64
    stamp = AudioTimeStamp()
    stamp.host_time = 123000000000
    stamp.flags = 2
    actual, quality = coreaudio_timestamp(
        ct.cast(ct.pointer(stamp), ct.c_void_p),
        lambda ticks: ticks,
        fallback=999,
        frames=512,
        rate=48000,
    )
    assert actual == 123
    assert quality == "CoreAudio host timestamp"
    actual, quality = portaudio_timestamp(
        SimpleNamespace(inputBufferAdcTime=40.25, currentTime=40.5),
        fallback=100,
        frames=160,
        rate=16000,
    )
    assert actual == 99.75
    assert "ADC" in quality
    actual, quality = portaudio_timestamp(
        SimpleNamespace(inputBufferAdcTime=0, currentTime=0),
        fallback=100,
        frames=160,
        rate=16000,
    )
    assert actual == 99.99
    assert "unavailable" in quality


def test_timestamped_resampling_does_not_accumulate_fractional_sample_drift():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from types import SimpleNamespace
    from story_copilot.mac_audio import SystemAudioTap

    tap = SystemAudioTap()
    tap.format = SimpleNamespace(rate=48000)
    total = 0
    previous_end = None
    for index in range(200):
        start = 100 + index * 5120 / 48000
        tap.chunks.append(np.zeros(5120, dtype=np.float32))
        tap.clock_chunks.append((start, "synthetic host timestamp"))
        tap.sample_count = 5120
        packet = tap.drain_timed()[0]
        if previous_end is not None:
            assert packet["start_monotonic"] == pytest.approx(previous_end, abs=1e-8)
        total += len(packet["data"])
        previous_end = packet["start_monotonic"] + len(packet["data"]) / 16000
    assert total == round(200 * 5120 / 3)


def test_silent_source_gap_flushes_pending_speech_without_fabricating_silence():
    from story_copilot.live_audio import TimedPCMChunker

    chunks = TimedPCMChunker(chunk_seconds=30, overlap_seconds=2)
    assert chunks.feed(pcm(1), start_seconds=5) == []
    assert chunks.flush_idle(7.9) == []
    first = chunks.flush_idle(8)
    assert (
        len(first) == 1
        and first[0]["commit_start"] == 5
        and first[0]["commit_end"] == 6
    )
    assert len(first[0]["pcm"]) == RATE * 2
    assert chunks.feed(pcm(1), start_seconds=20) == []
    later = chunks.feed(b"", final=True)
    assert later[0]["commit_start"] == 20 and later[0]["commit_end"] == 21
    assert later[0]["sequence"] == 1


def test_async_two_channel_delivery_uses_speech_chronology_and_stales_old_draft(
    tmp_path, monkeypatch
):
    from datetime import datetime, timezone
    import story_copilot.campaigns as campaign_module
    from story_copilot.campaigns import Campaigns
    from story_copilot.store import Store

    base = 1800000000
    clock = {
        "domain": "capture:synthetic-shared",
        "kind": "live_capture",
        "origin_utc": base,
        "origin_monotonic": 100,
        "utc_anchor_valid": True,
        "timestamp_quality": "synthetic device timestamps",
        "utc_quality": "synthetic common-clock fixture",
    }
    queue = AudioQueue(tmp_path / "audio")
    campaigns = Campaigns(Store(tmp_path / "campaign"))
    cid = campaigns.create("Synthetic asynchronous audio")
    sid = campaigns.create_session(cid, "Two channels")

    class TimedBackend:
        backend_key = "synthetic-timed"

        def __init__(self, words):
            self.words = words

        def process(self, data):
            return {
                "segments": [
                    {
                        "text": " ".join(w["word"] for w in self.words),
                        "words": self.words,
                    }
                ],
                "speaker_embeddings": {"a": [1, 0]},
                "diarization_segments": [{"start": 0, "end": 30, "speaker": "a"}],
            }

    def utc(seconds):
        return datetime.fromtimestamp(base + seconds, timezone.utc).isoformat()

    monkeypatch.setattr(campaign_module, "now", lambda: utc(100))
    queue.enqueue_pcm(
        "audio", "mic", 0, 0, pcm(30), source={"kind": "live_capture", "clock": clock}
    )
    PersistentAudioWorker(
        queue,
        TimedBackend(
            [
                {"word": "I search the desk.", "start": 1, "end": 2, "speaker": "a"},
                {"word": "I open the safe.", "start": 20, "end": 21, "speaker": "a"},
            ]
        ),
    ).run_once()
    assert deliver_to_campaign(queue, campaigns, "audio", sid) == 1
    monkeypatch.setattr(campaign_module, "now", lambda: utc(15))
    campaigns.add_message(
        sid,
        "Typed Facilitator note",
        "The room becomes quiet.",
        role="facilitator",
        visibility="private",
        source={"kind": "manual"},
    )
    run_id, _ = campaigns.start_run(sid)
    before = campaigns.context_hash(campaigns.snapshot(sid))
    queue.enqueue_pcm(
        "audio",
        "system",
        0,
        0,
        pcm(30),
        source={"kind": "live_capture", "clock": clock},
    )
    PersistentAudioWorker(
        queue,
        TimedBackend(
            [
                {
                    "word": "There is a key in the desk.",
                    "start": 10,
                    "end": 11,
                    "speaker": "a",
                },
            ]
        ),
    ).run_once()
    monkeypatch.setattr(campaign_module, "now", lambda: utc(200))
    assert deliver_to_campaign(queue, campaigns, "audio", sid) == 1
    raw = campaigns.messages(sid)
    assert [m["ordinal"] for m in raw] == [1, 2, 3, 4]
    snapshot = campaigns.snapshot(sid)
    assert [m["text"] for m in snapshot["messages"]] == [
        "I search the desk.",
        "There is a key in the desk.",
        "The room becomes quiet.",
        "I open the safe.",
    ]
    assert [m["ordinal"] for m in snapshot["messages"]] == [1, 4, 3, 2]
    assert campaigns.context_hash(snapshot) != before
    assert campaigns.finish_run(run_id, {"suggestions": []}) == "stale"
    assert deliver_to_campaign(queue, campaigns, "audio", sid) == 0
    assert campaigns.context_hash(campaigns.snapshot(sid)) == campaigns.context_hash(
        snapshot
    )
    assert campaigns.state(sid)["applied_events"] == []


def test_private_audio_clip_checks_session_hash_and_pruning_without_playback(tmp_path):
    import io

    queue = AudioQueue(tmp_path)
    cid = enqueue(queue)
    PersistentAudioWorker(queue, FixtureBackend()).run_once()
    clip = queue.audio_clip(cid, "session-a")
    assert clip["start"] == 0 and clip["end"] == 4 and clip["channel"] == "system"
    with wave.open(io.BytesIO(clip["wav_bytes"]), "rb") as wav:
        assert (
            wav.getframerate() == RATE
            and wav.getnchannels() == 1
            and wav.getsampwidth() == 2
        )
        assert wav.readframes(wav.getnframes()) == pcm()
    with pytest.raises(ValueError, match="not found"):
        queue.audio_clip(cid, "other-session")
    with queue.db() as db:
        db.execute("UPDATE chunks SET pcm=? WHERE id=?", (b"\x02\x00" * RATE * 4, cid))
    with pytest.raises(ValueError, match="source hash"):
        queue.audio_clip(cid, "session-a")
    queue.acknowledge("test", cid)
    queue.prune_audio("test", before=float("inf"))
    with pytest.raises(FileNotFoundError, match="pruned"):
        queue.audio_clip(cid, "session-a")


def test_transfer_timeout_covers_blocked_write_and_can_reconnect(tmp_path):
    import time
    from story_copilot.live_audio import SSHQueueTransport

    queue = AudioQueue(tmp_path / "local")
    enqueue(queue)
    transport = SSHQueueTransport("fixture-host", str(tmp_path / "remote"), timeout=0.1)
    # Child never readsstdin: a96000+byte frame fills thepipe. Timeout must cover
    # writing as well as waiting for a reply, and preserve the original chunk.
    transport.command = [sys.executable, "-c", "import time; time.sleep(20)"]
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        transport.transfer(queue)
    assert time.monotonic() - started < 4
    assert transport.process is None
    assert queue.status()["counts"] == {"pending": 1}
    transport.command = [
        sys.executable,
        "-m",
        "story_copilot.live_audio",
        "--queue",
        str(tmp_path / "remote"),
        "receive",
    ]
    transport.timeout = 5
    try:
        assert transport.transfer(queue) == 1
    finally:
        transport.close()
    assert queue.status()["counts"] == {"forwarded": 1}
