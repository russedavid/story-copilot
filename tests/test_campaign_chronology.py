"""Synthetic clocks verify source ordering, never acoustic/model accuracy."""

from copy import deepcopy
from datetime import datetime, timezone

from story_copilot.campaigns import Campaigns
from story_copilot.copilot import make_copilot
from story_copilot.schema import Extraction, NarrationAnswer
from story_copilot.store import Store, packed

EPOCH = 1_790_000_000


def table(tmp_path):
    c = Campaigns(Store(tmp_path / "private"))
    cid = c.create("Clock fixture")
    sid = c.create_session(cid, "One conversation")
    return c, sid


def audio(
    c,
    sid,
    text,
    start,
    *,
    domain="capture:shared",
    utc=True,
    kind="live_capture",
    speaker="mic:0",
    role="player",
    end=None,
    visibility="public",
):
    return c.add_message(
        sid,
        speaker,
        text,
        role=role,
        visibility=visibility,
        source={
            "kind": "live_audio",
            "private_path": "/private/audio/source.wav",
            "timeline": {
                "kind": kind,
                "domain": domain,
                "start_seconds": start,
                "end_seconds": start + 1 if end is None else end,
                "utc_start": EPOCH + start if utc else None,
                "utc_end": EPOCH + (start + 1 if end is None else end) if utc else None,
                "quality": "synthetic exact timing",
                "uncertainty": "fixture",
            },
        },
    )


def manual(c, sid, text, seconds):
    mid = c.add_message(
        sid, "Facilitator", text, role="facilitator", source={"kind": "manual"}
    )
    with c.store.db() as db:
        db.execute(
            "UPDATE play_messages SET created=? WHERE id=?",
            (datetime.fromtimestamp(EPOCH + seconds, timezone.utc).isoformat(), mid),
        )
    return mid


def test_shared_live_clock_and_manual_entry_sort_without_changing_ingestion(tmp_path):
    c, sid = table(tmp_path)
    first = audio(c, sid, "What is behind the door?", 1)
    last = audio(c, sid, "I choose the window instead.", 20)
    typed = manual(c, sid, "The door stays shut.", 15)
    reply = audio(
        c,
        sid,
        "A quiet chamber is behind it.",
        10,
        speaker="system:0",
        role="facilitator",
    )
    assert [m["id"] for m in c.messages(sid)] == [first, last, typed, reply]
    view = c.snapshot(sid)
    assert [m["id"] for m in view["messages"]] == [first, reply, typed, last]
    assert [m["ordinal"] for m in view["messages"]] == [1, 4, 3, 2]
    assert [m["order_index"] for m in view["messages"]] == [1, 2, 3, 4]
    assert not view["chronology"]["partial_order"]
    assert any("not calibrated" in n for n in view["chronology"]["notes"])


def test_recording_seconds_never_compared_to_manual_utc_even_if_utc_field_present(
    tmp_path,
):
    c, sid = table(tmp_path)
    later = audio(
        c,
        sid,
        "Recorded later.",
        20,
        kind="recording",
        domain="recording:sha",
        utc=True,
    )
    typed = manual(
        c, sid, "A live annotation with unknown position in the recording.", 15
    )
    earlier = audio(
        c,
        sid,
        "Recorded earlier.",
        10,
        kind="recording",
        domain="recording:sha",
        utc=True,
    )
    view = c.snapshot(sid)
    assert [m["id"] for m in view["messages"]] == [earlier, typed, later]
    assert view["messages"][0]["chronology"]["start"] == 10
    assert view["messages"][0]["chronology"]["clock_group"] != "utc"
    assert view["chronology"]["partial_order"]
    assert all(not m["chronology"]["cross_group_order_known"] for m in view["messages"])
    assert any("display anchors" in n for n in view["chronology"]["notes"])


def test_unanchored_live_restart_uses_only_its_shared_monotonic_domain(tmp_path):
    c, sid = table(tmp_path)
    after_restart = audio(
        c, sid, "Restarted capture at a later offset.", 350, utc=False
    )
    before_restart = audio(
        c,
        sid,
        "Earlier captured statement delivered late.",
        30,
        utc=False,
        speaker="system:1",
    )
    view = c.snapshot(sid)
    assert [m["id"] for m in view["messages"]] == [before_restart, after_restart]
    assert not view["chronology"]["partial_order"]
    other = audio(
        c,
        sid,
        "A separate boot has an incomparable clock.",
        1,
        utc=False,
        domain="capture:other-boot",
    )
    view = c.snapshot(sid)
    assert view["chronology"]["partial_order"]
    assert (
        view["messages"][-1]["id"] == other
    )  # Ingestion anchor, not earlier relative time.


def test_legacy_uncertain_audio_stays_in_ingestion_order(tmp_path):
    c, sid = table(tmp_path)
    one = audio(
        c, sid, "Unknown microphone clock.", 100, utc=False, kind="legacy_uncertain"
    )
    two = audio(
        c,
        sid,
        "Unknown system clock.",
        1,
        utc=False,
        kind="legacy_uncertain",
        speaker="system:0",
    )
    view = c.snapshot(sid)
    assert [m["id"] for m in view["messages"]] == [one, two]
    assert view["chronology"]["unknown_clock_messages"] == 2
    assert all(m["chronology"]["start"] is None for m in view["messages"])


def test_overlapping_speech_and_tied_start_order_are_explicit(tmp_path):
    c, sid = table(tmp_path)
    one = audio(c, sid, "A long player sentence.", 1, end=8)
    two = audio(
        c,
        sid,
        "An overlapping Facilitator interjection.",
        4,
        end=6,
        speaker="system:0",
        role="facilitator",
    )
    three = audio(c, sid, "A simultaneous second player.", 4, end=7, speaker="system:1")
    view = c.snapshot(sid)
    assert [m["id"] for m in view["messages"]] == [one, two, three]
    assert view["chronology"]["overlapping_messages"] == 2
    assert all(m["chronology"]["overlaps_prior"] for m in view["messages"][1:])


def test_late_duplicate_source_invalidates_evidence_even_when_novelty_unchanged(
    tmp_path,
):
    c, sid = table(tmp_path)
    audio(c, sid, "I inspect the door.", 10)
    before = c.snapshot(sid)
    rid, _ = c.start_run(sid)
    audio(c, sid, "I inspect the door.", 9)
    after = c.snapshot(sid)
    assert c.context_hash(before) == c.context_hash(after)
    assert c.evidence_hash(before) != c.evidence_hash(after)
    assert (
        c.finish_run(
            rid, {"suggestions": [{"kind": "narration", "text": "Old output"}]}
        )
        == "stale"
    )
    assert c.proposals(sid) == []
    timing_changed = deepcopy(after)
    timing_changed["messages"][0]["chronology"]["start"] -= 0.5
    assert c.context_hash(timing_changed) == c.context_hash(after)
    assert c.evidence_hash(timing_changed) != c.evidence_hash(after)


def test_public_view_is_sorted_before_private_source_metadata_is_removed(tmp_path):
    c, sid = table(tmp_path)
    later = audio(c, sid, "Public later.", 20)
    audio(c, sid, "PRIVATE_WORDS", 5, visibility="private")
    earlier = audio(c, sid, "Public earlier.", 10, speaker="system:0")
    public = c.snapshot(sid, public_only=True)
    assert [m["id"] for m in public["messages"]] == [earlier, later]
    assert "PRIVATE_WORDS" not in packed(public)
    assert "/private/audio" not in packed(public)
    assert all("source" not in m and "original" not in m for m in public["messages"])


def test_continuation_keeps_manual_original_time_instead_of_copy_creation_time(
    tmp_path,
):
    c, sid = table(tmp_path)
    later = audio(c, sid, "Later live utterance.", 20)
    typed = manual(c, sid, "Manual entry occurred first.", 10)
    branch = c.branch(sid, "Continuation")
    view = c.snapshot(branch)
    assert [m["source"]["branched_from_message"] for m in view["messages"]] == [
        typed,
        later,
    ]
    assert view["messages"][0]["chronology"]["start"] == EPOCH + 10


def test_copilot_trigger_and_classifier_prefix_follow_chronology_not_ingestion_number(
    tmp_path,
):
    c, sid = table(tmp_path)
    first = audio(c, sid, "I am ready.", 1)
    last = audio(c, sid, "I leave the lever alone.", 20)
    prompt = audio(
        c, sid, "Do you touch the lever?", 10, speaker="system:0", role="facilitator"
    )

    class Client:
        def __init__(self, task):
            self.task = task

        def complete(self, messages, schema, **kwargs):
            return (
                Extraction()
                if self.task == "classifier"
                else NarrationAnswer(narration="The lever is untouched.")
            ), {"finish_reason": "stop"}

    config = {
        "context_limit": 16384,
        "output_reserve": 1800,
        "safety_margin": 512,
        "token_counter": lambda t: (len(t.encode()) + 3) // 4,
    }
    build, generate = make_copilot(
        c.store, model_factory=Client, context_options=config, classify_limit=2
    )
    initial = build(c.snapshot(sid))
    assert [m["id"] for m in initial["target_messages"]] == [first, prompt]
    assert initial["body"]["new_player_input"].endswith("I leave the lever alone.")
    assert initial["body"]["dialogue"][-1]["ordinal"] == 2
    assert initial["body"]["dialogue"][-1]["chronology"]["start"] == EPOCH + 20
    c.generate(sid, generate, build_context=build)
    following = build(c.snapshot(sid))
    assert [m["id"] for m in following["target_messages"]] == [last]
    assert [m["id"] for m in following["classification_prefix"]] == [first, prompt]
