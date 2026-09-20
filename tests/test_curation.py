import pytest

from story_copilot.corpus import import_transcript
from story_copilot.curation import curate
from story_copilot.store import Store, digest
from story_copilot.training import training_rows


def test_curated_subspan_preserves_words_and_excludes_misattributed_reply(tmp_path):
    source = tmp_path / "scene.txt"
    source.write_text(
        "Speaker SPEAKER_01: I inspect the room.\nSpeaker SPEAKER_00: I rolled a success.\nSpeaker SPEAKER_00: The window is open.\nSpeaker SPEAKER_00: Mud covers the sill.\n"
    )
    store = Store(tmp_path / "library")
    cid = import_transcript(store, source, story="scene", split="train")
    for t in store.turns(cid):
        store.revise(
            t["id"],
            text=t["text"],
            role="player" if t["ordinal"] == 1 else "facilitator",
        )
    original = store.turns(cid)
    review = {
        "target_turns": [3, 4],
        "text_sha256": digest("The window is open. Mud covers the sill."),
        "verdict": "keep",
        "reason": "Complete Facilitator description; excludes the preceding player roll report.",
    }
    curate(store, cid, [review], "assistant-fixture")
    assert training_rows(store, cid)[0] == []
    rows, _ = training_rows(store, cid, require_curator=True)
    assert len(rows) == 1 and rows[0]["provenance"]["target_turns"] == [3, 4]
    assert (
        rows[0]["completion"][0]["content"]
        == "The window is open. Mud covers the sill."
    )
    assert store.turns(cid) == original
    store.revise(original[2]["id"], text="The window is shut.", role="facilitator")
    assert training_rows(store, cid, require_curator=True)[0] == []
    with pytest.raises(ValueError, match="differs"):
        curate(store, cid, [review], "assistant-fixture")
