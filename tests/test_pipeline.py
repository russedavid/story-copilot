import json
import re

import pytest
from starlette.testclient import TestClient

from story_copilot.corpus import import_transcript, parse_text, parse_whisperx
from story_copilot.model import draft, draft_request, extract
from story_copilot.schema import Event, Extraction, NarrationAnswer
from story_copilot.state import replay
from story_copilot.store import Store
from story_copilot.training import export_dataset, training_rows
from story_copilot.web import create_app

TEXT = """Speaker SPEAKER_00: You stand outside a shuttered station. Your name is Ada.
Speaker SPEAKER_01: I could fire my pistol, but I wait.
Speaker SPEAKER_00: The falling glass costs you two hit points.
Speaker SPEAKER_01: I examine the locked door.
Speaker SPEAKER_00: Please make a Observation check.
Speaker Unknown: Thirty-one.
Speaker SPEAKER_00: You discover a scratched brass key under the step.
"""


@pytest.fixture
def library(tmp_path):
    source = tmp_path / "episode.txt"
    source.write_text(TEXT)
    store = Store(tmp_path / "private")
    cid = import_transcript(store, source, story="station", split="train")
    return store, cid, source


def event(kind, turn, quote, **kwargs):
    kwargs.setdefault("value", None)
    return Event(
        kind=kind,
        entity="Ada",
        attribute="HP" if kind == "resource" else "observation",
        evidence=[{"turn": turn, "quote": quote}],
        **kwargs,
    )


def accept(store, cid, e):
    eid = store.add_event(cid, e)
    store.review_event(cid, eid, "accepted")
    return eid


def test_import_retains_unknown_and_unmatched_lines(library):
    store, cid, source = library
    assert len(store.turns(cid)) == 7
    assert store.turns(cid)[5]["speaker"] == "Unknown"
    assert (
        list(parse_text("unlabeled continuation"))[0]["text"]
        == "unlabeled continuation"
    )
    assert import_transcript(store, source, story="station", split="train") == cid


def test_story_variants_cannot_cross_splits(library):
    store, cid, source = library
    with pytest.raises(ValueError, match="split"):
        import_transcript(store, source, story="station", split="test")
    with pytest.raises(ValueError, match="identity"):
        import_transcript(store, source, story="another-name", split="test")


def test_word_level_speaker_boundaries():
    doc = {
        "segments": [
            {
                "text": "Roll. Twenty.",
                "start": 1,
                "end": 3,
                "words": [
                    {"word": "Roll.", "speaker": "SPEAKER_00", "start": 1, "end": 1.4},
                    {"word": "Twenty.", "speaker": "SPEAKER_01", "start": 2, "end": 3},
                ],
            }
        ]
    }
    turns = list(parse_whisperx(doc))
    assert len(turns) == 2
    assert turns[1]["start"] == 2 and turns[1]["speaker"] == "SPEAKER_01"


def test_no_unreviewed_event_updates_state(library):
    store, cid, _ = library
    store.add_event(cid, event("resource", 3, "costs you two hit points", delta=-2))
    assert replay(store, cid)["resources"] == {}


def test_unknown_total_stays_unknown_and_duplicate_is_idempotent(library):
    store, cid, _ = library
    e = event("resource", 3, "costs you two hit points", delta=-2)
    eid = accept(store, cid, e)
    assert accept(store, cid, e) == eid
    resource = replay(store, cid)["resources"]["Ada:HP"]
    assert resource["value"] is None and resource["known_delta"] == -2


def test_future_events_do_not_leak_into_prefix(library):
    store, cid, _ = library
    accept(store, cid, event("fact", 7, "scratched brass key", value="key discovered"))
    assert replay(store, cid, before=7)["facts"] == {}
    assert replay(store, cid, before=8)["facts"]


def test_claims_and_hypotheticals_do_not_become_facts(library):
    store, cid, _ = library
    accept(store, cid, event("claim", 2, "I could fire my pistol", value="armed"))
    accept(
        store,
        cid,
        event(
            "action", 2, "I could fire my pistol", value="shoot", stage="hypothetical"
        ),
    )
    state = replay(store, cid)
    assert state["facts"] == {} and state["resources"] == {} and state["pending"] == {}
    assert len(state["claims"]) == 1


def test_nonexistent_evidence_is_rejected(library):
    store, cid, _ = library
    with pytest.raises(ValueError, match="exact source quote"):
        store.add_event(cid, event("fact", 1, "a gold key", value="gold key"))


def test_correction_invalidates_old_evidence(library):
    store, cid, _ = library
    eid = accept(store, cid, event("resource", 3, "costs you two hit points", delta=-2))
    t = store.turns(cid)[2]
    store.revise(
        t["id"], text="The glass misses you.", role="facilitator", status="approved"
    )
    state = replay(store, cid)
    assert state["resources"] == {} and state["stale_events"] == [eid]
    with pytest.raises(ValueError, match="Source changed"):
        store.review_event(cid, eid, "accepted")


def test_optimistic_revision_guard(library):
    store, cid, _ = library
    t = store.turns(cid)[0]
    store.revise(
        t["id"], text=t["text"], role="facilitator", expected_revision=t["revision"]
    )
    with pytest.raises(ValueError, match="changed"):
        store.revise(
            t["id"],
            text="old edit",
            role="facilitator",
            expected_revision=t["revision"],
        )


def test_resolution_requires_prior_accepted_action(library):
    store, cid, _ = library
    aid = store.add_event(
        cid, event("action", 5, "Observation check", stage="requested", value="check")
    )
    eid = store.add_event(
        cid,
        event(
            "resolve",
            7,
            "discover a scratched brass key",
            value="success",
            resolves=aid,
        ),
    )
    with pytest.raises(ValueError, match="Accept the supported action"):
        store.review_event(cid, eid, "accepted")
    store.review_event(cid, aid, "accepted")
    store.review_event(cid, eid, "accepted")
    assert replay(store, cid)["pending"] == {}
    assert replay(store, cid, before=7)["pending"]


def test_private_facts_are_excluded_from_public_view(library):
    store, cid, _ = library
    accept(
        store,
        cid,
        event(
            "fact", 7, "scratched brass key", value="secret clue", visibility="private"
        ),
    )
    assert replay(store, cid)["facts"]
    assert replay(store, cid, public_only=True)["facts"] == {}


def test_resource_changes_need_established_integer_effects():
    with pytest.raises(ValueError):
        event("resource", 1, "x", delta=-1, stage="declared")
    with pytest.raises(ValueError):
        event("resource", 1, "x", value=True)
    with pytest.raises(ValueError):
        event("resource", 1, "x", value=5, delta=2)


def test_private_correction_does_not_remove_the_public_version(library):
    store, cid, _ = library
    original = accept(
        store,
        cid,
        event("fact", 1, "shuttered station", value="station", visibility="public"),
    )
    accept(
        store,
        cid,
        event(
            "fact",
            3,
            "falling glass",
            value="private correction",
            visibility="private",
            supersedes=original,
        ),
    )
    public = replay(store, cid, public_only=True)
    assert public["facts"]["Ada:observation"]["value"] == "station"
    assert (
        replay(store, cid)["facts"]["Ada:observation"]["value"] == "private correction"
    )


def test_only_reviewed_facilitator_targets_export_and_target_is_withheld(
    library, tmp_path
):
    store, cid, _ = library
    assert training_rows(store, cid)[0] == []
    for t in store.turns(cid):
        role = "facilitator" if t["speaker"] == "SPEAKER_00" else "player"
        store.revise(t["id"], text=t["text"], role=role, status="approved")
    accept(store, cid, event("fact", 7, "scratched brass key", value="key discovered"))
    rows, _ = training_rows(store, cid)
    last = next(r for r in rows if r["provenance"]["target_turn"] == 7)
    assert "scratched brass key" not in json.dumps(last["prompt"])
    assert "scratched brass key" in json.dumps(last["completion"])
    assert last["provenance"]["split"] == "train"
    out = tmp_path / "export.jsonl"
    manifest = export_dataset(store, cid, out)
    assert manifest["rows"] == len(rows) > 0
    with pytest.raises(ValueError, match="immutable"):
        export_dataset(store, cid, out)


def test_failed_model_call_is_a_reviewable_run(library):
    store, cid, _ = library

    class Broken:
        def complete(self, *a, **k):
            raise RuntimeError("Offline model")

    rid = extract(store, cid, 1, 3, model=Broken())
    run = next(r for r in store.runs(cid) if r["id"] == rid)
    assert run["status"] == "failed" and "Offline model" in run["result"]


def test_extraction_does_not_accept_out_of_window_evidence(library):
    store, cid, _ = library

    class Outside:
        def complete(self, *a, **k):
            return Extraction(
                events=[event("fact", 7, "scratched brass key", value="key")]
            ), {}

    extract(store, cid, 1, 3, model=Outside())
    assert store.events(cid) == []
    assert "target window" in store.runs(cid)[0]["result"]


def test_model_draft_never_sees_recorded_target(library):
    store, cid, _ = library
    request = draft_request(store, cid, 7)
    assert "scratched brass key" not in json.dumps(request)


def test_web_render_and_firefox_null_origin_submit(library):
    store, cid, _ = library
    with TestClient(create_app(store)) as client:
        page = client.get("/c/" + cid)
        assert page.status_code == 200 and "State before turn" in page.text
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        if token is None:
            token = re.search(r'value="([^"]+)" name="csrf_token"', page.text)
        assert token
        t = store.turns(cid)[0]
        form = {
            "csrf_token": token[1],
            "revision": t["revision"],
            "text": t["text"],
            "role": "facilitator",
            "status": "approved",
            "category": "narration",
            "character": "",
            "note": "checked",
        }
        response = client.post(
            f"/c/{cid}/turn/1",
            data=form,
            headers={"origin": "null", "sec-fetch-site": "same-origin"},
        )
        assert response.status_code == 200
        assert store.turns(cid)[0]["status"] == "approved"


def test_cross_origin_form_cannot_change_turn(library):
    store, cid, _ = library
    with TestClient(create_app(store)) as client:
        page = client.get("/c/" + cid)
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text) or re.search(
            r'value="([^"]+)" name="csrf_token"', page.text
        )
        t = store.turns(cid)[0]
        client.post(
            f"/c/{cid}/turn/1",
            data={
                "csrf_token": token[1],
                "revision": t["revision"],
                "text": "corrupt",
                "role": "facilitator",
                "status": "approved",
                "category": "gameplay",
            },
            headers={"origin": "https://evil.example"},
        )
        assert store.turns(cid)[0]["text"] == t["text"]


def test_rationale_rewording_does_not_duplicate_resource_effect(library):
    store, cid, _ = library
    first = event(
        "resource", 3, "costs you two hit points", delta=-2, rationale="explicit loss"
    )
    second = first.model_copy(update={"rationale": "The Facilitator says so."})
    assert accept(store, cid, first) == accept(store, cid, second)
    assert replay(store, cid)["resources"]["Ada:HP"]["known_delta"] == -2


def test_source_edit_during_model_run_marks_result_stale(library):
    store, cid, _ = library

    class EditingModel:
        def complete(self, *a, **k):
            t = store.turns(cid)[2]
            store.revise(t["id"], text="The glass misses.", role="facilitator")
            return Extraction(
                events=[event("resource", 3, "costs you two hit points", delta=-2)]
            ), {}

    extract(store, cid, 1, 3, model=EditingModel())
    assert store.runs(cid)[0]["status"] == "stale"
    assert store.events(cid) == []


def test_correction_only_changes_state_after_its_evidence(library):
    store, cid, _ = library
    old = accept(store, cid, event("fact", 1, "shuttered station", value="station"))
    accept(
        store, cid, event("fact", 7, "scratched brass key", value="key", supersedes=old)
    )
    assert (
        replay(store, cid, before=7)["facts"]["Ada:observation"]["value"] == "station"
    )
    assert replay(store, cid)["facts"]["Ada:observation"]["value"] == "key"


def test_declared_calculation_boundaries():
    from story_copilot.profiles import calculate

    profile = {
        "tools": [
            {
                "name": "at_most",
                "description": "Inclusive limit.",
                "operation": "less_equal",
            }
        ]
    }
    assert calculate(profile, "at_most", [30, 30])["result"] is True
    assert calculate(profile, "at_most", [31, 30])["result"] is False
    for values in ([True, 2], [float("nan"), 2], [float("inf"), 2], [1e13, 2], [1]):
        with pytest.raises(ValueError):
            calculate(profile, "at_most", values)
    with pytest.raises(ValueError, match="not enabled"):
        calculate(profile, "undefined", [1, 2])
    with pytest.raises(ValueError, match="zero"):
        calculate(
            {
                "tools": [
                    {"name": "ratio", "description": "Divide.", "operation": "quotient"}
                ]
            },
            "ratio",
            [1, 0],
        )


def test_rules_search_remains_within_selected_edition(library):
    from story_copilot.rules import search_rules

    store, cid, _ = library
    with store.db() as db:
        db.execute(
            "INSERT INTO rules VALUES (?,?,?,?,?,?)",
            ("r7", "Rules", 1, "custom", "Observation check", "hash7"),
        )
        db.execute(
            "INSERT INTO rules VALUES (?,?,?,?,?,?)",
            ("r6", "Rules", 1, "custom 6e", "Observation check", "hash6"),
        )
        db.execute("INSERT INTO rule_search VALUES (?,?)", ("r7", "Observation check"))
        db.execute("INSERT INTO rule_search VALUES (?,?)", ("r6", "Observation check"))
    assert [r["id"] for r in search_rules(store, "Observation")] == ["r7"]


def test_quote_whitespace_is_normalized_without_changing_words():
    from story_copilot.store import source_quote

    assert source_quote("The roll  is\n31.", "The roll is 31.") == "The roll  is\n31."
    with pytest.raises(ValueError):
        source_quote("The roll is 31.", "The roll is 13.")
    with pytest.raises(ValueError):
        source_quote("The door is not open.", "The door is open.")


def test_unknown_fact_values_and_completed_pending_actions_are_rejected():
    with pytest.raises(ValueError):
        event("fact", 1, "station", value=None)
    with pytest.raises(ValueError):
        event("action", 1, "station", value="arrive", stage="established")


def test_word_overlap_is_preserved_for_review():
    from story_copilot.diarize_legacy import attribute_words

    result = {"segments": [{"words": [{"word": "yes", "start": 1, "end": 2}]}]}
    attribute_words(result, [(0, 1.5, "A"), (1.4, 3, "B")])
    word = result["segments"][0]["words"][0]
    assert word["speaker"] == "B"
    assert word["overlap_ambiguous"] and len(word["speaker_candidates"]) == 2


def test_one_invalid_proposal_does_not_discard_other_supported_proposals(library):
    from story_copilot.schema import EventCandidate

    store, cid, _ = library

    class MixedBatch:
        def complete(self, *a, **k):
            good = event("resource", 3, "costs you two hit points", delta=-2)
            bad = EventCandidate(
                kind="fact",
                entity="Ada",
                attribute="secret",
                value=None,
                evidence=[{"turn": 1, "quote": "station"}],
            )
            return Extraction(events=[good, bad]), {}

    extract(store, cid, 1, 3, model=MixedBatch())
    assert store.runs(cid)[0]["status"] == "partial"
    assert len(store.events(cid)) == 1 and store.events(cid)[0]["status"] == "pending"
    assert replay(store, cid)["resources"] == {}


def test_adjacent_facilitator_sentences_form_one_training_target(tmp_path):
    source = tmp_path / "segments.txt"
    source.write_text(
        "Speaker SPEAKER_01: I look inside.\nSpeaker SPEAKER_00: A staircase descends.\nSpeaker SPEAKER_00: You hear dripping water.\nSpeaker SPEAKER_01: I light a match."
    )
    store = Store(tmp_path / "private")
    cid = import_transcript(store, source, story="cellar", split="train")
    for t in store.turns(cid):
        store.revise(
            t["id"],
            text=t["text"],
            role="facilitator" if t["speaker"] == "SPEAKER_00" else "player",
            status="approved",
        )
    rows, _ = training_rows(store, cid)
    assert len(rows) == 1 and rows[0]["provenance"]["target_turns"] == [2, 3]
    assert (
        rows[0]["completion"][0]["content"]
        == "A staircase descends. You hear dripping water."
    )
    assert "staircase" not in json.dumps(rows[0]["prompt"])


@pytest.mark.parametrize("change", ["transcript", "state"])
def test_facilitator_draft_detects_context_changes_during_generation(library, change):
    store, cid, _ = library

    class ChangingModel:
        def complete(self, *args, **kwargs):
            if change == "transcript":
                t = store.turns(cid)[0]
                store.revise(t["id"], text="A revised description.", role="facilitator")
            else:
                accept(
                    store, cid, event("fact", 1, "shuttered station", value="station")
                )
            return NarrationAnswer(narration="An answer based on earlier context."), {}

    rid = draft(store, cid, before=4, model=ChangingModel())
    run = next(r for r in store.runs(cid) if r["id"] == rid)
    assert run["status"] == "stale"
    assert json.loads(run["result"])["accepted"] is False


def test_diarization_comparison_aligns_arbitrary_speaker_ids():
    from story_copilot.compare_diarization import compare

    left = {
        "segments": [
            {
                "words": [
                    {"word": "a", "start": 0, "end": 1, "speaker": "A"},
                    {"word": "b", "start": 1, "end": 2, "speaker": "B"},
                ]
            }
        ]
    }
    right = {
        "segments": [
            {
                "words": [
                    {"word": "a", "start": 0, "end": 1, "speaker": "B"},
                    {"word": "b", "start": 1, "end": 2, "speaker": "A"},
                ]
            }
        ]
    }
    report = compare(left, right)
    assert report["disagreed_words"] == 0
    assert report["speaker_mapping_right_to_left"] == {"A": "B", "B": "A"}
    right["segments"][0]["words"][0]["word"] = "different"
    with pytest.raises(ValueError, match="same aligned words"):
        compare(left, right)
