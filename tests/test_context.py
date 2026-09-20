"""Original fixtures; no edits or human-review labels in the real corpus."""

import json
import sqlite3

import pytest

from story_copilot.context import (
    ContextBudgetError,
    StaleContextError,
    build_context,
    context_is_current,
    load_memory,
    pack_context,
)
from story_copilot.corpus import import_transcript
from story_copilot.schema import Event
from story_copilot.store import Store, packed


def tokens(text):
    return (len(text.encode()) + 3) // 4


def turns(count=50):
    return [
        {
            "id": f"t{i}",
            "ordinal": i,
            "revision": f"r{i}",
            "role": "facilitator" if i % 2 else "player",
            "speaker": "K" if i % 2 else "P",
            "visibility": "public",
            "text": (
                f"You reach room {i}. The dust covers a wooden desk and a chair."
                if i % 2
                else f"I carefully examine room {i} and look below the furniture."
            ),
        }
        for i in range(1, count + 1)
    ]


def empty_state():
    return {
        "entities": {},
        "facts": {},
        "resources": {},
        "knowledge": {},
        "pending": {},
        "claims": [],
    }


def compact(source, state=None, **kwargs):
    return pack_context(
        turns=source,
        state=state or empty_state(),
        context_limit=2200,
        output_reserve=400,
        safety_margin=100,
        token_counter=tokens,
        **kwargs,
    )


@pytest.fixture
def library(tmp_path):
    path = tmp_path / "source.json"
    source = turns(60)
    source[0]["text"] = "The brass key bears the name Morrow."
    source[1]["text"] = "Ada has eleven hit points."
    source[2]["text"] = "The fall costs Ada two hit points."
    source[3]["text"] = "Perhaps Morrow is the masked host."
    source[4]["text"] = "The masked host is secretly Miriam."
    source[-1]["text"] = "I return to the brass key. What does Morrow mean?"
    path.write_text(
        json.dumps(
            {"segments": [{"speaker": t["speaker"], "text": t["text"]} for t in source]}
        )
    )
    store = Store(tmp_path / "data")
    cid = import_transcript(store, path, story="original-scene")
    for t in store.turns(cid):
        store.revise(
            t["id"],
            text=t["text"],
            role="facilitator" if t["ordinal"] % 2 else "player",
            reviewer="synthetic-test",
        )
    return store, cid


def accept(store, cid, kind, turn, value=None, **kwargs):
    text = store.turns(cid)[turn - 1]["text"]
    event = Event(
        kind=kind,
        entity=kwargs.pop("entity", "Ada"),
        attribute=kwargs.pop("attribute", "HP"),
        value=value,
        evidence=[{"turn": turn, "quote": text}],
        **kwargs,
    )
    eid = store.add_event(cid, event)
    store.review_event(cid, eid, "accepted", reviewer="synthetic-test")
    return eid


def test_recent_player_trigger_and_setup_preserved_verbatim():
    source = turns()
    source[-1]["text"] = "I do not open the trapdoor; I seal it with wax."
    result = compact(
        source,
        player_input="I wait for Mara before moving.",
        direction="The knocking continues; do not reveal its cause.",
    )
    body, trace = result["body"], result["trace"]
    assert trace["compacted"]
    assert [t["text"] for t in body["dialogue"][-4:]] == [
        t["text"] for t in source[-4:]
    ]
    assert body["new_player_input"] == "I wait for Mara before moving."
    assert body["private_facilitator_direction"].startswith("The knocking")
    assert trace["prompt_tokens"] <= trace["prompt_budget"]
    assert trace["compacted_turns"]


def test_retrieved_observation_keeps_its_neighboring_ownership_reply():
    source = turns(80)
    source[0]["text"] = "The faded copper seal bears a seven-pointed star."
    source[1].update(
        speaker="Mara", text="I place it in my satchel and keep it with me."
    )
    result = compact(source, player_input="What symbol was on the faded seal?")
    passages = result["body"]["historical_source_passages"]
    assert any(p["ordinal"] == 1 and "seven-pointed" in p["quote"] for p in passages)
    owner = next(p for p in passages if p["ordinal"] == 2)
    assert owner["speaker"] == "Mara" and "my satchel" in owner["quote"]
    assert owner["revision"] == "r2"
    assert result["trace"]["prompt_tokens"] <= result["trace"]["prompt_budget"]


def test_old_clue_retrieved_without_becoming_fact_and_no_future(library):
    store, cid = library
    result = build_context(
        store,
        cid,
        60,
        player_input="Who is Morrow, whose name is on the brass key?",
        context_limit=2200,
        output_reserve=400,
        safety_margin=100,
        token_counter=tokens,
    )
    assert result["trace"]["retrieved_turns"]
    retrieved = result["body"]["historical_source_passages"]
    assert any(p["ordinal"] == 1 and "Morrow" in p["quote"] for p in retrieved)
    assert all(
        p["interpretation"] == "source_dialogue_not_established_fact" for p in retrieved
    )
    assert result["body"]["state"]["facts"] == {}
    assert 60 not in result["trace"]["retained_turns"]
    assert "60" not in result["source_revisions"]
    assert all(p["ordinal"] < 60 for p in retrieved)


def test_resources_pending_hypotheses_and_claims_remain_distinct(library):
    store, cid = library
    accept(store, cid, "resource", 2, 11)
    accept(store, cid, "resource", 3, delta=-2)
    pending = accept(
        store,
        cid,
        "action",
        59,
        "roll Observation",
        attribute="check",
        stage="requested",
    )
    accept(
        store,
        cid,
        "action",
        4,
        "Morrow may be host",
        attribute="guess",
        stage="hypothetical",
    )
    accept(store, cid, "claim", 4, "host identity", attribute="claim")
    result = build_context(
        store, cid, 61, player_input="Morrow host guess", token_counter=tokens
    )
    body = result["body"]
    assert body["state"]["resources"]["Ada:HP"]["value"] == 9
    assert pending in body["state"]["pending"]
    assert body["state"]["facts"] == {}
    uncertain = body["historical_claims_and_hypotheses"]
    assert len(uncertain) == 2
    assert any(u.get("stage") == "hypothetical" for u in uncertain)


def test_unknown_resource_total_never_invented(library):
    store, cid = library
    accept(store, cid, "resource", 3, delta=-2)
    result = build_context(store, cid, 61, token_counter=tokens)
    item = result["body"]["state"]["resources"]["Ada:HP"]
    assert item["value"] is None and item["known_delta"] == -2


def test_private_state_direction_documents_and_raw_dialogue_do_not_leak():
    state = empty_state()
    state["facts"] = {
        "host:identity": {
            "value": "Miriam",
            "visibility": "private",
            "event": "secret",
        },
        "key:metal": {"value": "brass", "visibility": "public", "event": "public"},
    }
    state["knowledge"] = {"Ada": {"code": {"value": "NIGHT", "visibility": "private"}}}
    source = turns(4)
    source[0].update(text="Miriam is the masked host", visibility="private")
    source[1].pop("visibility")
    source[1]["text"] = "Unmarked secret should not default to public"
    docs = [
        {"id": "s", "text": "The murderer's name is Miriam", "visibility": "private"}
    ]
    public = compact(
        source, state, direction="Hide Miriam", documents=docs, public_only=True
    )
    rendered = packed(public["body"])
    assert (
        "Miriam" not in rendered
        and "NIGHT" not in rendered
        and "Unmarked secret" not in rendered
    )
    assert public["body"]["private_facilitator_direction"] == ""
    private = compact(source, state, direction="Hide Miriam", documents=docs)
    assert "Miriam" in packed(private["body"])


def test_repeated_compactions_retrieve_originals_without_recursive_summary_drift(
    library,
):
    store, cid = library
    results = [
        build_context(
            store,
            cid,
            before,
            player_input="What is on the brass key?",
            context_limit=2200,
            output_reserve=400,
            safety_margin=100,
            token_counter=tokens,
        )
        for before in (25, 40, 61)
    ]
    assert all(r["trace"]["compacted"] for r in results)
    for result in results:
        memory = load_memory(store, result["memory_id"])
        assert memory["source_revisions"] == result["source_revisions"]
        clue = next(
            p for p in result["body"]["historical_source_passages"] if p["ordinal"] == 1
        )
        assert clue["quote"] == "The brass key bears the name Morrow."
    assert len(store.turns(cid)) == 60
    again = build_context(
        store,
        cid,
        61,
        player_input="What is on the brass key?",
        context_limit=2200,
        output_reserve=400,
        safety_margin=100,
        token_counter=tokens,
    )
    assert again["memory_id"] == results[-1]["memory_id"]
    with store.db() as db:
        assert db.execute("SELECT count(*) FROM context_memories").fetchone()[0] == 3
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("UPDATE context_memories SET payload='{}'")


def test_correction_invalidates_memory_and_rebuilds_from_changed_sources(library):
    store, cid = library
    eid = accept(store, cid, "fact", 1, "Morrow", entity="key", attribute="name")
    old = build_context(store, cid, 61, token_counter=tokens)
    assert context_is_current(store, cid, old)
    t = store.turns(cid)[0]
    store.revise(
        t["id"],
        text="The brass key bears the name Harker.",
        role="facilitator",
        reviewer="synthetic-test",
    )
    assert not context_is_current(store, cid, old)
    with pytest.raises(StaleContextError):
        load_memory(store, old["memory_id"])
    archived = load_memory(store, old["memory_id"], require_current=False)
    assert archived["state"]["facts"]["key:name"]["value"] == "Morrow"
    new = build_context(store, cid, 61, player_input="Harker key", token_counter=tokens)
    assert eid in load_memory(store, new["memory_id"])["state"]["stale_events"]
    assert new["body"]["state"]["facts"] == {}
    assert new["memory_id"] != old["memory_id"]
    assert "Harker" in packed(new["body"])
    assert "Morrow" not in new["body"]["dialogue"][0]["text"]


def test_review_decision_invalidates_without_source_edit(library):
    store, cid = library
    eid = accept(store, cid, "fact", 1, "Morrow", attribute="name")
    context = build_context(store, cid, 30, token_counter=tokens)
    store.review_event(cid, eid, "rejected", reviewer="synthetic-test")
    assert not context_is_current(store, cid, context)
    rebuilt = build_context(store, cid, 30, token_counter=tokens)
    assert rebuilt["body"]["state"]["facts"] == {}


def test_later_sources_and_events_do_not_invalidate_earlier_replay(library):
    store, cid = library
    old = build_context(store, cid, 25, token_counter=tokens)
    accept(store, cid, "fact", 59, "later clue", attribute="later")
    t = store.turns(cid)[55]
    store.revise(
        t["id"],
        text="A different future.",
        role="facilitator",
        reviewer="synthetic-test",
    )
    assert context_is_current(store, cid, old)


def test_inactive_fact_growth_archives_without_losing_resources_or_current_intent():
    state = empty_state()
    state["facts"] = {
        f"room{i}:clue": {
            "value": f"Closed case {i}: " + "faded evidence " * 30,
            "event": f"e{i}",
            "visibility": "public",
        }
        for i in range(500)
    }
    state["resources"]["Ada:HP"] = {"value": 9, "known_delta": -2, "event": "hp"}
    state["pending"]["roll"] = {"value": "Observation", "event": "r", "entity": "Ada"}
    result = compact(turns(8), state, player_input="I refuse to enter room 210.")
    assert result["trace"]["omitted_state"]
    assert result["body"]["state"]["resources"]["Ada:HP"]["value"] == 9
    assert result["body"]["state"]["pending"]["roll"]["value"] == "Observation"
    assert "refuse" in result["body"]["new_player_input"]
    assert len(state["facts"]) == 500


def test_budget_fails_instead_of_discarding_mandatory_input():
    with pytest.raises(ContextBudgetError) as raised:
        compact(turns(4), player_input="I explicitly attempt " * 3000)
    assert raised.value.trace["mandatory_tokens"] > raised.value.trace["prompt_budget"]


def test_native_tokenizer_counts_actual_chat_template_and_reserves_output():
    class Tokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return list(range(tokens(packed(messages)) + 13))

    tokenizer = Tokenizer()
    result = pack_context(
        turns=turns(4),
        state=empty_state(),
        tokenizer=tokenizer,
        context_limit=2000,
        output_reserve=300,
        safety_margin=200,
    )
    assert result["trace"]["token_counter"] == "native-chat-template"
    assert result["trace"]["prompt_budget"] == 1500
    assert tokenizer.calls[-1][1]["add_generation_prompt"]
    assert tokenizer.calls[-1][1]["return_dict"] is False
    fallback = pack_context(turns=turns(4), state=empty_state())
    assert fallback["trace"]["token_counter"] == "conservative-utf8-bytes"
    assert fallback["trace"]["prompt_tokens"] > result["trace"]["prompt_tokens"]


def test_retrieved_excerpts_are_exact_and_have_offsets():
    source = turns()
    source[0]["text"] = (
        "Dust. " * 250 + "The jade scarab opens the vault. " + "Silence. " * 200
    )
    result = compact(source, player_input="What does the jade scarab open?")
    excerpt = next(
        p for p in result["body"]["historical_source_passages"] if p["ordinal"] == 1
    )
    assert (
        excerpt["quote"]
        == source[0]["text"][excerpt["char_start"] : excerpt["char_end"]]
    )
    assert "jade scarab" in excerpt["quote"]
    assert excerpt["truncated"]


def test_source_mutation_during_packing_is_rejected(library):
    store, cid = library
    changed = False

    def racing_counter(text):
        nonlocal changed
        if not changed:
            changed = True
            t = store.turns(cid)[0]
            store.revise(
                t["id"],
                text="The key is iron.",
                role="facilitator",
                reviewer="synthetic-test",
            )
        return tokens(text)

    with pytest.raises(StaleContextError, match="changed"):
        build_context(store, cid, 61, token_counter=racing_counter)
    with store.db() as db:
        assert db.execute("SELECT count(*) FROM context_memories").fetchone()[0] == 0


def test_long_scenario_document_retrieves_exact_relevant_excerpt():
    document = {
        "id": "scenario-1",
        "visibility": "private",
        "text": "Dust settles. " * 300
        + "The cobalt bell awakens the drowned ferryman. "
        + "Silence falls. " * 300,
    }
    result = compact(
        turns(4), documents=[document], player_input="I ring the cobalt bell."
    )
    excerpt = result["body"]["documents"][0]
    assert "cobalt bell" in excerpt["text"] and excerpt["truncated"]
    assert (
        excerpt["text"] == document["text"][excerpt["char_start"] : excerpt["char_end"]]
    )
    assert result["trace"]["excerpted_documents"]


def test_unpunctuated_old_source_can_retrieve_detail_near_its_end():
    source = turns()
    source[0]["text"] = (
        "dust " * 700 + "cobalt bell awakens drowned ferryman " + "dust " * 500
    )
    result = compact(source, player_input="I ring the cobalt bell.")
    excerpt = next(
        p for p in result["body"]["historical_source_passages"] if p["ordinal"] == 1
    )
    assert "cobalt bell" in excerpt["quote"]


def test_active_character_knowledge_stays_owned_and_can_be_archived():
    state = empty_state()
    state["entities"]["Ada"] = {
        "occupation": {"value": "nurse", "visibility": "public"}
    }
    state["knowledge"]["Ada"] = {
        f"clue{i}": {"value": "forgotten scene " * 30, "event": f"e{i}"}
        for i in range(200)
    }
    state["knowledge"]["Mara"] = {
        "password": {"value": "ASPHODEL", "visibility": "private"}
    }
    result = compact(
        turns(8),
        state,
        active_entities=["Ada"],
        player_input="Does Mara know the password?",
    )
    body = result["body"]
    assert body["state"]["entities"]["Ada"]["occupation"]["value"] == "nurse"
    assert body["state"]["knowledge"]["Mara"]["password"]["value"] == "ASPHODEL"
    assert "password" not in body["state"]["knowledge"].get("Ada", {})
    assert result["trace"]["omitted_state"]


def test_current_scene_and_resolved_roll_survive_compaction():
    state = empty_state()
    state["facts"]["party:current_scene"] = {
        "value": "The flooded crypt",
        "event": "scene",
    }
    state["resolutions"] = [{"action": "roll1", "value": "failure", "event": "outcome"}]
    result = compact(turns(), state)
    assert (
        result["body"]["state"]["facts"]["party:current_scene"]["value"]
        == "The flooded crypt"
    )
    assert result["body"]["state"]["resolutions"][0]["value"] == "failure"


def test_unknown_roles_do_not_force_entire_transcript_into_current_exchange():
    source = turns(300)
    for t in source:
        t["role"] = "unknown"
    result = compact(source)
    assert result["trace"]["compacted"]
    assert result["body"]["dialogue"][-1]["text"] == source[-1]["text"]


def test_counterfactual_choices_retrieve_different_older_evidence():
    source = turns()
    source[0]["text"] = "The emerald door is sealed by an inscription."
    source[1]["text"] = "The obsidian drain runs beneath the prison."
    door = compact(source, player_input="I inspect the emerald door inscription.")
    drain = compact(
        source, player_input="I enter the obsidian drain beneath the prison."
    )
    assert door["trace"]["matched_source_turns"][0] == 1
    assert drain["trace"]["matched_source_turns"][0] == 2
    # Source exchanges remain chronological even when the matching utterance
    # is a reply; the previous observation supplies its interpretive context.
    assert [p["ordinal"] for p in drain["body"]["historical_source_passages"][:2]] == [
        1,
        2,
    ]
    assert door["body"]["new_player_input"] != drain["body"]["new_player_input"]


@pytest.mark.parametrize("role", ["unknown", "player", "facilitator"])
def test_hour_of_single_speaker_audio_is_compactable_at_utterance_boundaries(role):
    source = [
        {
            "id": f"speech-{i}",
            "ordinal": i,
            "revision": f"r{i}",
            "role": role,
            "speaker": "mic:0",
            "start": i * 12.0,
            "end": i * 12.0 + 4.0,
            "text": f"Utterance {i}: I describe the room and consider the various details without choosing an action.",
        }
        for i in range(1, 301)
    ]
    state = empty_state()
    state["pending"]["unresolved"] = {
        "entity": "Ada",
        "attribute": "check",
        "value": "Observation roll has not been reported",
    }
    result = compact(source, state, player_input=source[-1]["text"])
    assert result["trace"]["compacted"]
    assert result["body"]["dialogue"][-1]["text"] == source[-1]["text"]
    assert result["body"]["new_player_input"] == source[-1]["text"]
    assert "unresolved" in result["body"]["state"]["pending"]
    assert len(result["body"]["dialogue"]) < 30
    assert result["trace"]["prompt_tokens"] <= result["trace"]["prompt_budget"]


def test_single_speaker_unknown_clock_distinct_sources_are_not_one_mandatory_turn():
    source = [
        {
            "id": f"m{i}",
            "ordinal": i,
            "speaker": "unknown-mic",
            "role": "unknown",
            "text": "an incomplete ASR utterance " * 8,
        }
        for i in range(200)
    ]
    result = compact(source)
    assert result["trace"]["compacted"]
    assert result["body"]["dialogue"][-1]["text"] == source[-1]["text"]
    assert len(result["body"]["dialogue"]) < 30


def test_single_oversized_current_source_is_not_truncated_to_fix_monologue_budget():
    source = [
        {
            "id": "whole-current-input",
            "ordinal": 1,
            "speaker": "mic:0",
            "role": "unknown",
            "text": "A single actual current utterance " * 2000,
        }
    ]
    with pytest.raises(ContextBudgetError):
        compact(source)


def test_early_compaction_never_inflates_an_already_fitting_prompt_or_duplicates_source():
    source = turns(20)
    source[0]["text"] = "The brass key bears HARKER."
    # Trigger early compaction deliberately even when the complete input fits.
    result = pack_context(
        turns=source,
        state=empty_state(),
        player_input="What does the brass key say?",
        context_limit=4000,
        output_reserve=400,
        safety_margin=100,
        token_counter=tokens,
        compact_at=0.1,
    )
    trace = result["trace"]
    assert trace["uncompacted_prompt_tokens"] <= trace["prompt_budget"]
    assert trace["prompt_tokens"] <= trace["uncompacted_prompt_tokens"]
    retained = {t["id"] for t in result["body"]["dialogue"]}
    assert all(
        p["id"] not in retained for p in result["body"]["historical_source_passages"]
    )
