"""Agent boundaries and source continuity; scripted models do not measure prose quality."""

import json

import pytest
from starlette.testclient import TestClient

from story_copilot.campaigns import Campaigns
from story_copilot.copilot import make_copilot
from story_copilot.decisions import Decision, recall
from story_copilot.mechanics import numeric_sources, grounded_calculation
from story_copilot.profiles import validate_profile
from story_copilot.rule_advice import Calculation, RulesAnswer
from story_copilot.rules import search_rules
from story_copilot.schema import Extraction, NarrationAnswer
from story_copilot.response_review import ResponseReview
from story_copilot.settings import save, load
from story_copilot.store import Store, packed
from story_copilot.web import create_app

CONFIG = {
    "context_limit": 16384,
    "output_reserve": 1800,
    "safety_margin": 512,
    "token_counter": lambda text: (len(text.encode()) + 3) // 4,
}


@pytest.fixture
def table(tmp_path):
    store = Store(tmp_path / "workspace")
    campaigns = Campaigns(store)
    cid = campaigns.create("Original clockwork garden")
    sid = campaigns.create_session(cid, "Morning")
    char_id = campaigns.save_character(
        cid, "Vale", {"resources": {"charge": 3}, "expertise": ["repair"]}
    )
    campaigns.map_participant(
        cid, "Participant", "Player", role="player", character_id=char_id
    )
    campaigns.add_message(sid, "Player", "Can I power the gate?", role="player")
    return store, campaigns, cid, sid


class Models:
    def __init__(self, decisions, on_decision=None):
        self.decisions, self.calls, self.on_decision = iter(decisions), [], on_decision

    def __call__(self, task):
        owner = self

        class Client:
            def complete(self, messages, schema, **kwargs):
                body = json.loads(messages[-1]["content"])
                owner.calls.append((task, body, kwargs))
                if schema is Decision:
                    if owner.on_decision:
                        owner.on_decision()
                    answer = next(owner.decisions)
                elif schema is ResponseReview:
                    answer = ResponseReview(issues=[], revision=None)
                elif schema is Extraction:
                    answer = Extraction()
                elif schema is RulesAnswer:
                    answer = RulesAnswer(
                        answer="Use the supplied rule.",
                        citations=[
                            {
                                "id": body["rules"][0]["id"],
                                "quote": body["rules"][0]["text"],
                            }
                        ],
                        calculation=None,
                        missing_information=[],
                    )
                else:
                    answer = NarrationAnswer(
                        narration="What do you try next?",
                        private_notes="A private possibility.",
                    )
                return answer, {"model": "scripted", "seconds": 0.01}

        return Client()


def run(table, models):
    store, c, _, sid = table
    build, generate = make_copilot(store, model_factory=models, context_options=CONFIG)
    rid = c.generate(sid, generate, build_context=build, force=True)
    return next(r for r in c.runs(sid) if r["id"] == rid)


def test_default_agent_chooses_tools_then_preserves_private_guidance(table):
    store, c, cid, sid = table
    c.add_document(
        cid,
        "Power rule",
        "Powering the gate costs one charge.",
        metadata={"kind": "rules"},
    )
    models = Models(
        [
            Decision(action="character", character="Vale"),
            Decision(action="rules", query="power gate charge"),
            Decision(action="respond"),
        ]
    )
    result = run(table, models)
    trace = result["result"]["trace"]
    assert trace["decision"]["status"] == "ready"
    assert [x[0] for x in models.calls] == [
        "classifier",
        "auditor",
        "auditor",
        "auditor",
        "rules",
        "storyteller",
        "auditor",
    ]
    assert (
        trace["decision"]["steps"][0]["observation"]["result"]["current_resources"][
            "Vale:charge"
        ]["value"]
        == 3
    )
    narrator = next(body for task, body, _ in models.calls if task == "storyteller")
    assert any(d["id"] == "evidence-investigation" for d in narrator["documents"])
    assert len(c.messages(sid)) == 1
    assert c.state(sid)["resources"]["Vale:charge"]["value"] == 3
    assert all(p["visibility"] == "private" for p in c.proposals(sid))
    assert "A private possibility" not in packed(c.snapshot(sid, public_only=True))


def test_clarification_is_terminal_without_a_second_narration_call(table):
    _, c, _, sid = table
    result = run(
        table,
        Models(
            [
                Decision(
                    action="clarify", question="Which character is operating the gate?"
                )
            ]
        ),
    )
    trace = result["result"]["trace"]
    assert trace["storyteller"] == {"status": "complete", "clarification_only": True}
    build, generate = make_copilot(
        table[0], model_factory=Models([]), context_options=CONFIG
    )
    assert not build(c.snapshot(sid))["narration_needed"]
    assert len(c.messages(sid)) == 1


def test_source_correction_during_decision_discards_the_entire_reply(table):
    _, c, _, sid = table
    message = c.messages(sid)[0]

    def change():
        c.revise_message(
            sid,
            message["id"],
            text="I leave the gate unpowered.",
            speaker="Player",
            role="player",
            character="Vale",
            visibility="public",
            expected_revision=message["revision"],
        )

    models = Models([Decision(action="respond")], on_decision=change)
    result = run(table, models)
    assert result["status"] == "stale" and c.proposals(sid) == []
    assert not any(task == "storyteller" for task, _, _ in models.calls)


def test_rule_profile_change_invalidates_in_flight_decision(table):
    _, c, cid, sid = table
    models = Models(
        [Decision(action="respond")],
        on_decision=lambda: c.set_rule_profile(
            cid, {"resource_aliases": {"battery": "charge"}}
        ),
    )
    result = run(table, models)
    assert result["status"] == "stale"


def test_repeated_tool_calls_are_bounded_and_visible(table):
    models = Models([Decision(action="recall", query="gate") for _ in range(4)])
    result = run(table, models)
    trace = result["result"]["trace"]["decision"]
    assert trace["status"] == "step_budget" and len(trace["steps"]) == 4
    assert "already has a result" in trace["steps"][1]["error"]
    assert "uncertainties" in packed(
        next(body for task, body, _ in models.calls if task == "storyteller")
    )
    assert all(
        0 < kwargs["timeout"] <= 75
        for task, _, kwargs in models.calls
        if task == "auditor"
    )


def test_rules_are_campaign_scoped_and_retrieved_from_frozen_version(table):
    store, c, cid, sid = table
    own = c.add_document(
        cid, "A", "Gate power costs one charge.", metadata={"kind": "rules"}
    )
    c.add_document(cid, "Secret notes", "Gate power costs 999 charges.")
    foreign = c.create("Unrelated setting")
    c.add_document(
        foreign, "B", "Gate power costs 888 charges.", metadata={"kind": "rules"}
    )
    frozen = c.snapshot(sid)
    sources = search_rules(store, "gate power", snapshot=frozen)
    assert len(sources) == 1 and sources[0]["document_id"] == own
    c.delete_document(cid, own)
    assert search_rules(store, "gate power", snapshot=frozen) == sources
    assert not search_rules(store, "gate power", snapshot=c.snapshot(sid))


def test_recall_keeps_the_player_initiation_reply_and_correction(table):
    _, c, _, sid = table
    c.add_message(
        sid, "Facilitator", "The gate opens toward the greenhouse.", role="facilitator"
    )
    c.add_message(
        sid,
        "Player",
        "Correction: I wanted to examine the gate, not power it.",
        role="player",
    )
    sources = recall(c.snapshot(sid), "greenhouse")
    assert len(sources) == 3
    assert sources[0]["text"] == "Can I power the gate?"
    assert sources[-1]["text"].startswith("Correction:")
    assert all(m.get("revision") for m in sources)


def test_numeric_slots_use_current_resources_and_explicit_actor_only(table):
    _, c, _, sid = table
    state = c.state(sid)
    state["resources"]["Vale:charge"]["value"] = 1
    messages = [
        {
            "id": "m1",
            "text": "I rolled 24. Rating: 60. Cost 1.5.",
            "role": "player",
            "character": "Vale",
        }
    ]
    sources = numeric_sources(messages, state)
    assert {s["value"] for s in sources} >= {24, 60, 1.5, 1}
    assert not any(s["value"] == 3 for s in sources)
    assert not any("character_id" in s["id"] for s in sources)
    messages[0]["character"] = ""
    assert not any(
        s["id"].startswith("resource:") for s in numeric_sources(messages, state)
    )


def test_numeric_slot_cannot_be_forged():
    profile = {
        "tools": [
            {
                "name": "enough",
                "operation": "greater_equal",
                "description": "Compare available with cost.",
            }
        ]
    }
    call = Calculation(
        tool="enough",
        inputs=[
            {"source": "resource:Vale:charge", "value": 9},
            {"source": "rule:cost", "value": 1},
        ],
    )
    sources = [
        {"id": "resource:Vale:charge", "value": 1},
        {"id": "rule:cost", "value": 1},
    ]
    with pytest.raises(ValueError, match="explicitly supplied"):
        grounded_calculation(call, profile, sources)


def test_model_settings_do_not_accept_keys_in_urls_or_unknown_backends(table):
    store, *_ = table
    for change in [
        {"url": "http://user:secret@localhost/v1"},
        {"url": "http://localhost/v1?key=secret"},
        {"api_key_env": "not an env var"},
        {"backend": "unsupported"},
    ]:
        with pytest.raises(ValueError):
            save(store.home, change)
    save(store.home, {"model": "chosen-model"})
    assert load(store.home)["model"] == "chosen-model"
    assert (store.home / "model-settings.json").stat().st_mode & 0o777 == 0o600


def test_new_app_refuses_to_modify_an_unrecognized_database(tmp_path):
    (tmp_path / "library.sqlite").write_bytes(b"existing database")
    with pytest.raises(ValueError, match="unrecognized"):
        Store(tmp_path)
    assert (tmp_path / "library.sqlite").read_bytes() == b"existing database"


def test_requested_checks_require_real_supplied_rule_quotes():
    from story_copilot.copilot import validate_checks
    from story_copilot.schema import CheckSuggestion

    sources = [
        {
            "id": "supplied",
            "text": "Crossing an unstable bridge requires a balance check.",
        }
    ]
    checks = [
        CheckSuggestion(
            text="Balance check for crossing.",
            rule_id="supplied",
            quote="requires a balance check",
        ),
        CheckSuggestion(
            text="Invented rule.", rule_id="remembered", quote="Make a check."
        ),
        CheckSuggestion(
            text="Invented quotation.",
            rule_id="supplied",
            quote="Roll a perception check.",
        ),
    ]
    accepted, rejected = validate_checks(checks, sources)
    assert accepted == ["Balance check for crossing."]
    assert len(rejected) == 2
    assert rejected[1]["check"]["text"] == "Invented quotation."


def test_withheld_check_does_not_repeat_an_otherwise_delivered_reply(table):
    from story_copilot.schema import CheckSuggestion

    class InvalidCheckModels(Models):
        def __call__(self, task):
            if task != "storyteller":
                return super().__call__(task)

            class Writer:
                def complete(self, *args, **kwargs):
                    return NarrationAnswer(
                        narration="The choice is yours.",
                        requested_checks=[
                            CheckSuggestion(
                                text="An invented check.",
                                rule_id="absent",
                                quote="Some imagined rule.",
                            )
                        ],
                    ), {}

            return Writer()

    _, c, _, sid = table
    result = run(table, InvalidCheckModels([Decision(action="respond")]))
    assert result["result"]["trace"]["storyteller"]["status"] == "partial"
    assert not any(p["kind"] == "action" for p in c.proposals(sid))
    build, _ = make_copilot(table[0], model_factory=Models([]), context_options=CONFIG)
    assert not build(c.snapshot(sid))["narration_needed"]


def test_numeric_citation_is_verified_without_treating_a_resource_as_a_rule():
    from story_copilot.rule_advice import validate_advice

    sources = [
        {"id": "rule1", "title": "Example rule", "text": "The device costs 1 charge."}
    ]
    numbers = numeric_sources([], {}, sources)
    answer = RulesAnswer(
        answer="Use the stated cost.",
        citations=[{"id": numbers[0]["id"], "quote": "costs 1 charge"}],
        calculation=None,
        missing_information=[],
    )
    assert validate_advice(answer, sources, {}, numbers) is None
    numbers.append({"id": "resource:charge", "value": 3, "quote": "3"})
    answer.citations = [type(answer.citations[0])(id="resource:charge", quote="3")]
    with pytest.raises(ValueError, match="rule citation"):
        validate_advice(answer, sources, {}, numbers)


def test_consistent_total_and_delta_apply_once_but_ambiguous_cases_are_rejected():
    from story_copilot.copilot import _validate_event
    from story_copilot.schema import EventCandidate

    message = {
        "id": "original",
        "ordinal": 1,
        "revision": "original",
        "role": "player",
        "text": "I spend 2 supplies; I have 6 left.",
        "visibility": "public",
    }
    state = {"resources": {"Tess:supplies": {"value": 8}}}
    event = EventCandidate(
        kind="resource",
        entity="Tess",
        attribute="supplies",
        value=6,
        delta=-2,
        evidence=[{"turn": 1, "quote": message["text"]}],
    )
    parsed, evidence = _validate_event(event, [message], state)
    assert (
        parsed.value == 6
        and parsed.delta is None
        and evidence[0]["quote"] == message["text"]
    )
    for prior in [None, 7, 9]:
        with pytest.raises(ValueError, match="total or a delta"):
            _validate_event(
                event, [message], {"resources": {"Tess:supplies": {"value": prior}}}
            )
    event.value, event.delta = 5, -3
    with pytest.raises(ValueError, match="total or a delta"):
        _validate_event(event, [message], state)
