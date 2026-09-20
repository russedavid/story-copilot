"""Real orchestration with fake model backends; these do not measure model quality."""

import json

import pytest

from story_copilot.campaigns import Campaigns
from story_copilot.copilot import make_copilot as _make_copilot


def make_copilot(*args, **kwargs):
    # Retain regression coverage of the original deterministic workflow.
    return _make_copilot(*args, policy="workflow", quality_review=False, **kwargs)


from story_copilot.rule_advice import RulesAnswer, RuleSearchPlan
from story_copilot.schema import EventCandidate, Extraction, NarrationAnswer
from story_copilot.store import Store, packed

CONFIG = {
    "context_limit": 16384,
    "output_reserve": 1800,
    "safety_margin": 512,
    "token_counter": lambda text: (len(text.encode()) + 3) // 4,
}


def calculation(body):
    # Nonexistent sources stay explicit so fabrication tests exercise validation.
    def ref(value):
        return next(
            (s["id"] for s in body["numeric_sources"] if s["value"] == value),
            "invented",
        )

    return {
        "tool": "within_limit",
        "inputs": [{"source": ref(24), "value": 24}, {"source": ref(60), "value": 60}],
    }


class Models:
    def __init__(self, classifier=None, storyteller=None, rules=None):
        self.handlers = {
            "auditor": lambda body: RuleSearchPlan(queries=["skill check"]),
            "classifier": classifier or (lambda body: Extraction()),
            "storyteller": storyteller
            or (
                lambda body: NarrationAnswer(
                    narration="The choice remains yours.",
                    questions=["What do you try?"],
                    private_notes="Keep the hidden chamber concealed.",
                )
            ),
            "rules": rules
            or (
                lambda body: RulesAnswer(
                    answer="Please provide the missing inputs.",
                    citations=[],
                    calculation=None,
                    missing_information=["skill, roll and difficulty"],
                )
            ),
        }
        self.calls = []

    def __call__(self, task):
        owner = self

        class Client:
            def complete(self, messages, schema, **kwargs):
                try:
                    body = json.loads(messages[-1]["content"])
                except ValueError:
                    body = {"question": messages[-1]["content"]}
                owner.calls.append(
                    {"task": task, "body": body, "schema": schema, "settings": kwargs}
                )
                return owner.handlers[task](body), {
                    "model": "fake-" + task,
                    "seconds": 0.01,
                    "finish_reason": "stop",
                }

        return Client()


@pytest.fixture
def table(tmp_path):
    store = Store(tmp_path / "private")
    c = Campaigns(store)
    cid = c.create(
        "Original synthetic station", direction="The Facilitator chooses the pace."
    )
    sid = c.create_session(cid, "First evening")
    c.set_rule_profile(
        cid,
        {
            "resource_aliases": {"hit_points": "HP", "hitpoint": "HP"},
            "tools": [
                {
                    "name": "within_limit",
                    "operation": "less_equal",
                    "description": "Compare the stated check value with the stated limit.",
                }
            ],
        },
    )
    c.add_document(
        cid,
        "Original test rule",
        "For this invented skill check, the stated result must be at most the stated limit.",
        metadata={"kind": "rules"},
    )
    c.save_character(
        cid, "Ada", {"resources": {"HP": 12}, "skills": {"Observation": 60}}
    )
    c.add_document(
        cid,
        "Hidden chamber",
        "The chamber contains a brass telescope.",
        visibility="private",
    )
    return store, c, cid, sid


def event(kind, turn, quote, **kwargs):
    return EventCandidate(
        kind=kind,
        entity=kwargs.pop("entity", "Ada"),
        attribute=kwargs.pop("attribute", "HP"),
        value=kwargs.pop("value", None),
        evidence=[{"turn": turn, "quote": quote}],
        **kwargs,
    )


def run(table, models, *, force=False, classify_limit=12):
    store, c, _, sid = table
    build, generate = make_copilot(
        store,
        model_factory=models,
        context_options=CONFIG,
        classify_limit=classify_limit,
    )
    rid = c.generate(sid, generate, build_context=build, force=force)
    return next((r for r in c.runs(sid) if r["id"] == rid), None)


def test_callbacks_are_lazy_and_plain_turns_skip_rules_planner(table):
    store, c, _, sid = table
    models = Models()
    build, generate = make_copilot(store, model_factory=models, context_options=CONFIG)
    assert not models.calls
    mid = c.add_message(sid, "Ada", "I look through the window.", role="player")
    context = build(c.snapshot(sid))
    assert not models.calls
    result = generate(context)
    assert [call["task"] for call in models.calls] == ["classifier", "storyteller"]
    assert all(s["visibility"] == "private" for s in result["suggestions"])
    assert all(not s["payload"]["state_changes"] for s in result["suggestions"])
    assert context["source_revisions"][mid] == mid
    body = models.calls[-1]["body"]
    assert "Ada (player): I look through the window." == body["new_player_input"]
    assert body["state"]["resources"]["Ada:HP"]["value"] == 12
    assert "brass telescope" in packed(body["documents"])
    assert result["trace"]["guidance_updates_state"] is False


def test_source_observations_update_state_without_acceptance_and_keep_exact_evidence(
    table,
):
    _, c, _, sid = table
    mid = c.add_message(
        sid,
        "Facilitator",
        "Ada loses two hit points from the falling glass.",
        role="facilitator",
    )
    c.add_message(sid, "Ada", "I get away from the window.", role="player")
    models = Models(
        classifier=lambda body: Extraction(
            events=[event("resource", 1, "Ada loses two hit points", delta=-2)]
        )
    )
    result = run(table, models)
    assert result["status"] == "complete"
    proposal = next(p for p in c.proposals(sid) if p["kind"] == "state")
    assert proposal["evidence"] == [
        {"message_id": mid, "revision": mid, "quote": "Ada loses two hit points"}
    ]
    assert proposal["payload"]["source_event_sha256"]
    assert proposal["observed"]
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 10
    assert models.calls[-1]["body"]["state"]["resources"]["Ada:HP"]["value"] == 10
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 10
    assert len(c.messages(sid)) == 2
    assert c.proposals(sid, public_only=True) == []


def test_speaking_only_part_of_a_suggestion_records_only_that_part(table):
    _, c, _, sid = table
    c.add_message(sid, "Ada", "I inspect the window.", role="player")
    models = Models(
        storyteller=lambda body: NarrationAnswer(
            narration="Ada loses five hit points and drops her torch."
        )
    )
    run(table, models)
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 12
    assert len(c.messages(sid)) == 1
    c.add_message(
        sid,
        "Facilitator",
        "Ada loses one hit point, but keeps the torch.",
        role="facilitator",
    )
    models.handlers["classifier"] = lambda body: Extraction(
        events=[event("resource", 2, "Ada loses one hit point", delta=-1)]
    )
    run(table, models)
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 11
    assert len(c.messages(sid)) == 2
    assert not c.publications(sid)


def test_continuing_after_spoken_correction_does_not_reapply_superseded_loss(table):
    store, c, cid, sid = table
    c.add_message(sid, "Facilitator", "Ada loses two hit points.", role="facilitator")
    models = Models(
        classifier=lambda body: Extraction(
            events=[event("resource", 1, "Ada loses two hit points", delta=-2)]
        )
    )
    run(table, models)
    prior = c.state(sid)["resources"]["Ada:HP"]["id"]
    c.add_message(
        sid, "Facilitator", "Correction: Ada lost one hit point.", role="facilitator"
    )
    models.handlers["classifier"] = lambda body: Extraction(
        events=[
            event("resource", 2, "Ada lost one hit point", delta=-1, supersedes=prior)
        ]
    )
    run(table, models)
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 11
    branch = c.continue_session(cid, "Next evening", parent_id=sid)
    assert c.state(branch)["resources"]["Ada:HP"]["value"] == 11
    build, generate = make_copilot(store, model_factory=models, context_options=CONFIG)
    context = build(c.snapshot(branch))
    assert context["target_messages"] == []
    c.generate(branch, generate, build_context=build)
    assert c.state(branch)["resources"]["Ada:HP"]["value"] == 11


def test_invalid_classifier_fragment_does_not_discard_valid_fact_or_draft(table):
    _, c, _, sid = table
    c.add_message(
        sid,
        "Facilitator",
        "The door is red. Ada loses two hit points.",
        role="facilitator",
    )
    c.add_message(sid, "Ada", "I step back.", role="player")
    models = Models(
        classifier=lambda body: Extraction(
            events=[
                event("resource", 1, "Ada loses two hit points", value=2, delta=-2),
                event(
                    "fact",
                    1,
                    "The door is red",
                    entity="door",
                    attribute="color",
                    value="red",
                ),
            ]
        )
    )
    result = run(table, models)
    trace = result["result"]["trace"]
    assert trace["classification"]["status"] == "partial"
    assert len(trace["classification"]["rejected_fragments"]) == 1
    assert trace["storyteller"]["status"] == "complete"
    assert any(p["kind"] == "state" for p in c.proposals(sid))
    assert any(p["kind"] == "narration" for p in c.proposals(sid))


def test_unknown_speaker_does_not_acquire_an_invented_character(table):
    _, c, _, sid = table
    c.add_message(sid, "system:SPEAKER_03", "I lose two hit points.", role="unknown")
    models = Models(
        classifier=lambda body: Extraction(
            events=[event("resource", 1, "I lose two hit points", delta=-2)]
        )
    )
    result = run(table, models)
    trace = result["result"]["trace"]["classification"]
    assert "unknown speaker" in trace["rejected_fragments"][0]["error"].lower()
    assert not any(p["kind"] == "state" for p in c.proposals(sid))
    assert models.calls[-1]["body"]["dialogue"][-1]["role"] == "unknown"


def test_resource_synonym_uses_existing_sheet_total_and_records_normalization(table):
    _, c, _, sid = table
    c.add_message(sid, "Facilitator", "Ada loses two hit points.", role="facilitator")
    models = Models(
        classifier=lambda body: Extraction(
            events=[
                event(
                    "resource",
                    1,
                    "Ada loses two hit points",
                    attribute="hit_points",
                    delta=-2,
                )
            ]
        )
    )
    result = run(table, models)
    proposal = next(p for p in c.proposals(sid) if p["kind"] == "state")
    c.decide(sid, proposal["id"], "accepted")
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 10
    assert "Ada:hit_points" not in c.state(sid)["resources"]
    assert (
        result["result"]["trace"]["classification"]["resource_name_normalizations"][0][
            "from"
        ]
        == "hit_points"
    )


def test_ambiguous_resource_synonyms_do_not_choose_a_sheet_total(table):
    _, c, cid, sid = table
    character = c.characters(cid)[0]
    c.save_character(
        cid,
        "Ada",
        {"resources": {"HP": 12, "hit_points": 8}},
        character_id=character["id"],
    )
    c.add_message(sid, "Facilitator", "Ada loses two hit points.", role="facilitator")
    models = Models(
        classifier=lambda body: Extraction(
            events=[
                event(
                    "resource",
                    1,
                    "Ada loses two hit points",
                    attribute="hit-points",
                    delta=-2,
                )
            ]
        )
    )
    result = run(table, models)
    assert (
        "Multiple sheet resources"
        in result["result"]["trace"]["classification"]["rejected_fragments"][0]["error"]
    )
    assert not any(p["kind"] == "state" for p in c.proposals(sid))


def test_processed_sources_not_reproposed_on_later_turns_or_forced_refresh(table):
    _, c, _, sid = table
    mid = c.add_message(
        sid, "Facilitator", "Ada loses two hit points.", role="facilitator"
    )

    def classify(body):
        if body["target_turns"][0]["ordinal"] == 1:
            return Extraction(
                events=[event("resource", 1, "Ada loses two hit points", delta=-2)]
            )
        return Extraction()

    models = Models(classifier=classify)
    run(table, models)
    p = next(p for p in c.proposals(sid) if p["kind"] == "state")
    c.decide(sid, p["id"], "accepted")
    c.add_message(sid, "Ada", "I search elsewhere.", role="player")
    later = run(table, models)
    assert [t["id"] for t in later["request"]["target_messages"]] != [mid]
    forced = run(table, models, force=True)
    assert forced["result"]["trace"]["classification"]["status"] == "up_to_date"
    assert len([p for p in c.proposals(sid) if p["kind"] == "state"]) == 1
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 10


def test_pending_hypothesis_and_resolution_mapping(table):
    _, c, _, sid = table
    c.add_message(
        sid, "Ada", "Maybe I could try the lock, but I do not act yet.", role="player"
    )
    models = Models(
        classifier=lambda body: Extraction(
            events=[
                event(
                    "action",
                    1,
                    "Maybe I could try the lock",
                    attribute="open",
                    value="lock",
                    stage="hypothetical",
                )
            ]
        )
    )
    run(table, models)
    p = next(p for p in c.proposals(sid) if p["kind"] == "state")
    assert not c.state(sid)["pending"]
    assert c.state(sid)["claims"][0]["stage"] == "hypothetical"
    c.add_message(
        sid, "Facilitator", "Ada, make a regular Observation check.", role="facilitator"
    )
    models.handlers["classifier"] = lambda body: Extraction(
        events=[
            event(
                "action",
                2,
                "Ada, make a regular Observation check",
                attribute="check",
                value="Observation",
                stage="requested",
            )
        ]
    )
    run(table, models)
    p = [p for p in c.proposals(sid) if p["kind"] == "state"][-1]
    action_id = next(iter(c.state(sid)["pending"]))
    c.add_message(
        sid, "Facilitator", "Ada succeeds and finds the key.", role="facilitator"
    )
    models.handlers["classifier"] = lambda body: Extraction(
        events=[
            event(
                "resolve",
                3,
                "Ada succeeds and finds the key",
                attribute="check",
                value="success",
                resolves=action_id,
            )
        ]
    )
    run(table, models)
    p = [p for p in c.proposals(sid) if p["kind"] == "state"][-1]
    assert p["payload"]["state_changes"][0]["action_id"] == action_id
    assert not c.state(sid)["pending"] and c.state(sid)["resolutions"]


def test_unaccepted_resolution_is_rejected_without_stalling_narration(table):
    _, c, _, sid = table
    c.add_message(sid, "Facilitator", "Ada finds the key.", role="facilitator")
    models = Models(
        classifier=lambda body: Extraction(
            events=[
                event(
                    "resolve",
                    1,
                    "Ada finds the key",
                    attribute="check",
                    value="success",
                    resolves="unaccepted-action",
                )
            ]
        )
    )
    result = run(table, models)
    assert result["result"]["trace"]["classification"]["rejected_fragments"]
    assert result["result"]["trace"]["storyteller"]["status"] == "complete"


def test_private_evidence_cannot_be_declassified_by_model(table):
    _, c, _, sid = table
    c.add_message(
        sid,
        "Facilitator",
        "The sealed chamber contains a telescope.",
        role="facilitator",
        visibility="private",
    )
    models = Models(
        classifier=lambda body: Extraction(
            events=[
                event(
                    "fact",
                    1,
                    "chamber contains a telescope",
                    entity="chamber",
                    attribute="contents",
                    value="telescope",
                    visibility="public",
                )
            ]
        )
    )
    run(table, models)
    p = next(p for p in c.proposals(sid) if p["kind"] == "state")
    assert p["payload"]["source_event"]["visibility"] == "private"
    assert p["payload"]["state_changes"][0]["visibility"] == "private"


def test_rules_use_real_task_adapter_grounded_tools_and_feed_bounded_narrator(table):
    _, c, _, sid = table
    c.add_message(
        sid,
        "Facilitator",
        "Make a regular Observation check at 60.",
        role="facilitator",
    )
    c.add_message(sid, "Ada", "I rolled 24. What is the result?", role="player")
    models = Models(
        rules=lambda body: RulesAnswer(
            answer="Use the deterministic check result.",
            citations=[
                {
                    "id": body["rules"][0]["id"],
                    "quote": "the stated result must be at most the stated limit",
                }
            ],
            calculation=calculation(body),
            missing_information=[],
        )
    )
    result = run(table, models)
    assert [x["task"] for x in models.calls] == [
        "classifier",
        "auditor",
        "rules",
        "storyteller",
    ]
    assert (
        result["result"]["trace"]["rules"]["search_trace"]["planner"]["model"]
        == "fake-auditor"
    )
    assert "brass telescope" in packed(
        next(x["body"] for x in models.calls if x["task"] == "rules")[
            "scenario_context"
        ]
    )
    rule = next(p for p in c.proposals(sid) if p["kind"] == "rule")
    assert rule["payload"]["calculated_result"]["result"] is True
    assert not rule["payload"]["state_changes"]
    narrator = models.calls[-1]["body"]
    doc = next(d for d in narrator["documents"] if d["id"] == "validated-rule-advice")
    assert json.loads(doc["text"])["deterministic_calculation"]["inputs"] == [24, 60]
    trace = result["result"]["trace"]["narrator_context"]
    assert trace["prompt_tokens"] <= trace["prompt_budget"]


def test_fabricated_calculation_and_rule_quote_are_retained_as_failures(table):
    _, c, _, sid = table
    c.add_message(
        sid, "Ada", "What does a regular check mean? I have not rolled.", role="player"
    )
    models = Models(
        rules=lambda body: RulesAnswer(
            answer="A result",
            citations=[
                {
                    "id": body["rules"][0]["id"],
                    "quote": "the stated result must be at most the stated limit",
                }
            ],
            calculation=calculation(body),
            missing_information=[],
        )
    )
    result = run(table, models)
    assert result["result"]["trace"]["rules"]["status"] == "failed"
    assert not any(p["kind"] == "rule" for p in c.proposals(sid))
    models.handlers["rules"] = lambda body: RulesAnswer(
        answer="An unsupported rule",
        citations=[{"id": "invented", "quote": "rule"}],
        calculation=None,
        missing_information=[],
    )
    result = run(table, models, force=True)
    assert "not supplied" in result["result"]["trace"]["rules"]["error"]
    assert result["result"]["trace"]["storyteller"]["status"] == "complete"


def test_source_change_cancels_run_and_does_not_mark_classification_processed(table):
    store, c, _, sid = table
    mid = c.add_message(sid, "Ada", "I inspect the door.", role="player")

    def classify(body):
        c.revise_message(
            sid,
            mid,
            text="I inspect the window instead.",
            speaker="Ada",
            role="player",
            visibility="public",
            expected_revision=mid,
        )
        return Extraction()

    models = Models(classifier=classify)
    result = run(table, models)
    assert result["status"] == "stale"
    assert c.proposals(sid) == []
    assert [x["task"] for x in models.calls] == ["classifier"]
    build, _ = make_copilot(store, model_factory=Models(), context_options=CONFIG)
    assert build(c.snapshot(sid))["target_messages"][0]["id"] == mid


def test_old_snapshot_skips_all_models_and_reports_stale(table):
    store, c, _, sid = table
    c.add_message(sid, "Ada", "I inspect the door.", role="player")
    models = Models()
    build, generate = make_copilot(store, model_factory=models, context_options=CONFIG)
    context = build(c.snapshot(sid))
    c.add_message(sid, "Ada", "Actually I wait outside.", role="player")
    result = generate(context)
    assert result["trace"]["stale"] and not models.calls


def test_backlog_is_explicit_and_no_source_text_is_truncated(table):
    _, c, _, sid = table
    mids = [
        c.add_message(sid, "Ada", f"I inspect numbered hatch {i}.", role="player")
        for i in range(5)
    ]
    result = run(table, Models(), classify_limit=2)
    trace = result["result"]["trace"]["classification"]
    assert set(trace["processed_source_revisions"]) == set(mids[:2])
    assert trace["backlog"] == mids[2:]
    assert (
        result["request"]["target_messages"][0]["text"] == "I inspect numbered hatch 0."
    )


def test_source_revision_reprocesses_without_reapplying_the_old_effect(table):
    _, c, _, sid = table
    mid = c.add_message(
        sid, "Facilitator", "Ada loses two hit points.", role="facilitator"
    )
    models = Models(
        classifier=lambda body: Extraction(
            events=[
                event(
                    "resource",
                    1,
                    body["target_turns"][0]["text"],
                    delta=-2 if "two" in body["target_turns"][0]["text"] else -1,
                )
            ]
        )
    )
    run(table, models)
    p = next(p for p in c.proposals(sid) if p["kind"] == "state")
    c.decide(sid, p["id"], "accepted")
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 10
    c.revise_message(
        sid,
        mid,
        text="Ada loses one hit point.",
        speaker="Facilitator",
        role="facilitator",
        visibility="public",
        expected_revision=mid,
    )
    result = run(table, models)
    assert result["request"]["target_messages"][0]["id"] == mid
    newer = [p for p in c.proposals(sid) if p["kind"] == "state"][-1]
    c.decide(sid, newer["id"], "accepted")
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 11


def test_feedback_is_private_draft_feedback_not_source_speech(table):
    _, c, _, sid = table
    c.add_message(sid, "Ada", "I examine the latch.", role="player")
    models = Models()
    run(table, models)
    p = next(p for p in c.proposals(sid) if p["kind"] == "narration")
    c.decide(sid, p["id"], "rejected")
    c.add_message(sid, "Ada", "I want a quieter approach.", role="player")
    run(table, models)
    body = models.calls[-1]["body"]
    assert "rejected" in packed(body["documents"])
    assert "Selection does not mean" in body["private_facilitator_direction"]
    assert all(m["text"] != p["text"] for m in body["dialogue"])


def test_all_backend_failures_remain_retryable_failed_runs(table):
    _, c, _, sid = table
    c.add_message(sid, "Ada", "I inspect the hatch.", role="player")

    def fail(body):
        raise RuntimeError("fake backend offline")

    result = run(table, Models(classifier=fail, storyteller=fail))
    assert result["status"] == "failed"
    assert result["result"]["trace"]["storyteller"]["status"] == "failed"
    retry = run(table, Models())
    assert retry["status"] == "complete"


def test_branch_replay_does_not_double_apply_inherited_source_events(table):
    store, c, cid, sid = table
    c.add_message(sid, "Facilitator", "Ada loses two hit points.", role="facilitator")
    models = Models(
        classifier=lambda body: Extraction(
            events=[event("resource", 1, "Ada loses two hit points", delta=-2)]
        )
    )
    run(table, models)
    p = next(p for p in c.proposals(sid) if p["kind"] == "state")
    branch = c.branch(sid, "Continue tomorrow")
    continued = run((store, c, cid, branch), models)
    assert continued["result"]["trace"]["classification"]["status"] == "up_to_date"
    assert len([p for p in c.proposals(branch) if p["kind"] == "state"]) == 1
    assert c.state(branch)["resources"]["Ada:HP"]["value"] == 10
    second = c.branch(branch, "Another continuation")
    continued = run((store, c, cid, second), models)
    assert continued["result"]["trace"]["classification"]["status"] == "up_to_date"
    assert c.state(second)["resources"]["Ada:HP"]["value"] == 10


def test_overflow_fragments_remain_backlogged_then_dedupe_on_retry(table):
    _, c, _, sid = table
    mid = c.add_message(
        sid,
        "Facilitator",
        "The register records many numbered stations.",
        role="facilitator",
    )
    models = Models(
        classifier=lambda body: Extraction(
            events=[
                event(
                    "fact",
                    1,
                    "numbered stations",
                    entity="register",
                    attribute=f"entry{i}",
                    value=f"station {i}",
                )
                for i in range(14)
            ]
        )
    )
    first = run(table, models)
    trace = first["result"]["trace"]["classification"]
    assert trace["status"] == "partial"
    assert len(trace["deferred_fragments"]) == 2
    assert trace["processed_source_revisions"][mid] == mid
    assert trace["cached_proposals_remaining"] == 2
    second = run(table, models, force=True)
    trace = second["result"]["trace"]["classification"]
    assert len(trace["reused_cache"]) == 2
    assert sum(call["task"] == "classifier" for call in models.calls) == 1
    assert len([p for p in c.proposals(sid) if p["kind"] == "state"]) == 14


def test_spoken_correction_replaces_prior_resource_effect_without_approval(
    table,
):
    _, c, _, sid = table
    old = c.add_proposal(
        sid,
        {
            "kind": "state",
            "title": "Observed loss",
            "text": "Ada lost two HP.",
            "visibility": "private",
            "payload": {
                "state_changes": [
                    {
                        "kind": "resource",
                        "entity": "Ada",
                        "attribute": "HP",
                        "delta": -2,
                    }
                ]
            },
        },
    )
    c.decide(sid, old, "accepted")
    c.add_message(
        sid,
        "Facilitator",
        "Correction: Ada lost one hit point, not two.",
        role="facilitator",
    )
    models = Models(
        classifier=lambda body: Extraction(
            events=[
                event(
                    "resource",
                    1,
                    "Ada lost one hit point",
                    delta=-1,
                    supersedes=old + ":0",
                )
            ]
        )
    )
    result = run(table, models)
    assert result["status"] == "complete"
    proposal = next(p for p in c.proposals(sid) if p.get("observed"))
    assert proposal["payload"]["state_changes"][0]["supersedes"] == old + ":0"
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 11


def test_rule_failure_retains_raw_model_response_and_model_metadata(table):
    _, c, _, sid = table
    c.add_message(sid, "Ada", "What does a regular check mean?", role="player")
    models = Models(
        rules=lambda body: RulesAnswer(
            answer="Invented calculation",
            citations=[
                {
                    "id": body["rules"][0]["id"],
                    "quote": "the stated result must be at most the stated limit",
                }
            ],
            calculation=calculation(body),
            missing_information=[],
        )
    )
    result = run(table, models)
    trace = result["result"]["trace"]["rules"]
    assert trace["raw_response"]["calculation"]["inputs"][0]["value"] == 24
    assert trace["model"]["model"] == "fake-rules"
    assert trace["prompt_tokens"] <= trace["prompt_budget"]


def test_check_as_an_ordinary_verb_does_not_launch_rules_model(table):
    _, c, _, sid = table
    c.add_message(
        sid, "Ada", "I check the desk without opening its drawers.", role="player"
    )
    models = Models()
    run(table, models)
    assert [call["task"] for call in models.calls] == ["classifier", "storyteller"]


def test_rejected_advice_is_not_retrieved_as_scenario_evidence(table):
    _, c, _, sid = table
    c.add_message(sid, "Ada", "Should I make a skill roll?", role="player")
    models = Models()
    run(table, models)
    old = next(p for p in c.proposals(sid) if p["kind"] == "rule")
    c.decide(
        sid, old["id"], "rejected", "The previous clock assumption was unsupported."
    )
    models.calls.clear()
    run(table, models, force=True)
    request = next(x["body"] for x in models.calls if x["task"] == "rules")
    assert "brass telescope" in packed(request["scenario_context"])
    assert not any(
        str(d.get("id", "")).startswith("feedback:")
        for d in request["scenario_context"]
    )
    assert any(
        d.get("id") == "feedback:" + old["id"]
        for d in models.calls[-1]["body"]["documents"]
    )


def test_planned_retrieval_sources_reach_both_adviser_and_narrator(table):
    store, c, cid, sid = table
    c.add_document(
        cid,
        "Invented test guidance",
        "Alternative effort is the test instruction.",
        metadata={"kind": "rules"},
    )
    c.add_message(sid, "Ada", "Could I repeat my skill roll?", role="player")
    models = Models(
        rules=lambda body: RulesAnswer(
            answer="Use the supplied test guidance.",
            citations=[
                {
                    "id": next(
                        s["id"]
                        for s in body["rules"]
                        if "Alternative effort" in s["text"]
                    ),
                    "quote": "Alternative effort",
                }
            ],
            calculation=None,
            missing_information=[],
        )
    )
    models.handlers["auditor"] = lambda body: RuleSearchPlan(
        queries=["alternative effort"]
    )
    result = run(table, models)
    assert result["result"]["trace"]["rules"]["status"] == "complete"
    assert any(
        "Alternative effort" in s["text"] for s in models.calls[-1]["body"]["rules"]
    )


class QueuedWorkers:
    """Deterministic executor used to verify fair slices, not model latency."""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        self.jobs.append((fn, args))

    def step(self):
        fn, args = self.jobs.pop(0)
        fn(*args)

    def drain(self, limit=100):
        count = 0
        while self.jobs and count < limit:
            self.step()
            count += 1
        assert not self.jobs, "Background drain failed to converge"
        return count


def scheduler_for(table, models, limit=2, workers=None):
    from story_copilot.campaign_web import CampaignScheduler

    store, c, _, _ = table
    workers = workers or QueuedWorkers()
    build, generate = make_copilot(
        store, model_factory=models, context_options=CONFIG, classify_limit=limit
    )
    return CampaignScheduler(c, workers, generate, build), workers


def test_quiet_burst_drains_every_source_without_repeated_story_or_rules(table):
    _, c, _, sid = table
    mids = [
        c.add_message(
            sid,
            "Ada",
            f"I inspect hatch {i} without touching its controls.",
            role="player",
        )
        for i in range(17)
    ]
    models = Models()
    scheduler, workers = scheduler_for(table, models, limit=3)
    scheduler.schedule(sid)
    assert workers.drain() == 6
    processed = {
        mid
        for r in c.runs(sid)
        for mid in r["result"]["trace"]["classification"]["processed_source_revisions"]
    }
    assert processed == set(mids)
    assert sum(call["task"] == "classifier" for call in models.calls) == 6
    assert sum(call["task"] == "storyteller" for call in models.calls) == 1
    assert sum(p["kind"] == "narration" for p in c.proposals(sid)) == 1
    assert c.runs(sid)[0]["result"]["trace"]["classification"]["backlog"] == []
    assert scheduler.wait_idle(0.1)


def test_work_is_rescheduled_behind_other_sessions_between_slices(table):
    _, c, cid, sid = table
    other = c.create_session(cid, "Another table")
    for i in range(7):
        c.add_message(sid, "Ada", f"First table hatch {i}", role="player")
    c.add_message(other, "Ada", "Second table's immediate question", role="player")
    models = Models()
    scheduler, workers = scheduler_for(table, models)
    scheduler.schedule(sid)
    scheduler.schedule(other)
    workers.step()
    assert len(c.runs(sid)) == 1 and not c.runs(other)
    workers.step()
    assert len(c.runs(other)) == 1 and len(c.runs(sid)) == 1
    workers.drain()
    assert len(c.runs(sid)) == 4
    assert not scheduler.is_pending(sid) and not scheduler.is_pending(other)


def test_new_input_retains_private_history_then_latest_trigger_gets_current_story(
    table,
):
    _, c, _, sid = table
    c.add_message(sid, "Ada", "I consider entering the cellar.", role="player")
    scheduler = None
    first = True

    def classify(body):
        nonlocal first
        if first:
            first = False
            c.add_message(
                sid,
                "Ada",
                "Actually, I refuse to enter and wait outside.",
                role="player",
            )
            scheduler.schedule(sid)
        return Extraction()

    models = Models(classifier=classify)
    scheduler, workers = scheduler_for(table, models)
    scheduler.schedule(sid)
    workers.drain()
    assert sorted(r["status"] for r in c.runs(sid)) == ["complete", "superseded"]
    narration_calls = [call for call in models.calls if call["task"] == "storyteller"]
    assert len(narration_calls) == 2
    assert "refuse to enter" in narration_calls[-1]["body"]["new_player_input"]
    history = next(r for r in c.runs(sid) if r["status"] == "superseded")["result"][
        "historical_draft"
    ]
    assert history["visibility"] == "private" and history["items"]
    assert not history["state_changes_applied"] and not history["published"]
    assert sum(p["kind"] == "narration" for p in c.proposals(sid)) == 2


def test_failed_background_slice_stops_visibly_and_can_resume_without_duplicate_story(
    table,
):
    _, c, _, sid = table
    for i in range(5):
        c.add_message(sid, "Ada", f"I examine room {i}", role="player")
    calls = 0

    def classify(body):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Temporary backend failure")
        return Extraction()

    models = Models(classifier=classify)
    scheduler, workers = scheduler_for(table, models)
    scheduler.schedule(sid)
    assert workers.drain() == 2
    failed = c.runs(sid)[0]
    assert failed["status"] == "failed"
    assert len(failed["result"]["trace"]["classification"]["backlog"]) == 3
    assert not scheduler.is_pending(sid)
    scheduler.schedule(sid)
    workers.drain()
    assert c.runs(sid)[0]["status"] == "complete"
    assert sum(call["task"] == "storyteller" for call in models.calls) == 1
    assert c.runs(sid)[0]["result"]["trace"]["classification"]["backlog"] == []


def test_pause_cancels_queued_and_inflight_slices_then_resume_drains(table):
    _, c, _, sid = table
    for i in range(5):
        c.add_message(sid, "Ada", f"I examine alcove {i}", role="player")
    models = Models()
    scheduler, workers = scheduler_for(table, models)
    scheduler.schedule(sid)
    c.set_proactive(sid, False)
    workers.drain()
    assert not models.calls and not c.runs(sid)
    c.set_proactive(sid, True)
    first = True

    def classify(body):
        nonlocal first
        if first:
            first = False
            c.set_proactive(sid, False)
        return Extraction()

    models.handlers["classifier"] = classify
    scheduler.schedule(sid)
    workers.drain()
    assert c.runs(sid)[0]["status"] == "cancelled"
    assert c.proposals(sid) == []
    assert [call["task"] for call in models.calls] == ["classifier"]
    c.set_proactive(sid, True)
    scheduler.schedule(sid)
    workers.drain()
    assert c.runs(sid)[0]["status"] == "complete"
    assert sum(call["task"] == "storyteller" for call in models.calls) == 1


def test_at_most_one_job_per_session_even_with_multiworker_executor(table):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event as Signal
    from threading import Lock

    _, c, _, sid = table
    for i in range(6):
        c.add_message(sid, "Ada", f"I examine crate {i}", role="player")
    entered, release, mutex = Signal(), Signal(), Lock()
    active = 0
    maximum = 0

    def classify(body):
        nonlocal active, maximum
        with mutex:
            active += 1
            maximum = max(maximum, active)
        entered.set()
        assert release.wait(3)
        with mutex:
            active -= 1
        return Extraction()

    models = Models(classifier=classify)
    with ThreadPoolExecutor(max_workers=3) as workers:
        scheduler, _ = scheduler_for(table, models, workers=workers)
        scheduler.schedule(sid)
        assert entered.wait(3)
        for _ in range(10):
            scheduler.schedule(sid)
        release.set()
        assert scheduler.wait_idle(5)
    assert maximum == 1
    assert sum(call["task"] == "storyteller" for call in models.calls) == 1
    assert c.runs(sid)[0]["result"]["trace"]["classification"]["backlog"] == []


def test_reject_requests_one_private_alternative_without_changing_record(table):
    _, c, _, sid = table
    c.add_message(sid, "Ada", "I inspect the locked cabinet.", role="player")
    models = Models()
    scheduler, workers = scheduler_for(table, models)
    scheduler.schedule(sid)
    workers.drain()
    first = next(p for p in c.proposals(sid) if p["kind"] == "narration")
    initial = c.state(sid)
    scheduler.schedule(sid)
    workers.drain()
    assert sum(call["task"] == "storyteller" for call in models.calls) == 1
    with pytest.raises(ValueError, match="private"):
        c.publish(sid, first["id"], first["text"])
    c.decide(sid, first["id"], "rejected")
    scheduler.schedule(sid, force=True)
    workers.drain()
    assert sum(call["task"] == "storyteller" for call in models.calls) == 2
    assert c.state(sid) == initial and len(c.messages(sid)) == 1
    assert c.snapshot(sid)["feedback"][0]["decision"] == "rejected"
    assert c.snapshot(sid, public_only=True)["proposals"] == []


def test_accepting_a_draft_during_quiet_drain_does_not_interrupt_or_repeat_story(table):
    _, c, _, sid = table
    for i in range(5):
        c.add_message(sid, "Ada", f"I record the label on crate {i}", role="player")
    models = Models()
    scheduler, workers = scheduler_for(table, models)
    scheduler.schedule(sid)
    workers.step()
    p = next(p for p in c.proposals(sid) if p["kind"] == "narration")
    c.decide(sid, p["id"], "accepted")
    scheduler.schedule(sid)
    workers.drain()
    assert all(r["status"] == "complete" for r in c.runs(sid))
    assert sum(call["task"] == "storyteller" for call in models.calls) == 1
    assert c.runs(sid)[0]["result"]["trace"]["classification"]["backlog"] == []


def test_single_speaker_capture_does_not_duplicate_whole_history_as_current_trigger(
    table,
):
    store, c, _, sid = table
    for i in range(150):
        c.add_message(
            sid,
            "mic:unmapped",
            f"Distinct captured utterance {i}: I consider the room carefully without acting yet.",
            role="unknown",
            source={"kind": "live_audio", "channel": "mic"},
        )
    config = {
        **CONFIG,
        "context_limit": 2600,
        "output_reserve": 500,
        "safety_margin": 100,
    }
    build, _ = make_copilot(store, model_factory=Models(), context_options=config)
    context = build(c.snapshot(sid))
    assert "utterance 149" in context["body"]["new_player_input"]
    assert "utterance 0:" not in context["body"]["new_player_input"]
    assert len(context["body"]["new_player_input"]) < 200
    assert context["context_trace"]["compacted"]
    assert (
        context["context_trace"]["prompt_tokens"]
        <= context["context_trace"]["prompt_budget"]
    )


def test_continuous_appends_produce_private_as_of_drafts_and_reuse_classifier_without_starvation(
    table,
):
    _, c, _, sid = table
    epoch = 1_800_000_000
    sequence = 0

    def append(text, role="player"):
        nonlocal sequence
        sequence += 1
        return c.add_message(
            sid,
            "mic:facilitator" if role == "facilitator" else "system:player",
            text,
            role=role,
            source={
                "kind": "live_audio",
                "timeline": {
                    "kind": "live_capture",
                    "domain": "capture:shared",
                    "start_seconds": sequence * 10,
                    "end_seconds": sequence * 10 + 1,
                    "utc_start": epoch + sequence * 10,
                    "utc_end": epoch + sequence * 10 + 1,
                },
            },
        )

    original = append("Ada loses two hit points.", "facilitator")
    append("I inspect the closed door.")
    scheduler = None
    narration = 0

    def classify(body):
        events = []
        if any(m["id"] == original for m in body["target_turns"]):
            events = [event("resource", 1, "Ada loses two hit points", delta=-2)]
        return Extraction(events=events)

    def narrate(body):
        nonlocal narration
        narration += 1
        if narration <= 3:
            append(f"A new player contribution arrives during generation {narration}.")
            scheduler.schedule(sid)
        return NarrationAnswer(narration=f"Snapshot draft number {narration}.")

    models = Models(classifier=classify, storyteller=narrate)
    scheduler, workers = scheduler_for(table, models)
    scheduler.schedule(sid)
    workers.step()
    first = c.runs(sid)[0]
    assert first["status"] == "superseded"
    assert (
        first["result"]["historical_draft"]["items"][0]["text"]
        == "Snapshot draft number 1."
    )
    assert first["result"][
        "proposal_ids"
    ]  # Private point-in-time guidance can be refreshed.
    assert len(first["result"]["observation_ids"]) == 1
    assert c.publications(sid) == []
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 10
    workers.drain()
    histories = [r for r in c.runs(sid) if r["status"] == "superseded"]
    assert len(histories) == 3 and narration == 4
    assert all(
        r["result"]["historical_draft"]["visibility"] == "private" for r in histories
    )
    assert all(r["result"]["proposal_ids"] for r in histories)
    observed_batches = [
        call["body"]["target_turns"]
        for call in models.calls
        if call["task"] == "classifier"
    ]
    assert (
        sum(any(m["id"] == original for m in batch) for batch in observed_batches) == 1
    )
    pending = [p for p in c.proposals(sid) if p["kind"] == "state"]
    assert len(pending) == 1 and pending[0]["status"] == "pending"
    assert pending[0]["observed"]
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 10
    assert c.snapshot(sid, public_only=True)["proposals"] == []


def test_append_only_cached_analysis_is_not_reused_after_canon_or_direction_change(
    table,
):
    store, c, cid, sid = table
    original = c.add_message(
        sid, "Facilitator", "Ada loses two hit points.", role="facilitator"
    )
    first = True

    def narrate(body):
        nonlocal first
        if first:
            first = False
            c.add_message(sid, "Ada", "I step away.", role="player")
        return NarrationAnswer(narration="A private historical draft.")

    models = Models(
        classifier=lambda body: Extraction(
            events=[event("resource", 1, "Ada loses two hit points", delta=-2)]
        ),
        storyteller=narrate,
    )
    result = run(table, models)
    assert result["status"] == "superseded"
    build, _ = make_copilot(store, model_factory=models, context_options=CONFIG)
    valid = build(c.snapshot(sid))
    assert not valid[
        "cached_classification"
    ]  # Already materialized once as an observation.
    assert original not in [m["id"] for m in valid["target_messages"]]
    c.update(
        cid,
        title="Original synthetic station",
        direction="New instruction: focus only on the exterior.",
    )
    changed = build(c.snapshot(sid))
    assert not changed["cached_classification"]
    assert original in [m["id"] for m in changed["target_messages"]]


def test_late_clock_arrival_still_cancels_before_narrator_and_never_makes_history(
    table,
):
    _, c, _, sid = table

    def timed(text, seconds):
        return c.add_message(
            sid,
            "mic:0",
            text,
            role="player",
            source={
                "kind": "live_audio",
                "timeline": {
                    "kind": "live_capture",
                    "domain": "shared",
                    "start_seconds": seconds,
                    "end_seconds": seconds + 1,
                    "utc_start": 1_800_000_000 + seconds,
                    "utc_end": 1_800_000_000 + seconds + 1,
                },
            },
        )

    timed("I inspect the latch.", 20)

    def classify(body):
        timed("An earlier instruction was delivered late.", 10)
        return Extraction()

    models = Models(classifier=classify)
    result = run(table, models)
    assert result["status"] == "stale"
    assert "historical_draft" not in result["result"]
    assert [call["task"] for call in models.calls] == ["classifier"]
    assert c.proposals(sid) == []


def test_known_mapping_reanalyzes_old_unknown_speech_without_rewriting_it(table):
    store, c, cid, sid = table
    mid = c.add_message(
        sid, "system:SPEAKER_03", "I lose two hit points.", role="unknown"
    )
    models = Models(
        classifier=lambda body: Extraction(
            events=[event("resource", 1, "I lose two hit points", delta=-2)]
        )
    )
    first = run(table, models)
    assert first["result"]["trace"]["classification"]["rejected_fragments"]
    char = c.characters(cid)[0]
    c.map_participant(
        cid, "Alex", "system:SPEAKER_03", role="player", character_id=char["id"]
    )
    second = run(table, models)
    assert second["request"]["target_messages"][0]["id"] == mid
    assert second["request"]["body"]["dialogue"][0]["role"] == "unknown"
    assert (
        second["request"]["body"]["dialogue"][0]["speaker_mapping"]["character"]
        == "Ada"
    )
    assert (
        c.messages(sid)[0]["role"] == "unknown"
        and c.messages(sid)[0]["revision"] == mid
    )
    p = next(p for p in c.proposals(sid) if p["kind"] == "state")
    assert p["evidence"][0]["speaker_mapping"]["character_id"] == char["id"]
    c.decide(sid, p["id"], "accepted")
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 10
    other = c.save_character(cid, "Bea", {"resources": {"HP": 20}})
    c.map_participant(
        cid, "Alex", "system:SPEAKER_03", role="player", character_id=other
    )
    assert next(p2 for p2 in c.proposals(sid) if p2["id"] == p["id"])["stale"]
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 12
    build, _ = make_copilot(store, model_factory=Models(), context_options=CONFIG)
    assert build(c.snapshot(sid))["target_messages"][0]["id"] == mid


def test_history_ui_is_private_copyable_and_has_no_state_acceptance_or_publish_controls(
    table,
):
    from starlette.testclient import TestClient

    from story_copilot.web import create_app

    _, c, _, sid = table
    c.add_message(sid, "Ada", "I inspect the door.", role="player")

    def narrate(body):
        c.add_message(
            sid, "Ada", "I keep talking while the model responds.", role="player"
        )
        return NarrationAnswer(narration="HISTORICAL_PRIVATE_WORDING")

    result = run(table, Models(storyteller=narrate))
    assert result["status"] == "superseded"
    app = create_app(c.store)
    with TestClient(app) as client:
        private = client.get(f"/play/{sid}").text
        assert "HISTORICAL_PRIVATE_WORDING" in private
        assert "As of source #1" in private and "Copy wording" in private
        assert (
            "Accept suggestion" not in private
            and "Publish wording to player view" not in private
        )
        public = client.get(f"/play/{sid}/public").text
        assert "HISTORICAL_PRIVATE_WORDING" not in public


def test_superseded_cache_and_copyable_history_are_invalidated_by_later_source_correction(
    table,
):
    from starlette.testclient import TestClient

    from story_copilot.web import create_app

    store, c, _, sid = table
    mid = c.add_message(
        sid, "Facilitator", "Ada loses two hit points.", role="facilitator"
    )

    def narrate(body):
        c.add_message(sid, "Ada", "I step away.", role="player")
        return NarrationAnswer(narration="HISTORY_TO_INVALIDATE")

    models = Models(
        classifier=lambda body: Extraction(
            events=[event("resource", 1, "Ada loses two hit points", delta=-2)]
        ),
        storyteller=narrate,
    )
    assert run(table, models)["status"] == "superseded"
    c.revise_message(
        sid,
        mid,
        text="Ada was not hurt.",
        speaker="Facilitator",
        role="facilitator",
        visibility="public",
        expected_revision=mid,
    )
    build, _ = make_copilot(store, model_factory=Models(), context_options=CONFIG)
    current = build(c.snapshot(sid))
    assert not current["cached_classification"]
    assert mid in [m["id"] for m in current["target_messages"]]
    with TestClient(create_app(store)) as client:
        page = client.get(f"/play/{sid}").text
        # Earlier advice stays private history, never the current response or record.
        assert page.index("Earlier private suggestions") < page.index(
            "HISTORY_TO_INVALIDATE"
        )
        assert c.state(sid)["resources"]["Ada:HP"]["value"] == 12
        assert "HISTORY_TO_INVALIDATE" not in client.get(f"/play/{sid}/public").text
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 12


def test_unknown_mapped_numeric_roll_uses_explicit_identity_without_changing_transcript_role(
    table,
):
    _, c, cid, sid = table
    c.add_message(
        sid,
        "Facilitator",
        "Make a regular Observation check at 60.",
        role="facilitator",
    )
    mid = c.add_message(sid, "system:unknown", "I rolled 24.", role="unknown")
    c.map_participant(
        cid,
        "Alex",
        "system:unknown",
        role="player",
        character_id=c.characters(cid)[0]["id"],
    )
    models = Models(
        rules=lambda body: RulesAnswer(
            answer="Use the supplied values.",
            citations=[
                {
                    "id": body["rules"][0]["id"],
                    "quote": "the stated result must be at most the stated limit",
                }
            ],
            calculation=calculation(body),
            missing_information=[],
        )
    )
    result = run(table, models)
    assert result["result"]["trace"]["rules"]["status"] == "complete"
    assert result["result"]["trace"]["rules"]["calculation"]["result"]
    assert next(m for m in c.messages(sid) if m["id"] == mid)["role"] == "unknown"
