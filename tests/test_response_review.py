import pytest
import json
from story_copilot.response_grounding import ClaimCheck, claim_units
from jsonschema import Draft202012Validator

from story_copilot.response_review import ResponseReview, review_response
from story_copilot.schema import NarrationAnswer, GenerationExtraction
from story_copilot.wire_schema import wire_schema

OPTIONS = {"context_limit": 16384, "token_counter": lambda text: len(text) // 4}


class Model:
    def __init__(self, answer):
        self.answer = answer

    def complete(self, *args, **kwargs):
        return self.answer, {"model": "scripted"}


def test_anchored_edit_preserves_original_and_does_not_claim_independent_accuracy():
    draft = NarrationAnswer(narration='Tess says, "I agree."')
    edited = NarrationAnswer(
        narration="The caretaker waits.", questions=["What does Tess decide?"]
    )
    model = Model(
        ResponseReview(
            issues=[
                {
                    "kind": "player_agency",
                    "quote": 'Tess says, "I agree."',
                    "reason": "Tess is controlled by a participant.",
                }
            ],
            revision=edited,
        )
    )
    first = model.answer

    class Repair:
        def complete(self, messages, *args, **kwargs):
            request = json.loads(messages[-1]["content"])
            if request["draft"]["narration"] == draft.narration:
                first.claim_checks = [
                    ClaimCheck(
                        id=u["id"],
                        verdict="unsupported",
                        reason="Player has not spoken.",
                    )
                    for u in request["required_claims"]
                ]
                return first, {}
            return (
                ResponseReview(
                    issues=[],
                    revision=None,
                    claim_checks=[
                        ClaimCheck(
                            id=u["id"],
                            verdict="creative_proposal",
                            actor="caretaker",
                            reason="NPC proposal.",
                        )
                        for u in request["required_claims"]
                    ],
                ),
                {},
            )

    result, trace = review_response({}, draft, Repair(), OPTIONS)
    assert result == edited and trace["original"] == draft.model_dump()
    assert trace["status"] == "revised"


def test_unanchored_criticism_does_not_silently_replace_the_draft():
    draft = NarrationAnswer(narration="The caretaker waits.")
    model = Model(
        ResponseReview(
            issues=[
                {
                    "kind": "player_agency",
                    "quote": "Words that were never written",
                    "reason": "An invented defect.",
                }
            ],
            revision=NarrationAnswer(narration="Replacement."),
        )
    )
    model.answer.claim_checks = [
        ClaimCheck(
            id=u["id"],
            verdict="creative_proposal",
            actor="caretaker",
            reason="NPC proposal.",
        )
        for u in claim_units({}, draft)
    ]
    result, trace = review_response({}, draft, model, OPTIONS)
    assert result == draft and trace["status"] == "guarded"


def test_context_limit_does_not_drop_sources_to_force_a_review():
    draft = NarrationAnswer(narration="An optional reply.")
    result, trace = review_response(
        {"source": "a" * 20000}, draft, Model(None), {"context_limit": 4096}
    )
    assert not result.narration and trace["status"] == "guarded"
    assert trace["original"] == draft.model_dump()
    assert "model" not in trace


def test_review_can_use_explicit_serving_headroom_without_dropping_sources(monkeypatch):
    monkeypatch.setenv("STORY_MAX_CONTEXT_TOKENS", "16384")
    source = {"source": "x" * 18000}
    draft = NarrationAnswer(narration="The caretaker waits.")

    class Check:
        def complete(self, messages, *args, **kwargs):
            request = json.loads(messages[-1]["content"])
            assert request["context"] == source
            assert all(
                "preceding_text" not in unit for unit in request["required_claims"]
            )
            return (
                ResponseReview(
                    issues=[],
                    revision=None,
                    claim_checks=[
                        ClaimCheck(
                            id=u["id"],
                            verdict="creative_proposal",
                            actor="caretaker",
                            reason="NPC proposal.",
                        )
                        for u in request["required_claims"]
                    ],
                ),
                {},
            )

    answer, trace = review_response(
        source, draft, Check(), {**OPTIONS, "context_limit": 4096}
    )
    assert answer == draft and trace["expanded_for_review"]
    assert trace["preferred_context_limit"] == 4096
    assert trace["prompt_tokens"] <= trace["prompt_budget"] < 16384


def test_generation_schema_forces_valid_event_shapes_without_losing_fragments():
    validator = Draft202012Validator(wire_schema(GenerationExtraction))
    base = {
        "kind": "resource",
        "entity": "Tess",
        "attribute": "supplies",
        "stage": "established",
        "value": 6,
        "delta": None,
        "evidence": [{"turn": 1, "quote": "I have 6 supplies."}],
        "resolves": None,
        "supersedes": None,
        "rationale": "",
        "visibility": "public",
    }
    assert not list(validator.iter_errors({"events": [base], "uncertainties": []}))
    for changes in [
        {"delta": -2},
        {"stage": "reported"},
        {"kind": "fact", "resolves": "old-action"},
    ]:
        assert list(
            validator.iter_errors(
                {"events": [{**base, **changes}], "uncertainties": []}
            )
        )
    assert not list(
        validator.iter_errors(
            {
                "events": [{**base, "kind": "claim", "stage": "reported"}],
                "uncertainties": [],
            }
        )
    )
    assert not list(
        validator.iter_errors(
            {"events": [{**base, "value": None, "delta": -2}], "uncertainties": []}
        )
    )


def test_rule_display_does_not_repeat_confident_prose_when_a_rule_is_missing():
    from story_copilot.rule_advice import RulesAnswer, display_advice

    answer = RulesAnswer(
        answer="There is no bonus.",
        citations=[],
        calculation=None,
        missing_information=["The assistance rule is unspecified."],
    )
    assert (
        display_advice(answer, None)
        == "A ruling needs more information: The assistance rule is unspecified."
    )
    answer = answer.model_copy(update={"missing_information": []})
    display = display_advice(
        answer,
        {
            "tool": "can_afford",
            "result": True,
            "operation": "greater_equal",
            "inputs": [2, 1],
        },
    )
    assert display == "Can afford: yes.\nCalculation: 2 ≥ 1"


def test_compact_review_resolves_source_spans_before_existing_claim_guards():
    from story_copilot.model import LocalModel
    from story_copilot.response_review import CompactResponseReview, CompactClaim

    context = {
        "state": {"entities": {"Ivo": {"sheet": {}}}},
        "dialogue": [
            {
                "id": "seen",
                "text": "Ivo has not seen the thread.",
                "visibility": "public",
            }
        ],
    }

    class CompactModel(LocalModel):
        def __init__(self):
            pass

        def complete(self, messages, schema, **kwargs):
            request = json.loads(messages[-1]["content"])
            assert schema is CompactResponseReview
            assert request["required_claims"][0]["id"] == 0
            assert request["source_spans"][0]["quote"] == "Ivo has not seen the thread."
            return CompactResponseReview(
                claim_checks=[CompactClaim(claim=0, verdict="supported", sources=[0])],
                issues=[],
                revision=None,
            ), {"model": "scripted compact review"}

    answer, trace = review_response(
        context,
        NarrationAnswer(narration="", direct_answer="Ivo looks uneasy."),
        CompactModel(),
        OPTIONS,
    )
    assert answer.direct_answer == "Ivo has not seen the thread."
    assert trace["attempts"][0]["compact_review"]["claim_checks"][0]["sources"] == [0]
    assert trace["review"]["claim_checks"][0]["support"] == [
        {"source_id": "seen", "quote": "Ivo has not seen the thread."}
    ]


def test_compact_indices_cannot_invent_a_source_or_coerce_a_boolean_reference():
    from story_copilot.response_review import (
        CompactResponseReview,
        CompactClaim,
        expand_review,
    )
    from pydantic import ValidationError

    units = claim_units({}, NarrationAnswer(narration="The bell rings."))
    result = CompactResponseReview(
        claim_checks=[CompactClaim(claim=0, verdict="supported", sources=[9])],
        issues=[],
        revision=None,
    )
    with pytest.raises(ValueError, match="unavailable"):
        expand_review(result, units, [])
    with pytest.raises(ValidationError):
        CompactClaim(claim=0, verdict="supported", sources=[True])


def test_source_catalog_keeps_complete_qualifiers_decimals_and_visibility():
    from story_copilot.response_review import source_spans

    spans = source_spans(
        [
            {
                "id": "rule",
                "kind": "rule",
                "visibility": "private",
                "text": "The cost is 1.5 cells. A teamwork modifier is unknown.",
            }
        ]
    )
    assert [s["quote"] for s in spans] == [
        "The cost is 1.5 cells.",
        "A teamwork modifier is unknown.",
    ]
    assert all(s["source_id"] == "rule" and s["visibility"] == "private" for s in spans)


def test_scene_contract_does_not_generate_an_unrelated_bookkeeping_slot():
    from story_copilot.schema import SceneAnswer, DirectAnswer
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SceneAnswer(
            narration="The caretaker replies.",
            direct_answer="An unrelated old balance.",
        )
    assert SceneAnswer(narration="The caretaker replies.").direct_answer == ""
    assert DirectAnswer(direct_answer="Three charges remain.").narration == ""
    assert wire_schema(SceneAnswer)["properties"]["direct_answer"]["const"] == ""
