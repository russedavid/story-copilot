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
