import json

import pytest

from story_copilot.response_grounding import (
    ClaimCheck,
    claim_units,
    sources,
    validate_checks,
)
from story_copilot.response_review import ResponseReview, review_response
from story_copilot.schema import DirectAnswer, NarrationAnswer


OPTIONS = {"context_limit": 16384, "token_counter": lambda value: len(value) // 4}


def body():
    return {
        "state": {"entities": {"Tess": {"sheet": {}}, "Ivo": {"sheet": {}}}},
        "dialogue": [
            {
                "id": "clue",
                "speaker": "Facilitator",
                "role": "facilitator",
                "visibility": "private",
                "text": "Only Tess notices the blue thread. Ivo has not seen it.",
            }
        ],
        "documents": [],
        "rules": [],
    }


def checks(request, *, verdict="unsupported", support=()):
    return [
        dict(
            id=unit["id"],
            verdict=verdict,
            support=list(support),
            reason="Fixture assessment.",
        )
        for unit in request["required_claims"]
    ]


def test_all_fields_include_unassigned_player_reactions_and_zero_rule_claims():
    draft = DirectAnswer(
        direct_answer="Ivo looks uneasy.",
        private_notes="The rule is unspecified, so there is no teamwork bonus.",
    )
    units = claim_units(body(), draft)
    assert {u["field"] for u in units} == {"direct_answer", "private_notes"}
    zero = next(u for u in units if u["field"] == "private_notes")
    assert zero["explicit_zero_rule"] and not zero["can_be_nonassertion"]


def test_unknown_does_not_become_zero_even_with_an_exact_dialogue_quote():
    context = body()
    context["dialogue"].append(
        {
            "id": "question",
            "text": "We have not chosen a teamwork rule.",
            "visibility": "public",
        }
    )
    draft = NarrationAnswer(narration="There is no teamwork bonus.")
    units = claim_units(context, draft)
    assessment = ClaimCheck(
        id=units[0]["id"],
        verdict="supported",
        support=[
            {"source_id": "question", "quote": "We have not chosen a teamwork rule."}
        ],
        reason="Bad model inference.",
    )
    with pytest.raises(ValueError, match="applicable supplied rule"):
        validate_checks(units, [assessment], sources(context))


def test_explicit_zero_rule_is_accepted_but_an_unrelated_rule_is_not():
    context = body()
    context["rules"] = [
        {"id": "rule", "text": "The teamwork modifier is zero.", "visibility": "public"}
    ]
    units = claim_units(
        context, NarrationAnswer(narration="There is no teamwork modifier.")
    )
    assessment = ClaimCheck(
        id=units[0]["id"],
        verdict="supported",
        support=[{"source_id": "rule", "quote": "The teamwork modifier is zero."}],
        reason="Explicit supplied value.",
    )
    assert not validate_checks(units, [assessment], sources(context))
    context["rules"][0]["text"] = "The door opens after three turns."
    assessment.support[0].quote = context["rules"][0]["text"]
    with pytest.raises(ValueError, match="cannot establish zero"):
        validate_checks(units, [assessment], sources(context))


def test_missing_claim_assessment_never_silently_passes_an_invented_reaction():
    class Missed:
        def complete(self, *args, **kwargs):
            return ResponseReview(issues=[], revision=None), {}

    draft = DirectAnswer(direct_answer="Ivo looks uneasy.")
    result, trace = review_response(body(), draft, Missed(), OPTIONS)
    assert trace["status"] == "guarded" and "uneasy" not in result.direct_answer
    assert trace["original"] == draft.model_dump()


def test_valid_source_quote_cannot_launder_an_invented_player_expression():
    context = body()
    units = claim_units(context, DirectAnswer(direct_answer="Ivo looks uneasy."))
    check = ClaimCheck(
        id=units[0]["id"],
        verdict="supported",
        reason="Faulty entailment judgment.",
        support=[{"source_id": "clue", "quote": "Ivo has not seen it."}],
    )
    with pytest.raises(ValueError, match="unsupported gestures"):
        validate_checks(units, [check], sources(context))


def test_npc_invention_and_questions_remain_available():
    class Clean:
        def complete(self, messages, *args, **kwargs):
            request = json.loads(messages[-1]["content"])
            assert request["required_claims"] == []
            return ResponseReview(issues=[], revision=None), {}

    draft = NarrationAnswer(
        narration='The caretaker says, "The ferry leaves at dusk."',
        questions=["What does Tess decide?"],
    )
    result, trace = review_response(body(), draft, Clean(), OPTIONS)
    assert result == draft and trace["status"] == "no_issue_found"


def test_repair_cannot_replace_one_invented_player_action_with_another():
    class RepeatedError:
        calls = 0

        def complete(self, messages, *args, **kwargs):
            self.calls += 1
            request = json.loads(messages[-1]["content"])
            return ResponseReview(
                claim_checks=checks(request),
                issues=[
                    {
                        "kind": "player_agency",
                        "quote": request["draft"]["narration"],
                        "reason": "No source supports this reaction.",
                    }
                ],
                revision=NarrationAnswer(
                    narration="Ivo nods." if self.calls == 1 else "Ivo waves."
                ),
            ), {}

    model = RepeatedError()
    result, trace = review_response(
        body(), NarrationAnswer(narration="Ivo smiles."), model, OPTIONS
    )
    assert model.calls == 2 and trace["status"] == "guarded"
    assert not result.narration and "waves" not in result.direct_answer


def test_generated_feedback_cannot_ground_a_player_reaction():
    context = body()
    context["documents"] = [
        {"id": "feedback:earlier", "text": "Ivo looks uneasy.", "visibility": "private"}
    ]
    units = claim_units(context, DirectAnswer(direct_answer="Ivo looks uneasy."))
    assessment = ClaimCheck(
        id=units[0]["id"],
        verdict="supported",
        support=[{"source_id": "feedback:earlier", "quote": "Ivo looks uneasy."}],
        reason="Bad source selection.",
    )
    with pytest.raises(ValueError, match="unavailable"):
        validate_checks(units, [assessment], sources(context))


def test_source_supported_private_fact_does_not_become_shared_narration():
    context = body()
    units = claim_units(
        context, NarrationAnswer(narration="Only Tess notices the blue thread.")
    )
    assessment = ClaimCheck(
        id=units[0]["id"],
        verdict="supported",
        support=[{"source_id": "clue", "quote": "Only Tess notices the blue thread."}],
        reason="Exact but private.",
    )
    with pytest.raises(ValueError, match="private sources"):
        validate_checks(units, [assessment], sources(context))


def test_evaluation_counts_all_review_attempts_without_double_counting_last_model():
    from story_copilot.evaluate import summarize

    first = {"model": "base", "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    second = {"model": "base", "usage": {"prompt_tokens": 120, "completion_tokens": 10}}
    run = {
        "status": "complete",
        "result": {
            "trace": {
                "response_review": {
                    "model": second,
                    "attempts": [{"model": first}, {"model": second}],
                    "status": "revised",
                }
            }
        },
    }
    result = summarize(run)
    assert result["calls"] == 2
    assert result["prompt_tokens"] == 220 and result["completion_tokens"] == 30
