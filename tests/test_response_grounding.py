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
    from story_copilot.response_grounding import source_wording

    context = body()
    answer = DirectAnswer(direct_answer="Ivo looks uneasy.")
    units = claim_units(context, answer)
    check = ClaimCheck(
        id=units[0]["id"],
        verdict="supported",
        reason="Faulty entailment judgment.",
        support=[{"source_id": "clue", "quote": "Ivo has not seen it."}],
    )
    validate_checks(units, [check], sources(context))
    rendered, edits = source_wording(answer, units, [check], sources(context))
    assert rendered.direct_answer == "Ivo has not seen it."
    assert edits[0]["original"] == "Ivo looks uneasy."


def test_npc_invention_and_questions_remain_available():
    class Clean:
        def complete(self, messages, *args, **kwargs):
            request = json.loads(messages[-1]["content"])
            return (
                ResponseReview(
                    issues=[],
                    revision=None,
                    claim_checks=[
                        ClaimCheck(
                            id=u["id"],
                            verdict="creative_proposal",
                            actor="caretaker",
                            reason="Proposed NPC reply.",
                        )
                        for u in request["required_claims"]
                    ],
                ),
                {},
            )

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
            return (
                ResponseReview(
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
                ),
                {},
            )

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


def test_private_commentary_does_not_turn_banter_into_an_event():
    from story_copilot.response_grounding import source_wording

    context = {
        "dialogue": [
            {
                "id": "banter",
                "text": "Your hat looks ridiculous.",
                "speaker": "voice-1",
                "visibility": "private",
            }
        ]
    }
    answer = NarrationAnswer(
        narration="The door stays closed.",
        private_notes="Someone was attacked during the conversation.",
    )
    units = [u for u in claim_units(context, answer) if u["field"] == "private_notes"]
    assert len(units) == 1 and units[0]["kinds"] == ["private_context"]
    check = ClaimCheck(
        id=units[0]["id"],
        verdict="supported",
        reason="Deliberately faulty assessment.",
        support=[{"source_id": "banter", "quote": "Your hat looks ridiculous."}],
    )
    validate_checks(units, [check], sources(context))
    rendered, edits = source_wording(answer, units, [check], sources(context))
    assert "attacked" not in rendered.private_notes
    assert rendered.private_notes == "Recorded speech: “Your hat looks ridiculous.”"
    assert rendered.narration == answer.narration and edits


def test_private_advice_can_remain_advice_without_a_fabricated_source():
    class Advice:
        def complete(self, messages, *args, **kwargs):
            request = json.loads(messages[-1]["content"])
            return (
                ResponseReview(
                    issues=[],
                    revision=None,
                    claim_checks=[
                        ClaimCheck(
                            id=u["id"],
                            verdict=(
                                "creative_proposal"
                                if u["field"] == "narration"
                                else "not_an_assertion"
                            ),
                            actor="ferryman" if u["field"] == "narration" else "",
                            reason="NPC proposal or imperative advice.",
                        )
                        for u in request["required_claims"]
                    ],
                ),
                {},
            )

    answer = NarrationAnswer(
        narration="The ferryman waits.",
        private_notes="Consider asking about the weather before proposing a crossing.",
    )
    actual, trace = review_response(body(), answer, Advice(), OPTIONS)
    assert actual == answer and trace["status"] == "no_issue_found"


def test_tool_queries_and_rejected_interpretations_are_not_evidence():
    context = body()
    context["documents"] = [
        {
            "id": "evidence-investigation",
            "text": json.dumps(
                {
                    "observations": [
                        {
                            "tool": "recall",
                            "query": "Ivo looks uneasy",
                            "result": [
                                {
                                    "id": "earlier",
                                    "text": "Ivo waits at the door.",
                                    "visibility": "public",
                                }
                            ],
                        }
                    ]
                }
            ),
        },
        {"id": "unresolved-source-analysis", "text": "Ivo looks uneasy."},
    ]
    evidence = sources(context)
    assert any(x["id"] == "earlier" for x in evidence)
    assert all("uneasy" not in x["text"] for x in evidence)


def test_decimal_cost_is_one_complete_claim():
    units = claim_units(
        {}, NarrationAnswer(narration="", direct_answer="The cost is 1.5 credits.")
    )
    assert len(units) == 1 and units[0]["text"] == "The cost is 1.5 credits."


def test_creative_npc_can_address_a_player_without_inventing_the_players_reply():
    context = body()
    answer = NarrationAnswer(narration="The caretaker offers Ivo a seat by the fire.")
    units = claim_units(context, answer)
    check = ClaimCheck(
        id=units[0]["id"],
        verdict="creative_proposal",
        actor="caretaker",
        reason="NPC action; no player acceptance implied.",
    )
    assert not validate_checks(units, [check], sources(context))
    bad = claim_units(context, NarrationAnswer(narration="Ivo sits by the fire."))
    check.id = bad[0]["id"]
    with pytest.raises(ValueError, match="player performance"):
        validate_checks(bad, [check], sources(context))
    history = claim_units(
        context,
        NarrationAnswer(
            narration="The room is quiet.",
            private_notes="Someone attacked the visitor.",
        ),
    )
    history = [u for u in history if u["field"] == "private_notes"]
    check.id = history[0]["id"]
    check.actor = "Someone"
    with pytest.raises(ValueError, match="private factual history"):
        validate_checks(history, [check], sources(context))


def test_quoted_npc_dialogue_keeps_ownership_across_sentences():
    context = body()
    answer = NarrationAnswer(
        narration="The caretaker says, “The ferry leaves at dusk. You can wait here until then.”"
    )
    units = [u for u in claim_units(context, answer) if u["text"].startswith("You")]
    assert (
        len(units) == 1
        and units[0]["quoted_dialogue"]
        and not units[0]["literal_player_subject"]
    )
    check = ClaimCheck(
        id=units[0]["id"],
        verdict="creative_proposal",
        actor="caretaker",
        reason="Continuation of NPC speech.",
    )
    assert not validate_checks(units, [check], sources(context))
    player = NarrationAnswer(narration="Ivo says, “I agree. You can count on me.”")
    units = claim_units(context, player)
    continued = next(u for u in units if u["text"].startswith("You"))
    assert continued["quoted_player_speech"]
    check.id = continued["id"]
    with pytest.raises(ValueError, match="player performance"):
        validate_checks([continued], [check], sources(context))


def test_npc_reply_in_direct_answer_and_pronoun_continuations_are_preserved():
    context = body()
    answer = NarrationAnswer(
        narration="",
        direct_answer="The caretaker says, “You can wait here. Your choice.”",
    )
    units = claim_units(context, answer)
    for unit in units:
        check = ClaimCheck(
            id=unit["id"],
            verdict="creative_proposal",
            actor="caretaker",
            reason="NPC offers a choice.",
        )
        assert not validate_checks([unit], [check], sources(context))
    answer = NarrationAnswer(
        narration="The caretaker lights the fire. She offers Ivo a seat."
    )
    units = [u for u in claim_units(context, answer) if u["text"].startswith("She")]
    check = ClaimCheck(
        id=units[0]["id"],
        verdict="creative_proposal",
        actor="caretaker",
        reason="NPC action; player may decline.",
    )
    assert not validate_checks(units, [check], sources(context))
    check.actor = "Tess"
    with pytest.raises(ValueError, match="player performance"):
        validate_checks(units, [check], sources(context))


def test_clause_removal_does_not_leave_unbalanced_dialogue_quotes():
    from story_copilot.response_grounding import remove_claims

    answer = NarrationAnswer(narration="The caretaker says, “Wait here. Ivo nods.”")
    result = remove_claims(answer, [{"field": "narration", "text": "Ivo nods.”"}])
    assert result.narration == "The caretaker says, Wait here."
    answer = NarrationAnswer(narration="“Ivo nods. The boat is ready.”")
    result = remove_claims(answer, [{"field": "narration", "text": "“Ivo nods."}])
    assert result.narration == "The boat is ready."
    answer = NarrationAnswer(narration="“Wait here. Ivo nods. The boat is ready.”")
    result = remove_claims(answer, [{"field": "narration", "text": "Ivo nods."}])
    assert result.narration.startswith("“") and result.narration.endswith("”")


def test_npc_absence_claim_cannot_be_supported_by_explicit_uncertainty():
    context = {
        "documents": [
            {
                "id": "scene",
                "text": "Nobody knows whether visitors arrived.",
                "visibility": "public",
            }
        ]
    }
    units = claim_units(context, NarrationAnswer(narration='"No visitors."'))
    assert units[0]["kinds"] == ["world_context"]
    check = ClaimCheck(
        id=units[0]["id"],
        verdict="supported",
        support=[
            {"source_id": "scene", "quote": "Nobody knows whether visitors arrived."}
        ],
        reason="Deliberately false entailment.",
    )
    with pytest.raises(ValueError, match="unknown source"):
        validate_checks(units, [check], sources(context))
    context["documents"][0]["text"] = "No visitors."
    check.support[0].quote = "No visitors."
    assert not validate_checks(units, [check], sources(context))


def test_last_repair_keeps_verified_uncertainty_instead_of_restoring_old_certainty():
    context = {
        "documents": [
            {
                "id": "scene",
                "text": "Nobody knows whether visitors arrived.",
                "visibility": "public",
            }
        ]
    }

    class Repair:
        calls = 0

        def complete(self, messages, *args, **kwargs):
            self.calls += 1
            request = json.loads(messages[-1]["content"])
            assessed = [
                ClaimCheck(
                    id=u["id"],
                    verdict="supported",
                    support=[
                        {
                            "source_id": "scene",
                            "quote": "Nobody knows whether visitors arrived.",
                        }
                    ],
                    reason="Faulty judgment, independently rejected.",
                )
                for u in request["required_claims"]
            ]
            if self.calls == 1:
                return (
                    ResponseReview(issues=[], revision=None, claim_checks=assessed),
                    {},
                )
            return (
                ResponseReview(
                    issues=[
                        {
                            "kind": "continuity",
                            "quote": '"No visitors."',
                            "reason": "Absence is unknown.",
                        }
                    ],
                    claim_checks=assessed,
                    revision=NarrationAnswer(
                        narration='"I don\'t know." The stranger adjusts his hat.'
                    ),
                ),
                {},
            )

    answer, trace = review_response(
        context, NarrationAnswer(narration='"No visitors."'), Repair(), OPTIONS
    )
    assert answer.direct_answer == "Nobody knows whether visitors arrived."
    assert "No visitors" not in answer.narration
    assert trace["status"] == "guarded" and len(trace["attempts"]) == 2


def test_atomic_uncertainty_does_not_whitelist_embedded_events_or_player_speech():
    from story_copilot.response_grounding import literal_checks

    for text in [
        '"I do not know why the stranger left."',
        'Ivo says, "I do not know."',
    ]:
        units = claim_units(body(), NarrationAnswer(narration=text))
        assert not literal_checks(units, [])
