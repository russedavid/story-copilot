import json

import pytest

from story_copilot.decisions import Decision
from story_copilot.live_planner import LiveDecision, LivePlanner, source_view
from story_copilot.settings import validate


def brief(observations=()):
    return [{"role":"system","content":"original"},{"role":"user","content":json.dumps({
        "current_task":"How many charges remain?","available_characters":["Neri"],
        "current_context":{"dialogue":[{"id":"message","revision":"revision-not-a-citation","text":"Three remain."}],"state":{},"private_facilitator_direction":"Keep it brief."},
        "tool_results":list(observations),"rules_available":[]})}]


class Fallback:
    calls = 0
    def complete(self, messages, schema, **kwargs):
        self.calls += 1
        return Decision(action="respond",intent="state"), {"model":"shared"}


def test_dedicated_policy_preserves_live_dialogue_and_ignores_unverified_terminal_math():
    calls=[]
    class Model:
        def complete(self,messages,schema,**kwargs):
            calls.append(json.loads(json.dumps(messages)))
            return LiveDecision(action="respond",intent="state",value=999,reason="There are 999."), {"model":"small"}
    result,trace=LivePlanner(Model(),Fallback(),options={"token_counter":lambda _:1}).complete(brief(),Decision)
    initial=json.loads(calls[0][1]["content"])
    assert initial["visible_evidence"][0]["text"]=="Three remain."
    assert "revision-not-a-citation" not in json.dumps(calls)
    assert "999" not in result.reason and trace["terminal_proposal_not_applied"]["value"]==999
    assert trace["planner_route"]=="dedicated"


def test_planner_failure_falls_back_once_then_stays_on_the_shared_client():
    class Broken:
        calls=0
        def complete(self,*args,**kwargs):
            self.calls+=1
            raise ValueError("Endpoint unavailable")
    small,fallback=Broken(),Fallback();client=LivePlanner(small,fallback,options={"token_counter":lambda _:1})
    for _ in range(2):
        result,trace=client.complete(brief(),Decision,timeout=5)
        assert result.action=="respond" and trace["planner_route"]=="shared_fallback"
    assert small.calls==1 and fallback.calls==2


def test_dedicated_settings_validate_inventory_without_changing_default_or_auditor():
    settings=validate({"planner":{"model":"small","routing":{"adapters":[{"id":0}],"tasks":{"planner":0}}}})
    assert settings["planner"]["model"]=="small"
    assert validate({})["planner"] is None
    with pytest.raises(ValueError,match="independent"):
        validate({"routing":{"adapters":[{"id":0}],"tasks":{"auditor":0}}})
    with pytest.raises(ValueError,match="nested"):
        validate({"planner":{"planner":{}}})


def test_latest_facilitator_update_and_complete_audio_question_replace_older_trigger():
    from story_copilot.copilot import _trigger, _message_view
    messages = [dict(id="1", ordinal=1, speaker="P", role="player", text="How many charges?", visibility="public"),
                dict(id="2", ordinal=2, speaker="F", role="facilitator", text="Privately, you notice a scratch.", visibility="private")]
    assert _trigger(messages) == "F (facilitator): Privately, you notice a scratch."
    messages += [dict(id="3", ordinal=3, speaker="P", role="player", text="Can I afford it?", visibility="public", source={"chunk_id":"audio"}),
                 dict(id="4", ordinal=4, speaker="P", role="player", text="I have not activated it.", visibility="public", source={"chunk_id":"audio"})]
    assert _trigger([_message_view(m) for m in messages]) == "P (player): Can I afford it? I have not activated it."
    assert "scratch" not in _trigger(messages)


def test_context_uses_current_resources_without_mutating_starting_sheet():
    from story_copilot.context import pack_context
    state = {"entities":{"Neri":{"sheet":{"value":{"occupation":"mechanic","resources":{"cells":9}},"visibility":"public"}}},
             "resources":{"Neri:cells":{"value":2,"visibility":"public"}}}
    result = pack_context(turns=[], state=state, player_input="How many cells?", active_entities=["Neri"])
    assert "resources" not in result["body"]["state"]["entities"]["Neri"]["sheet"]["value"]
    assert result["body"]["state"]["resources"]["Neri:cells"]["value"] == 2
    assert state["entities"]["Neri"]["sheet"]["value"]["resources"]["cells"] == 9


def test_untrained_fiction_scope_uses_the_main_planner_without_a_small_model_call():
    class Unexpected:
        def complete(self, *args, **kwargs):
            raise AssertionError("Narrative task must retain the main planner")
    messages=brief();body=json.loads(messages[-1]["content"]);body["current_task"]="I ask the caretaker about her missing colleague.";messages[-1]["content"]=json.dumps(body)
    fallback=Fallback();_,trace=LivePlanner(Unexpected(),fallback).complete(messages,Decision)
    assert trace["planner_route"]=="shared_scope" and fallback.calls==1


def test_original_learned_contract_infers_intent_without_losing_terminal_sources():
    class Model:
        def complete(self,*args,**kwargs):
            assert kwargs["constrain"] is False
            return LiveDecision(action="respond",value=2,sources=["message"]), {}
    result,trace=LivePlanner(Model(),Fallback(),options={"token_counter":lambda _:1}).complete(brief(),Decision)
    assert result.intent=="state" and trace["intent_from_evidence_fields"]
    assert trace["terminal_proposal_not_applied"]["sources"]==["message"]
