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
