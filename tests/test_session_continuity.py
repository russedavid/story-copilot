"""Explicit campaign continuity and alternatives never multiply prior effects."""

import re

import pytest
from starlette.testclient import TestClient

from story_copilot.campaigns import Campaigns
from story_copilot.store import Store
from story_copilot.web import create_app


def setup(c):
    cid = c.create("A fictional coast")
    c.save_character(cid, "Jo", {"resources": {"HP": 11, "FOCUS": 60}})
    sid = c.create_session(cid, "First night", proactive=False)
    mid = c.add_message(
        sid,
        "Facilitator",
        "Jo loses two hit points and discovers a brass key beneath the chart table.",
        role="facilitator",
    )
    pid = c.add_proposal(
        sid,
        {
            "kind": "state",
            "title": "Cost and clue",
            "text": "Recorded fictional outcome",
            "visibility": "private",
            "payload": {
                "state_changes": [
                    {
                        "kind": "resource",
                        "entity": "Jo",
                        "attribute": "HP",
                        "delta": -2,
                    },
                    {
                        "kind": "knowledge",
                        "entity": "Jo",
                        "attribute": "key",
                        "value": "brass key beneath the chart table",
                    },
                ]
            },
            "evidence": [
                {
                    "message_id": mid,
                    "quote": "Jo loses two hit points and discovers a brass key beneath the chart table.",
                }
            ],
        },
    )
    c.decide(sid, pid, "accepted")
    return cid, sid


def test_continue_keeps_resources_knowledge_and_history_once_across_sessions(tmp_path):
    c = Campaigns(Store(tmp_path / "private"))
    cid, first = setup(c)
    second = c.continue_session(cid, "Second night")
    experiment = c.branch(first, "What if things went badly?")
    bad = c.add_proposal(
        experiment,
        {
            "kind": "state",
            "text": "Alternative damage",
            "payload": {
                "state_changes": [
                    {"kind": "resource", "entity": "Jo", "attribute": "HP", "delta": -5}
                ]
            },
        },
    )
    c.decide(experiment, bad, "accepted")
    assert c.current_session(cid)["id"] == second
    third = c.continue_session(cid, "Third night")
    assert c.session(third)["parent_id"] == second
    for sid in (first, second, third):
        state = c.state(sid)
        assert state["resources"]["Jo:HP"]["value"] == 9
        assert (
            state["knowledge"]["Jo"]["key"]["value"]
            == "brass key beneath the chart table"
        )
        assert len(state["applied_events"]) == 2
        assert len(c.messages(sid)) == 1
    assert c.state(experiment)["resources"]["Jo:HP"]["value"] == 4
    assert (
        c.session(third)["kind"] == "continue"
        and c.session(experiment)["kind"] == "branch"
    )
    assert c.session(third)["proactive"] == 0
    fresh = c.create_session(cid, "A fresh beginning")
    assert c.current_session(cid)["id"] == fresh
    assert not c.messages(fresh) and not c.state(fresh)["knowledge"]
    assert c.state(fresh)["resources"]["Jo:HP"]["value"] == 11
    assert c.state(third)["resources"]["Jo:HP"]["value"] == 9


def test_legacy_or_ambiguous_history_requires_an_explicit_choice(tmp_path):
    c = Campaigns(Store(tmp_path / "private"))
    cid, first = setup(c)
    alternative = c.branch(first, "Experiment")
    with c.store.db() as db:
        db.execute("DELETE FROM play_campaign_progress WHERE campaign_id=?", (cid,))
    with pytest.raises(ValueError, match="Select the session"):
        c.continue_session(cid, "Cannot guess")
    second = c.continue_session(cid, "Chosen continuation", parent_id=first)
    assert c.session(second)["parent_id"] == first
    other = c.create("Other campaign")
    with pytest.raises(ValueError, match="this campaign"):
        c.continue_session(other, "Wrong source", parent_id=alternative)


def test_web_choices_default_to_main_story_not_latest_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "story_copilot.copilot.make_copilot",
        lambda store: (lambda snapshot: snapshot, None),
    )
    app = create_app(Store(tmp_path / "private"))
    c = app.state.campaigns
    cid, first = setup(c)
    branch = c.branch(first, "Newer experiment")
    with TestClient(app) as client:
        response = client.get(f"/campaigns/{cid}")
        source = re.search(
            r'<select[^>]*name="parent_id".*?</select>', response.text, re.S
        )[0]
        selected = re.search(r"<option[^>]*selected[^>]*>", source)[0]
        assert first in selected and branch not in selected
        assert all(
            f'value="{kind}"' in response.text
            for kind in ["continue", "branch", "fresh"]
        )
        control = re.search(r'<input[^>]*name="csrf_token"[^>]*>', response.text)[0]
        csrf = re.search(r'value="([^"]+)"', control)[1]
        response = client.post(
            f"/campaigns/{cid}/session",
            data={
                "csrf_token": csrf,
                "title": "Second night",
                "mode": "continue",
                "parent_id": first,
            },
        )
        assert response.status_code == 200
        new = c.current_session(cid)
        assert new["kind"] == "continue" and new["parent_id"] == first
        assert c.state(new["id"])["resources"]["Jo:HP"]["value"] == 9
