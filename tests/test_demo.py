"""The fictional example is authored UI content, not evaluation ground truth."""

from concurrent.futures import ThreadPoolExecutor
import json
import re

from starlette.testclient import TestClient

from story_copilot.campaigns import Campaigns
from story_copilot.demo import DISCLOSURE, ensure_demo
from story_copilot.store import Store
from story_copilot.web import create_app


def token(page):
    control = re.search(r'<input[^>]+name="csrf_token"[^>]*>', page).group()
    return re.search(r'value="([^"]+)"', control)[1]


def test_original_demo_is_complete_private_and_idempotent(tmp_path):
    c = Campaigns(Store(tmp_path / "private"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: ensure_demo(c), range(2)))
    assert sum(r["created_now"] for r in results) == 1
    cid, sid = results[0]["campaign_id"], results[0]["session_id"]
    assert len(c.campaigns()) == 1 and len(c.sessions(cid)) == 1
    assert len(c.characters(cid)) == 2 and len(c.participants(cid)) == 3
    assert c.session(sid)["proactive"] == 0 and c.runs(sid) == []
    assert c.current_session(cid)["id"] == sid
    assert len([m for m in c.messages(sid) if m["role"] == "player"]) == 3
    assert all(
        m["source"]["not_real_play"] and m["source"]["not_evaluation_gold"]
        for m in c.messages(sid)
    )
    public = json.dumps(c.snapshot(sid, public_only=True))
    assert "Mara quietly disconnected the main relay herself" not in public
    assert DISCLOSURE in public
    c.update(cid, title="My own continuation", direction="A new direction")
    c.add_message(sid, "Ari", "I inspect the windows.", role="player")
    c.set_proactive(sid, True)
    assert not ensure_demo(c)["created_now"]
    assert c.campaign(cid)["title"] == "My own continuation"
    assert len(c.messages(sid)) == 5 and c.session(sid)["proactive"] == 1


def test_try_example_waits_for_explicit_request_with_offline_generator(
    tmp_path, monkeypatch
):
    calls = []

    def generate(context):
        calls.append(context)
        return {
            "suggestions": [
                {
                    "kind": "question",
                    "text": "Synthetic test reply",
                    "visibility": "private",
                }
            ],
            "trace": {"synthetic_test": True},
        }

    monkeypatch.setattr(
        "story_copilot.copilot.make_copilot",
        lambda store: (lambda snapshot: snapshot, generate),
    )
    app = create_app(Store(tmp_path / "private"))
    with TestClient(app) as client:
        page = client.get("/campaigns")
        assert "Try an example" in page.text
        csrf = token(page.text)
        headers = {"origin": "null", "sec-fetch-site": "same-origin"}
        response = client.post(
            "/campaigns/example", data={"csrf_token": csrf}, headers=headers
        )
        c = app.state.campaigns
        demo = ensure_demo(c)
        sid = demo["session_id"]
        assert "not recorded play or an evaluation gold answer" in response.text
        assert not calls and not c.runs(sid)
        client.post("/campaigns/example", data={"csrf_token": csrf}, headers=headers)
        assert len(c.campaigns()) == 1 and not calls
        client.post(f"/play/{sid}/request", data={"csrf_token": csrf}, headers=headers)
        assert app.state.campaign_scheduler.wait_idle()
        assert len(calls) == 1 and c.runs(sid)[0]["status"] == "complete"
        assert c.session(sid)["proactive"] == 0
