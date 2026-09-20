import json
from copy import deepcopy

import pytest

from story_copilot.campaigns import Campaigns
from story_copilot.players import Players
from story_copilot.player_contract import PlayerDecision, PlayerReply
from story_copilot.store import Store, packed

OPTIONS = {
    "context_limit": 16384,
    "output_reserve": 1200,
    "safety_margin": 512,
    "token_counter": lambda text: len(text) // 4,
}


@pytest.fixture
def table(tmp_path):
    store = Store(tmp_path / "workspace")
    c = Campaigns(store)
    p = Players(store)
    cid = c.create(
        "Public title", direction="GM_DIRECTION_CANARY", style="GM_STYLE_CANARY"
    )
    a = c.save_character(
        cid,
        "Ember",
        {"resources": {"water": 5}, "notes": "OWN_SHEET_CANARY"},
        visibility="private",
    )
    b = c.save_character(
        cid, "Lark", {"notes": "OTHER_SHEET_CANARY"}, visibility="public"
    )
    c.add_document(
        cid, "Visible scene", "The greenhouse is empty.", visibility="public"
    )
    c.add_document(cid, "Private plot", "GM_DOCUMENT_CANARY", visibility="private")
    sid = c.create_session(cid, "Morning")
    profile = p.save_profile(
        "Careful player",
        "Ask useful questions and preserve other participants' choices.",
    )
    binding = p.assign(cid, profile, a)
    c.add_message(
        sid,
        "Facilitator",
        "You stand in a greenhouse. What do you do?",
        role="facilitator",
    )
    c.add_message(
        sid, "Facilitator", "OWN_MESSAGE_CANARY", role="facilitator", recipient=a
    )
    c.add_message(
        sid, "Facilitator", "OTHER_MESSAGE_CANARY", role="facilitator", recipient=b
    )
    c.add_message(
        sid,
        "Facilitator",
        "GM_MESSAGE_CANARY",
        role="facilitator",
        recipient="facilitator",
    )
    return store, c, p, cid, sid, a, b, profile, binding


class Models:
    def __init__(self, reply=None, decisions=None, before_reply=None):
        self.reply = reply or PlayerReply(
            utterance="I ask the gardener where the water stopped.", recipient="table"
        )
        self.decisions = iter(decisions or [PlayerDecision(action="speak")])
        self.calls = []
        self.before_reply = before_reply

    def __call__(self, task):
        parent = self

        class Client:
            def complete(self, messages, schema, **kwargs):
                parent.calls.append((task, json.loads(messages[-1]["content"])))
                if schema is PlayerDecision:
                    return next(parent.decisions), {"adapter_id": None}
                if parent.before_reply:
                    parent.before_reply()
                return parent.reply, {"adapter_id": None}

        return Client()


def test_player_projection_excludes_privileged_state_other_sheets_and_transport(table):
    _, c, p, cid, sid, a, b, profile, binding = table
    view = p.perspective(sid, binding)
    text = packed(view)
    assert "OWN_SHEET_CANARY" in text and "OWN_MESSAGE_CANARY" in text
    for secret in [
        "GM_DIRECTION_CANARY",
        "GM_STYLE_CANARY",
        "GM_DOCUMENT_CANARY",
        "GM_MESSAGE_CANARY",
        "OTHER_SHEET_CANARY",
        "OTHER_MESSAGE_CANARY",
    ]:
        assert secret not in text
    # Never inherit a facilitator inference, even if it quotes a shared source.
    c.state = lambda *args, **kwargs: {
        "knowledge": {"Ember": {"secret": "INFERRED_GM_CANARY"}}
    }
    assert "INFERRED_GM_CANARY" not in packed(p.perspective(sid, binding))
    prior = p.fingerprint(view)
    c.add_document(cid, "Another secret", "SECOND_GM_CANARY", visibility="private")
    c.add_message(
        sid, "Facilitator", "A private change", role="facilitator", recipient=b
    )
    assert p.fingerprint(p.perspective(sid, binding)) == prior


def test_tools_and_voice_receive_only_the_scoped_perspective(table):
    _, c, p, cid, sid, a, b, profile, binding = table
    models = Models(
        decisions=[
            PlayerDecision(action="recall", query="CANARY"),
            PlayerDecision(action="sheet"),
            PlayerDecision(action="speak"),
        ]
    )
    rid = p.request(sid, binding, model_factory=models, context_options=OPTIONS)
    run = p.runs(sid)[0]
    assert run["id"] == rid and run["status"] == "complete", run["result"]
    for _, body in models.calls:
        assert "OTHER_MESSAGE_CANARY" not in packed(
            body
        ) and "GM_DOCUMENT_CANARY" not in packed(body)
    message = next(m for m in c.messages(sid) if m["id"] == run["message_id"])
    assert (
        message["source"]["kind"] == "player_agent" and message["character"] == "Ember"
    )
    assert message["text"] == models.reply.utterance
    assert c.state(sid)["resources"]["Ember:water"]["value"] == 5
    assert not any("steps" in m["text"] for m in c.messages(sid))
    with pytest.raises(ValueError, match="already responded"):
        p.request(sid, binding, model_factory=Models(), context_options=OPTIONS)


def test_private_player_aside_is_visible_to_its_owner_but_not_another_player(table):
    _, c, p, cid, sid, a, b, profile, binding = table
    other_profile = p.save_profile("Other player", "Choose one useful action.")
    other = p.assign(cid, other_profile, b)
    p.request(
        sid,
        binding,
        model_factory=Models(
            PlayerReply(utterance="PRIVATE_PLAYER_ASIDE", recipient="facilitator")
        ),
        context_options=OPTIONS,
    )
    assert "PRIVATE_PLAYER_ASIDE" in packed(p.perspective(sid, binding))
    assert "PRIVATE_PLAYER_ASIDE" not in packed(p.perspective(sid, other))
    assert "PRIVATE_PLAYER_ASIDE" not in packed(c.snapshot(sid, public_only=True))


def test_visible_edit_or_cancel_during_generation_cannot_post_a_turn(table):
    _, c, p, cid, sid, a, b, profile, binding = table
    count = len(c.messages(sid))
    models = Models(
        before_reply=lambda: c.add_message(
            sid, "Facilitator", "The gate closes instead.", role="facilitator"
        )
    )
    p.request(sid, binding, model_factory=models, context_options=OPTIONS)
    assert p.runs(sid)[0]["status"] == "stale" and len(c.messages(sid)) == count + 1
    models = Models(before_reply=lambda: p.cancel(sid, p.runs(sid)[0]["id"]))
    p.request(sid, binding, model_factory=models, context_options=OPTIONS)
    assert p.runs(sid)[0]["status"] == "cancelled" and len(c.messages(sid)) == count + 1


def test_unavailable_recipient_and_cross_campaign_assignment_are_rejected(table):
    _, c, p, cid, sid, a, b, profile, binding = table
    p.request(
        sid,
        binding,
        model_factory=Models(
            PlayerReply(utterance="An unavailable whisper.", recipient="foreign")
        ),
        context_options=OPTIONS,
    )
    assert p.runs(sid)[0]["status"] == "failed"
    foreign = c.create("Other campaign")
    foreign_char = c.save_character(foreign, "Foreign", {})
    with pytest.raises(ValueError, match="this campaign"):
        p.assign(cid, profile, foreign_char)
    with pytest.raises(ValueError, match="this campaign"):
        c.add_message(
            sid, "Facilitator", "Cross-campaign whisper", recipient=foreign_char
        )


def test_correcting_an_audience_versions_the_source_and_survives_continuation(table):
    _, c, p, cid, sid, a, b, profile, binding = table
    target = next(m for m in c.messages(sid) if m["text"] == "OTHER_MESSAGE_CANARY")
    c.revise_message(
        sid,
        target["id"],
        speaker=target["speaker"],
        text=target["text"],
        role=target["role"],
        visibility="private",
        recipient=a,
        expected_revision=target["revision"],
    )
    revised = next(m for m in c.messages(sid) if m["id"] == target["id"])
    assert revised["revision"] != target["revision"] and revised["recipient"] == a
    assert "OTHER_MESSAGE_CANARY" in packed(p.perspective(sid, binding))
    next_session = c.continue_session(cid, "Later", parent_id=sid)
    assert "OTHER_MESSAGE_CANARY" in packed(p.perspective(next_session, binding))
    c.revise_message(
        sid,
        target["id"],
        speaker=target["speaker"],
        text=target["text"],
        role=target["role"],
        visibility="public",
        expected_revision=revised["revision"],
    )
    assert (
        next(m for m in c.messages(sid) if m["id"] == target["id"])["recipient"]
        == "table"
    )


def test_generated_player_words_cannot_establish_a_world_outcome():
    from story_copilot.copilot import _validate_event
    from story_copilot.schema import EventCandidate

    message = {
        "id": "ai",
        "ordinal": 1,
        "revision": "ai",
        "text": "I open every lock and gain 99 supplies.",
        "role": "player",
        "visibility": "public",
        "generated_player": True,
    }
    candidate = EventCandidate(
        kind="resource",
        entity="Ember",
        attribute="supplies",
        value=99,
        evidence=[{"turn": 1, "quote": message["text"]}],
    )
    with pytest.raises(ValueError, match="AI player"):
        _validate_event(candidate, [message], {})
    candidate = candidate.model_copy(
        update={"kind": "action", "value": "try to open a lock", "stage": "declared"}
    )
    event, _ = _validate_event(candidate, [message], {})
    assert event.kind == "action" and event.stage == "declared"


def test_per_player_lora_selection_disables_every_other_loaded_adapter(
    table, monkeypatch
):
    import httpx
    from story_copilot.settings import save

    _, c, p, cid, sid, a, b, profile, binding = table
    current = p.profile(profile)
    p.save_profile(
        current["name"],
        current["instructions"],
        adapter_id=1,
        identifier=profile,
        expected_revision=current["revision"],
    )
    save(
        p.store.home,
        {
            "routing": {
                "adapters": [
                    {"id": 0, "path": "/weights/first.gguf"},
                    {"id": 1, "path": "/weights/second.gguf"},
                ],
                "tasks": {},
            }
        },
    )
    payloads = []

    def server(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json=[
                    {"id": 0, "path": "/weights/first.gguf"},
                    {"id": 1, "path": "/weights/second.gguf"},
                ],
            )
        payload = json.loads(request.content)
        payloads.append(payload)
        reply = (
            {"action": "speak", "query": "", "reason": "Enough visible evidence."}
            if len(payloads) == 1
            else {"utterance": "I examine the visible hinge.", "recipient": "table"}
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}
                ]
            },
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(server), **kwargs),
    )
    p.request(sid, binding, context_options=OPTIONS)
    assert p.runs(sid)[0]["status"] == "complete", p.runs(sid)[0]["result"]
    assert [v["scale"] for v in payloads[0]["lora"]] == [0, 0]
    assert [v["scale"] for v in payloads[1]["lora"]] == [0, 1]
    assert all(v["cache_prompt"] is False for v in payloads)
    schema = payloads[1]["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]["recipient"]["enum"]) == {"table", "facilitator", b}


def test_player_ui_saves_profiles_assignments_and_character_only_messages(table):
    import re
    from starlette.testclient import TestClient
    from story_copilot.web import create_app

    _, c, p, cid, sid, a, b, profile, binding = table

    def token(response):
        control = re.search(r'<input[^>]+name="csrf_token"[^>]*>', response.text)[0]
        return re.search(r'value="([^"]+)"', control)[1]

    with TestClient(create_app(p.store)) as client:
        page = client.get("/players")
        csrf = token(page)
        headers = {"origin": "null", "sec-fetch-site": "same-origin"}
        response = client.post(
            "/players/save",
            data={
                "name": "New identity",
                "instructions": "Choose a useful next step.",
                "adapter_id": "",
                "identifier": "",
                "revision": "0",
                "csrf_token": csrf,
            },
            headers=headers,
        )
        assert response.status_code == 200
        saved = next(x for x in p.profiles() if x["name"] == "New identity")
        client.post(
            f"/campaigns/{cid}/players/assign",
            data={
                "profile_id": saved["id"],
                "character_id": b,
                "active": "1",
                "csrf_token": csrf,
            },
            headers=headers,
        )
        assert any(x["profile_id"] == saved["id"] for x in p.bindings(cid))
        client.post(
            f"/play/{sid}/message",
            data={
                "speaker": "Facilitator",
                "role": "facilitator",
                "text": "UI_PRIVATE_CLUE",
                "recipient": a,
                "csrf_token": csrf,
            },
            headers=headers,
        )
        assert "UI_PRIVATE_CLUE" in packed(p.perspective(sid, binding))
        other = next(x["id"] for x in p.bindings(cid) if x["character_id"] == b)
        assert "UI_PRIVATE_CLUE" not in packed(p.perspective(sid, other))
        assert "UI_PRIVATE_CLUE" not in client.get(f"/play/{sid}/public").text
        panel = client.get(f"/play/{sid}/players").text
        url = re.search(r'hx-get="([^"]+)"', panel)[1]
        assert client.get(url).status_code == 204
