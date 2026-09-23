"""Synthetic fixtures test human control, privacy and state continuity, not model quality."""

import json
import re
from concurrent.futures import ThreadPoolExecutor

import pytest
from starlette.testclient import TestClient

from story_copilot.campaigns import Campaigns, imported_text
from story_copilot.store import Store


@pytest.fixture
def table(tmp_path):
    service = Campaigns(Store(tmp_path / "private"))
    campaign = service.create("The tide", direction="Keep the lighthouse secret")
    session = service.create_session(campaign, "Night one")
    return service, campaign, session


def proposal(**change):
    return {
        "kind": "state",
        "title": "A consequence",
        "text": "The glass costs two HP.",
        "visibility": "public",
        "payload": {"state_changes": [change]},
    }


def test_model_outputs_remain_drafts_and_undo_replays_state(table):
    c, cid, sid = table
    c.save_character(cid, "Ada", {"resources": {"HP": 12}})
    pid = c.add_proposal(
        sid, proposal(kind="resource", entity="Ada", attribute="HP", delta=-2)
    )
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 12
    c.decide(sid, pid, "accepted")
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 10
    assert c.messages(sid) == []
    c.decide(sid, pid, "rejected")
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 12
    c.undo(sid)
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 10
    c.undo(sid)
    assert c.state(sid)["resources"]["Ada:HP"]["value"] == 12


def test_unknown_resource_and_acceptance_order(table):
    c, _, sid = table
    early = c.add_proposal(
        sid, proposal(kind="resource", entity="Ada", attribute="FOCUS", delta=-2)
    )
    late = c.add_proposal(
        sid, proposal(kind="resource", entity="Ada", attribute="FOCUS", value=50)
    )
    c.decide(sid, early, "accepted")
    assert c.state(sid)["resources"]["Ada:FOCUS"]["value"] is None
    assert c.state(sid)["resources"]["Ada:FOCUS"]["known_delta"] == -2
    c.decide(sid, late, "accepted")
    c.decide(sid, early, "accepted")
    assert c.state(sid)["resources"]["Ada:FOCUS"]["value"] == 48


def test_private_material_never_enters_public_snapshot(table):
    c, cid, sid = table
    c.add_document(cid, "Hidden", "LIGHTHOUSE_SECRET")
    c.save_character(
        cid, "Hidden monster", {"resources": {"HP": 90}}, visibility="private"
    )
    secret = c.add_message(sid, "Facilitator", "NPC_SECRET", visibility="private")
    c.add_message(sid, "Ada", "I go north.", role="player")
    p = c.add_proposal(
        sid,
        {
            "kind": "note",
            "title": "Secret",
            "text": "SECRET_DRAFT",
            "visibility": "private",
        },
    )
    c.decide(sid, p, "accepted")
    public = json.dumps(c.snapshot(sid, public_only=True))
    for text in [
        "LIGHTHOUSE_SECRET",
        "NPC_SECRET",
        "SECRET_DRAFT",
        "Hidden monster",
        "Keep the lighthouse secret",
    ]:
        assert text not in public
    with pytest.raises(ValueError, match="private"):
        c.add_proposal(
            sid,
            {
                "kind": "narration",
                "text": "A reveal",
                "visibility": "public",
                "evidence": [{"message_id": secret, "quote": "NPC_SECRET"}],
            },
        )


def test_audio_redelivery_is_idempotent_and_does_not_assume_mic_identity(table):
    c, cid, sid = table
    first = c.add_message(
        sid, "mic:0", "I open it.", source={"channel": "mic"}, external_id="chunk:0"
    )
    assert (
        c.add_message(
            sid, "mic:0", "I open it.", source={"channel": "mic"}, external_id="chunk:0"
        )
        == first
    )
    assert c.messages(sid)[0]["role"] == "unknown"
    with pytest.raises(ValueError, match="different content"):
        c.add_message(
            sid,
            "mic:0",
            "I close it.",
            source={"channel": "mic"},
            external_id="chunk:0",
        )
    char = c.save_character(cid, "Ada", {"resources": {}})
    c.map_participant(cid, "Alex", "mic:0", role="player", character_id=char)
    c.add_message(sid, "mic:0", "I search.")
    assert c.messages(sid)[1]["character"] == "Ada"
    assert c.messages(sid)[1]["role"] == "player"


def test_novelty_gate_keeps_roll_changes_questions_and_pending_replies(table):
    c, _, sid = table
    c.add_message(sid, "Ada", "I rolled 21.", role="player")
    first = c.context_hash(c.snapshot(sid))
    c.add_message(sid, "Ada", "I rolled 21.", role="player")
    assert c.context_hash(c.snapshot(sid)) == first
    c.add_message(sid, "Ada", "Um", role="player")
    assert c.context_hash(c.snapshot(sid)) == first
    c.add_message(sid, "Ada", "I rolled 22.", role="player")
    assert c.context_hash(c.snapshot(sid)) != first
    c.add_message(sid, "Facilitator", "Do you open the door?", role="facilitator")
    before = c.context_hash(c.snapshot(sid))
    c.add_message(sid, "Ada", "Yes", role="player")
    assert c.context_hash(c.snapshot(sid)) != before
    p = c.add_proposal(
        sid,
        proposal(kind="pending", entity="Ada", attribute="check", value="Observation"),
    )
    c.decide(sid, p, "accepted")
    before = c.context_hash(c.snapshot(sid))
    c.add_message(sid, "Ada", "Okay", role="player")
    assert c.context_hash(c.snapshot(sid)) != before


def test_run_claim_duplicate_suppression_staleness_and_failures(table):
    c, _, sid = table
    c.add_message(sid, "Ada", "I open it.")
    run, context = c.start_run(sid)
    assert c.start_run(sid) is None
    c.add_message(sid, "Ada", "Actually I wait.")
    assert (
        c.finish_run(
            run, {"suggestions": [{"kind": "narration", "text": "A door opens"}]}
        )
        == "stale"
    )
    assert c.proposals(sid) == []
    call = lambda ctx: {
        "suggestions": [{"kind": "question", "text": "What will you do?"}],
        "trace": {"synthetic": True},
    }
    assert c.generate(sid, call)
    assert c.generate(sid, call) is None
    assert len(c.proposals(sid)) == 1
    assert c.generate(sid, call, force=True)

    def broken(ctx):
        raise ValueError("Synthetic failure")

    c.generate(sid, broken, force=True)
    assert c.runs(sid)[0]["status"] == "failed"
    assert "Synthetic failure" in c.runs(sid)[0]["result"]["error"]


def test_branch_isolated_decisions_and_resolved_action_references(table):
    c, _, sid = table
    message = c.add_message(sid, "Facilitator", "Roll Observation.", role="facilitator")
    p = proposal(kind="pending", entity="Ada", attribute="check", value="Observation")
    p["evidence"] = [{"message_id": message, "quote": "Roll Observation"}]
    action = c.add_proposal(sid, p)
    c.decide(sid, action, "accepted")
    resolved = c.add_proposal(
        sid,
        proposal(
            kind="resolve",
            entity="Ada",
            attribute="check",
            value="passed",
            action_id=action + ":0",
        ),
    )
    c.decide(sid, resolved, "accepted")
    branch = c.branch(sid, "Other path")
    assert not c.state(branch)["pending"]
    assert len(c.state(branch)["resolutions"]) == 1
    assert c.proposals(branch)[0]["evidence"][0]["message_id"] != message
    c.undo(branch)
    assert len(c.state(branch)["pending"]) == 1
    assert not c.state(sid)["pending"]
    assert c.messages(branch)[0]["source"]["branched_from_message"] == message


def test_cross_campaign_ids_sheet_revision_and_unknown_resolution(table):
    c, cid, sid = table
    char = c.save_character(cid, "Ada", {"resources": {"HP": 12}})
    with pytest.raises(ValueError, match="Reload"):
        c.save_character(cid, "Ada", {}, character_id=char, expected_revision=0)
    with pytest.raises(ValueError, match="distinct"):
        c.save_character(cid, "ada", {})
    other = c.create("Other")
    with pytest.raises(ValueError, match="different campaign"):
        c.save_character(other, "Ada", {}, character_id=char)
    invalid = c.add_proposal(
        sid,
        proposal(
            kind="resolve",
            entity="Ada",
            attribute="check",
            value="passed",
            action_id="missing:0",
        ),
    )
    with pytest.raises(ValueError, match="no longer pending"):
        c.decide(sid, invalid, "accepted")
    with pytest.raises(ValueError, match="integer"):
        c.add_proposal(
            sid, proposal(kind="resource", entity="Ada", attribute="HP", value=True)
        )


def test_safe_imports_do_not_read_arbitrary_paths():
    assert imported_text("../../sheet.json", b'{"name": "Ada"}') == '{"name": "Ada"}'
    assert (
        imported_text("people.csv", b"name,role\nAda,player")
        == "name | role\nAda | player"
    )
    with pytest.raises(ValueError, match="Use a text"):
        imported_text("script.py", b"print(1)")
    with pytest.raises(ValueError, match="not a PDF"):
        imported_text("rules.pdf", b"not a pdf")
    with pytest.raises(ValueError, match="invalid"):
        imported_text("sheet.json", b"{broken")
    with pytest.raises(ValueError, match="binary"):
        imported_text("text.txt", b"hello\x00world")


def test_real_fast_html_forms_firefox_origin_and_public_view(tmp_path, monkeypatch):
    from story_copilot.campaign_web import register_campaign_routes
    from story_copilot.web import create_app

    store = Store(tmp_path / "private")
    monkeypatch.setattr(
        "story_copilot.copilot.make_copilot",
        lambda store: (lambda snapshot: snapshot, None),
    )
    app = create_app(store)
    # Parent app may already install routes; this helper supports isolated checks
    # while the integration patch is developed independently.
    if not hasattr(app.state, "campaigns"):
        from fasthtml.common import Hidden, Main, Title

        def csrf(session):
            session["csrf"] = "fixture-csrf"
            return Hidden(session["csrf"], name="csrf_token")

        def validate(request, session, submitted):
            if submitted != session.get("csrf") or not submitted:
                raise ValueError("Reload this page")
            if request.headers.get("sec-fetch-site") == "cross-site":
                raise ValueError("Use the local application")

        register_campaign_routes(
            app,
            store,
            page=lambda title, *body: (Title(title), Main(*body)),
            csrf=csrf,
            validate=validate,
            workers=ThreadPoolExecutor(max_workers=1),
        )
    with TestClient(app) as client:
        response = client.get("/campaigns")
        assert response.status_code == 200
        token = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
        if token is None:
            token = re.search(r'value="([^"]+)" name="csrf_token"', response.text)
        assert token, response.text
        headers = {"origin": "null", "sec-fetch-site": "same-origin"}
        form = {
            "csrf_token": token[1],
            "title": "Fixture campaign",
            "direction": "PRIVATE_DIRECTIONS",
        }
        response = client.post("/campaigns/create", data=form, headers=headers)
        assert response.status_code == 200
        cid = app.state.campaigns.campaigns()[0]["id"]
        assert "Fixture campaign" in response.text
        denied = client.post(
            "/campaigns/create",
            data={**form, "title": "Unwanted"},
            headers={
                "origin": "https://elsewhere.invalid",
                "sec-fetch-site": "cross-site",
            },
        )
        assert "Use the local application" in denied.text
        assert len(app.state.campaigns.campaigns()) == 1
        response = client.post(
            f"/campaigns/{cid}/session",
            data={"csrf_token": token[1], "title": "First"},
            headers=headers,
        )
        sid = app.state.campaigns.sessions(cid)[0]["id"]
        assert "Facilitator suggestions" in response.text
        response = client.post(
            f"/play/{sid}/message",
            data={
                "csrf_token": token[1],
                "speaker": "Facilitator",
                "text": "PRIVATE_NPC",
                "visibility": "private",
            },
            headers=headers,
        )
        assert "PRIVATE_NPC" in response.text
        public = client.get(f"/play/{sid}/public").text
        assert "PRIVATE_NPC" not in public and "PRIVATE_DIRECTIONS" not in public
        upload = client.post(
            f"/campaigns/{cid}/document",
            data={"csrf_token": token[1], "title": "Clue", "visibility": "public"},
            files={"upload": ("clue.txt", b"A red door.", "text/plain")},
            headers=headers,
        )
        assert "A red door." in upload.text
        assert "A red door." in client.get(f"/play/{sid}/public").text


def test_corrections_keep_original_invalidate_evidence_and_allow_audio_redelivery(
    table,
):
    c, cid, sid = table
    mid = c.add_message(
        sid, "Facilitator", "Lose 2 HP.", role="facilitator", external_id="chunk:1"
    )
    p = proposal(kind="resource", entity="Ada", attribute="HP", delta=-2)
    p["evidence"] = [{"message_id": mid, "quote": "Lose 2 HP."}]
    pid = c.add_proposal(sid, p)
    c.decide(sid, pid, "accepted")
    c.revise_message(
        sid,
        mid,
        text="Lose 1 HP.",
        speaker="Facilitator",
        role="facilitator",
        visibility="private",
        expected_revision=mid,
    )
    assert c.messages(sid)[0]["original"] == "Lose 2 HP."
    assert c.state(sid)["resources"] == {}
    assert c.proposals(sid)[0]["stale"]
    assert (
        c.add_message(
            sid, "Facilitator", "Lose 2 HP.", role="facilitator", external_id="chunk:1"
        )
        == mid
    )
    assert not c.snapshot(sid, public_only=True)["messages"]
    assert not c.snapshot(sid, public_only=True)["proposals"]
    with pytest.raises(ValueError, match="Source changed"):
        c.decide(sid, pid, "accepted")
    with pytest.raises(ValueError, match="Reload"):
        c.revise_message(
            sid,
            mid,
            text="Something stale",
            speaker="Facilitator",
            role="facilitator",
            expected_revision=mid,
        )


def test_real_context_packer_preserves_claims_knowledge_pending_and_sheet(table):
    from story_copilot.context import pack_context

    c, cid, sid = table
    c.save_character(cid, "Ada", {"resources": {"HP": 12}, "occupation": "Reporter"})
    for change in [
        dict(
            kind="knowledge",
            entity="Ada",
            attribute="key",
            value="Found at the lighthouse",
        ),
        dict(
            kind="claim",
            entity="Caretaker",
            attribute="statement",
            value="Nobody went upstairs",
        ),
        dict(kind="pending", entity="Ada", attribute="check", value="Observation"),
    ]:
        pid = c.add_proposal(sid, proposal(**change))
        c.decide(sid, pid, "accepted")
    c.add_message(
        sid, "Ada", "Does the caretaker know where the key came from?", role="player"
    )
    snapshot = c.snapshot(sid)
    packed = pack_context(
        turns=snapshot["messages"],
        state=snapshot["state"],
        active_entities=["Ada"],
        context_limit=16000,
    )
    assert packed["body"]["state"]["resources"]["Ada:HP"]["value"] == 12
    assert (
        packed["body"]["state"]["knowledge"]["Ada"]["key"]["value"]
        == "Found at the lighthouse"
    )
    assert (
        packed["body"]["historical_claims_and_hypotheses"][0]["value"]
        == "Nobody went upstairs"
    )
    assert packed["body"]["state"]["pending"]


def test_retracted_pending_action_invalidates_compound_resolution_and_branch(table):
    c, _, sid = table
    action = c.add_proposal(
        sid, proposal(kind="pending", entity="Ada", attribute="check", value="FOCUS")
    )
    c.decide(sid, action, "accepted")
    answer = proposal(
        kind="resolve",
        entity="Ada",
        attribute="check",
        value="failed",
        action_id=action + ":0",
    )
    answer["payload"]["state_changes"].append(
        {"kind": "resource", "entity": "Ada", "attribute": "FOCUS", "delta": -2}
    )
    consequence = c.add_proposal(sid, answer)
    c.decide(sid, consequence, "accepted")
    c.decide(sid, action, "pending")
    assert c.state(sid)["resources"] == {}
    assert consequence in c.state(sid)["stale_events"]
    branch = c.branch(sid, "No roll")
    assert c.state(branch)["resources"] == {}


def test_context_build_failure_has_a_navigable_trace_without_model_call(table):
    c, _, sid = table

    def oversized(snapshot):
        raise ValueError("Mandatory context exceeds the budget")

    def never(context):
        pytest.fail("The model must not run with oversized context")

    rid = c.generate(sid, never, build_context=oversized)
    assert c.runs(sid)[0]["id"] == rid
    assert c.runs(sid)[0]["status"] == "failed"
    assert "budget" in c.runs(sid)[0]["result"]["error"]


def test_suggestions_never_enter_public_record_even_with_legacy_acceptance(table):
    c, _, sid = table
    pid = c.add_proposal(
        sid, {"kind": "narration", "text": "PRIVATE_SUGGESTION", "visibility": "public"}
    )
    for accepted in [False, True]:
        if accepted:
            c.decide(
                sid, pid, "accepted"
            )  # Existing records remain readable after migration.
        with pytest.raises(ValueError, match="private"):
            c.publish(sid, pid, "Some wording")
        assert c.snapshot(sid, public_only=True)["proposals"] == []
        assert c.snapshot(sid, public_only=True)["publications"] == []
        assert "PRIVATE_SUGGESTION" not in json.dumps(c.snapshot(sid, public_only=True))
    assert c.messages(sid) == []


def test_reject_and_refresh_ui_is_private_and_has_no_acceptance_gate(table):
    from story_copilot.web import create_app

    c, _, sid = table
    c.set_proactive(sid, False)
    pid = c.add_proposal(
        sid,
        {
            "kind": "narration",
            "title": "Facilitator response",
            "text": "PRIVATE_POINT_IN_TIME",
        },
    )
    app = create_app(c.store)
    scheduled = []
    app.state.campaign_scheduler._submit = lambda session_id, force: scheduled.append(
        (session_id, force)
    )
    with TestClient(app) as client:
        page = client.get(f"/play/{sid}").text
        assert "Reject and refresh" in page and "Accept suggestion" not in page
        assert "Publish wording" not in page and "Share selected wording" not in page
        token = re.search(r'value="([^"]+)" name="csrf_token"', page)[1]
        headers = {"origin": "null", "sec-fetch-site": "same-origin"}
        denied = client.post(
            f"/play/{sid}/proposal/{pid}",
            data={"csrf_token": token, "action": "accepted"},
            headers=headers,
        )
        assert "private guidance" in denied.text
        client.post(
            f"/play/{sid}/proposal/{pid}",
            data={"csrf_token": token, "action": "rejected"},
            headers=headers,
        )
        assert scheduled == [
            (sid, True)
        ]  # Refresh works with automatic suggestions paused.
        assert c.proposals(sid)[0]["status"] == "rejected"
        assert c.messages(sid) == []
        assert "PRIVATE_POINT_IN_TIME" not in client.get(f"/play/{sid}/public").text
        denied = client.post(
            f"/play/{sid}/proposal/{pid}/publish",
            data={"csrf_token": token, "text": "PRIVATE_POINT_IN_TIME"},
            headers=headers,
        )
        assert "cannot be published" in denied.text
        assert not c.publications(sid)


def test_working_state_panel_refreshes_independently_of_suggestion_clicks(table):
    from story_copilot.web import create_app
    from story_copilot.store import digest, packed

    c, cid, sid = table
    c.save_character(cid, "Ada", {"resources": {"HP": 12}})
    prior = digest(packed(c.state(sid)))
    with TestClient(create_app(c.store)) as client:
        assert client.get(f"/play/{sid}/state?version={prior}").status_code == 204
        character = c.characters(cid)[0]
        c.save_character(
            cid, "Ada", {"resources": {"HP": 14}}, character_id=character["id"]
        )
        updated = client.get(f"/play/{sid}/state?version={prior}")
        assert updated.status_code == 200 and 'id="working-game-state"' in updated.text
        assert "every 5s" in updated.text


def test_interrupted_run_recovery_keeps_trace_and_allows_fresh_request(table):
    c, _, sid = table
    run, context = c.start_run(sid)
    assert c.start_run(sid) is None
    c.recover_run(sid, run)
    assert c.runs(sid)[0]["status"] == "failed"
    assert c.runs(sid)[0]["request"] == context
    assert c.start_run(sid) is not None
    with pytest.raises(ValueError, match="running"):
        c.recover_run(sid, run)


def test_recovery_form_unblocks_interrupted_runs_but_refuses_an_active_worker(table):
    from story_copilot.web import create_app

    c, _, sid = table
    c.set_proactive(sid, False)
    run, snapshot = c.start_run(sid)
    app = create_app(c.store)
    with TestClient(app) as client:
        page = client.get(f"/play/{sid}").text
        token = re.search(r'value="([^"]+)" name="csrf_token"', page)[1]
        headers = {"origin": "null", "sec-fetch-site": "same-origin"}
        app.state.campaign_scheduler.is_pending = lambda session_id: True
        client.post(
            f"/play/{sid}/trace/{run}/recover",
            data={"csrf_token": token},
            headers=headers,
        )
        assert c.runs(sid)[0]["status"] == "running"
        app.state.campaign_scheduler.is_pending = lambda session_id: False
        client.post(
            f"/play/{sid}/trace/{run}/recover",
            data={"csrf_token": token},
            headers=headers,
        )
        recovered = next(row for row in c.runs(sid) if row["id"] == run)
        assert recovered["status"] == "failed" and recovered["request"] == snapshot
        assert c.start_run(sid) is not None
