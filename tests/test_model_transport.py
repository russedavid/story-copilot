import json

import httpx
import pytest

from story_copilot.model import LocalModel
from story_copilot.schema import NarrationAnswer
from story_copilot.settings import validate


def mock_server(monkeypatch, adapters, content='{"narration":"An optional reply."}'):
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=adapters)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10},
            },
        )

    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs),
    )
    return requests


def test_loaded_adapters_are_all_explicit_and_prompt_cache_is_off(monkeypatch):
    requests = mock_server(monkeypatch, [{"id": 0}, {"id": 1}])
    config = validate(
        {"routing": {"adapters": [{"id": 0}, {"id": 1}], "tasks": {"storyteller": 1}}}
    )
    model = LocalModel(task="storyteller", configuration=config)
    result, metrics = model.complete(
        [{"role": "user", "content": "A private scene."}], NarrationAnswer
    )
    payload = json.loads(requests[-1].content)
    assert payload["lora"] == [{"id": 0, "scale": 0}, {"id": 1, "scale": 1}]
    assert payload["cache_prompt"] is False and metrics["adapter_id"] == 1
    assert result.narration == "An optional reply."


def test_mismatched_inventory_fails_before_generation(monkeypatch):
    requests = mock_server(monkeypatch, [{"id": 0}])
    with pytest.raises(ValueError, match="inventory"):
        LocalModel(configuration=validate({})).complete([], NarrationAnswer)
    assert len(requests) == 1 and requests[0].method == "GET"


def test_compatible_server_receives_no_local_extensions_or_key_in_trace(monkeypatch):
    requests = mock_server(monkeypatch, [])
    monkeypatch.setenv("SYNTHETIC_TEST_KEY", "synthetic-credential-only")
    config = validate(
        {"backend": "chat-completions", "api_key_env": "SYNTHETIC_TEST_KEY"}
    )
    _, metrics = LocalModel(configuration=config).complete([], NarrationAnswer)
    assert len(requests) == 1 and requests[0].method == "POST"
    payload = json.loads(requests[0].content)
    assert not set(payload) & {
        "lora",
        "cache_prompt",
        "chat_template_kwargs",
        "top_k",
        "min_p",
        "repeat_penalty",
    }
    assert requests[0].headers["Authorization"] == "Bearer synthetic-credential-only"
    assert "synthetic-credential-only" not in json.dumps(metrics)


def test_ui_model_settings_post_persists_changes(tmp_path):
    import re
    from starlette.testclient import TestClient
    from story_copilot.store import Store
    from story_copilot.settings import load
    from story_copilot.web import create_app

    store = Store(tmp_path / "workspace")
    with TestClient(create_app(store)) as client:
        page = client.get("/settings").text
        token = re.search(r'<input[^>]+name="csrf_token"[^>]+value="([^"]+)"', page)
        if not token:
            control = re.search(r'<input[^>]+name="csrf_token"[^>]*>', page)[0]
            csrf = re.search(r'value="([^"]+)"', control)[1]
        else:
            csrf = token[1]
        response = client.post(
            "/settings",
            data={
                "csrf_token": csrf,
                "backend": "llama.cpp",
                "url": "http://localhost:8091/v1",
                "model": "saved-choice",
                "api_key_env": "",
                "context_limit": "16384",
                "output_reserve": "1800",
                "routing": '{"adapters":[],"tasks":{}}',
            },
            headers={"origin": "null", "sec-fetch-site": "same-origin"},
        )
        assert response.status_code == 200
        assert load(store.home)["model"] == "saved-choice"
        enabled = client.post("/settings", data={
            "csrf_token": csrf, "backend": "llama.cpp", "url": "http://localhost:8091/v1",
            "model": "saved-choice", "context_limit": "16384", "output_reserve": "1800",
            "routing": '{"adapters":[],"tasks":{}}', "planner_enabled": "on",
            "planner_url": "http://localhost:8093/v1", "planner_model": "learned-policy",
            "planner_context": "8192", "planner_backend": "llama.cpp",
            "planner_routing": '{"adapters":[{"id":0}],"tasks":{"planner":0}}',
        }, headers={"origin": "null", "sec-fetch-site": "same-origin"})
        assert enabled.status_code == 200
        assert load(store.home)["planner"]["routing"]["tasks"]["planner"] == 0
        assert 'learned-policy' in client.get("/settings").text



def test_generation_defaults_are_applied_by_event_kind_before_application_validation(
    monkeypatch,
):
    from story_copilot.schema import Extraction, Event

    content = json.dumps(
        {
            "events": [
                {
                    "kind": "action",
                    "entity": "Tess",
                    "attribute": "cross_bridge",
                    "value": "requested",
                    "evidence": [{"turn": 1, "quote": "Can I cross the bridge?"}],
                }
            ]
        }
    )
    mock_server(monkeypatch, [], content)
    result, metrics = LocalModel(configuration=validate({})).complete([], Extraction)
    assert result.events[0].stage == "declared"
    assert Event.model_validate(result.events[0].model_dump()).stage == "declared"
    assert metrics["generation_contract"] == "valid"


def test_learned_policy_preserves_free_key_order_but_still_validates_schema(monkeypatch):
    from story_copilot.rl_environment import EvidenceDecision
    original_client = httpx.Client
    requests = mock_server(monkeypatch, [], content='{"action":"respond","value":2,"sources":["current-note"]}')
    answer, metrics = LocalModel(task="planner", configuration=validate({})).complete(
        [], EvidenceDecision, constrain=False)
    assert json.loads(requests[-1].content)["response_format"] == {"type":"json_object"}
    assert answer.sources == ["current-note"] and metrics["response_format"] == "json_object"
    monkeypatch.setattr(httpx, "Client", original_client)
    mock_server(monkeypatch, [], content='{"action":"delete","value":2}')
    with pytest.raises(ValueError):
        LocalModel(task="planner", configuration=validate({})).complete([], EvidenceDecision, constrain=False)
