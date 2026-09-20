import json

import pytest

from starlette.testclient import TestClient

from story_copilot.evaluate import render
from story_copilot.store import Store, digest
from story_copilot.web import create_app


def test_review_trace_links_work_without_exposing_neighboring_files(
    tmp_path, monkeypatch
):
    experiments = tmp_path / "experiments"
    report = experiments / "original"
    (report / "agent").mkdir(parents=True)
    (report / "agent" / "opening.json").write_text('{"trace":"original example"}')
    (report / "model-settings.json").write_text('{"not_a_trace":true}')
    result = {
        "cases": [
            {
                "id": "opening",
                "title": "Opening",
                "expect": "Leave the choice open.",
                "policies": {
                    "agent": {
                        "seconds": 1,
                        "calls": 2,
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "component_status": {"storyteller": "complete"},
                        "checks": {},
                        "suggestions": [
                            {
                                "kind": "narration",
                                "text": '<script>alert("source")</script>',
                            }
                        ],
                        "trace_file": "agent/opening.json",
                    }
                },
            }
        ]
    }
    (report / "result.json").write_text(json.dumps(result))
    render(result, report / "index.html")
    monkeypatch.setenv("STORY_EXPERIMENTS", str(experiments))
    key = digest("original/index.html")[:20]
    with TestClient(create_app(Store(tmp_path / "workspace"))) as c:
        index = c.get("/experiments")
        assert f"/experiments/{key}" in index.text
        response = c.get(f"/experiments/{key}")
        assert (
            "&lt;script&gt;" in response.text and "<script>alert" not in response.text
        )
        assert f'href="/experiments/{key}/trace/agent/opening.json"' in response.text
        assert c.get(f"/experiments/{key}/trace/agent/opening.json").json() == {
            "trace": "original example"
        }
        assert c.get(f"/experiments/{key}/trace/model-settings.json").status_code == 404
        assert (
            c.get(f"/experiments/{key}/trace/../model-settings.json").status_code == 404
        )


@pytest.mark.parametrize("metadata", ["report.json", "results.json"])
def test_player_and_toolkit_reviews_are_discoverable(tmp_path, monkeypatch, metadata):
    experiments = tmp_path / "experiments"
    report = experiments / "players"
    report.mkdir(parents=True)
    (report / "review.html").write_text("<h1>Response review</h1>")
    (report / metadata).write_text(json.dumps({"cases": [{"id": "original"}]}))
    monkeypatch.setenv("STORY_EXPERIMENTS", str(experiments))
    key = digest("players/review.html")[:20]
    with TestClient(create_app(Store(tmp_path / "workspace"))) as c:
        assert f"/experiments/{key}" in c.get("/experiments").text
        assert "Response review" in c.get(f"/experiments/{key}").text
        assert c.get(f"/experiments/{key}/trace/{metadata}").status_code == 404
