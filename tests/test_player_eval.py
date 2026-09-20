from story_copilot.player_contract import PlayerDecision, PlayerReply
from story_copilot.settings import save
from story_copilot.store import Store
from story_copilot.player_eval import evaluate


def test_original_player_workflow_checks_cover_routing_privacy_staleness_and_duplicates(
    tmp_path, monkeypatch
):
    store = Store(tmp_path / "settings")
    save(store.home, {"routing": {"adapters": [{"id": 0}, {"id": 1}], "tasks": {}}})

    class Model:
        def __init__(self, *, task, configuration):
            self.task = task
            self.configuration = configuration

        def complete(self, messages, schema, **kwargs):
            answer = (
                PlayerDecision(
                    action="speak", query="", reason="Visible context suffices."
                )
                if self.task == "auditor"
                else PlayerReply(
                    utterance="I inspect the handwheel without turning it.",
                    recipient="table",
                )
            )
            return answer, {
                "adapter_id": self.configuration["routing"]["tasks"].get("player")
                if self.task == "player"
                else None
            }

    monkeypatch.setattr("story_copilot.player_eval.LocalModel", Model)
    result = evaluate(
        store.home,
        tmp_path / "evaluation",
        {"curious": 0, "decisive": 1},
        progress=lambda _: None,
    )
    assert len(result["cases"]) == 6
    assert all(all(case["checks"].values()) for case in result["cases"])
    assert (tmp_path / "evaluation" / "review.html").is_file()
