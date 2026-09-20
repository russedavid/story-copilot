import pytest

from story_copilot.routing import selection


def test_each_request_explicitly_disables_other_adapters():
    config = {
        "adapters": [{"id": 0}, {"id": 1}, {"id": 2}],
        "tasks": {"classifier": 0, "storyteller": 1, "rules": 2},
    }
    first, _ = selection("classifier", config)
    second, _ = selection("rules", config)
    base, _ = selection("auditor", config)
    assert [x["scale"] for x in first] == [1, 0, 0]
    assert [x["scale"] for x in second] == [0, 0, 1]
    assert [x["scale"] for x in base] == [0, 0, 0]
    assert config["tasks"]["classifier"] == 0


def test_bad_inventory_or_unknown_task_fails_before_inference():
    with pytest.raises(ValueError, match="inventory"):
        selection("classifier", {"adapters": [{"id": 2}], "tasks": {}})
    with pytest.raises(ValueError, match="unavailable"):
        selection("rules", {"adapters": [], "tasks": {"rules": 1}})
    with pytest.raises(ValueError, match="independent"):
        selection("auditor", {"adapters": [{"id": 0}], "tasks": {"auditor": 0}})


def test_serving_preserves_application_context_and_exposes_measured_split():
    from story_copilot.serve_models import command

    args = command(
        "/runtime", "/base.gguf", [], layout="split", context=131072, tensor_split="4,1"
    )
    assert args[args.index("--tensor-split") + 1] == "4,1"
    assert args[args.index("--ctx-size") + 1] == "131072"
    assert "--no-context-shift" in args
    for value in ["-1,2", "1,nan", "inf,1", "1,2,3", "x,y", "0,1"]:
        with pytest.raises(ValueError):
            command("r", "b", [], layout="split", tensor_split=value)
    with pytest.raises(ValueError, match="requires"):
        command("r", "b", [], tensor_split="4,1")
