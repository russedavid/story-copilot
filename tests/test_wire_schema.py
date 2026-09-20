from story_copilot.schema import Extraction, Event, NarrationAnswer
from story_copilot.wire_schema import wire_schema


def test_extraction_transport_does_not_omit_fields_that_change_event_meaning():
    schema = wire_schema(Extraction)
    event = schema["$defs"]["EventCandidate"]
    assert set(event["required"]) == set(event["properties"])
    assert {"stage", "delta", "resolves", "supersedes", "visibility"} <= set(
        event["required"]
    )
    assert list(event["properties"]) == sorted(event["properties"])
    assert list(schema["$defs"]["Evidence"]["properties"]) == ["quote", "turn"]
    # Internal review/canonical APIs still have defaults; only the wire contract
    # requires explicit values. Their semantic validation stays in force.
    assert (
        Event(
            kind="fact",
            entity="door",
            attribute="locked",
            value=True,
            evidence=[{"turn": 1, "quote": "locked"}],
        ).stage
        == "established"
    )


def test_narration_transport_remains_compatible_with_minimal_training_targets():
    assert wire_schema(NarrationAnswer)["required"] == ["narration"]
