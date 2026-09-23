"""Keep constrained JSON generation aligned with the training serialization."""

import copy


WIRE_SCHEMA_VERSION = 5


def wire_schema(model):
    schema = copy.deepcopy(model.model_json_schema())
    explicit = model.__name__ in {
        "Extraction",
        "GenerationExtraction",
        "Decision",
        "NarrationAnswer",
        "DirectAnswer",
        "SceneAnswer",
        "ResponseReview",
        "CompactResponseReview",
        "PlayerReply",
        "ScopedPlayerReply",
        "PlayerDecision",
    }

    def visit(value):
        if isinstance(value, dict):
            if "properties" in value:
                value["properties"] = dict(sorted(value["properties"].items()))
                if explicit:
                    # Explicit fields avoid ambiguous event defaults and prevent
                    # a model from omitting the answer slot to end a response early.
                    value["required"] = list(value["properties"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    return schema
