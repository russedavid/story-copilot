"""Keep constrained JSON generation aligned with the training serialization."""

import copy


WIRE_SCHEMA_VERSION = 2


def wire_schema(model):
    schema = copy.deepcopy(model.model_json_schema())
    explicit = model.__name__ == "Extraction"

    def visit(value):
        if isinstance(value, dict):
            if "properties" in value:
                value["properties"] = dict(sorted(value["properties"].items()))
                if explicit:
                    # All event fields must be present, with null where unused.
                    # Missing delta/stage/resolves silently changes event meaning.
                    value["required"] = list(value["properties"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    return schema
