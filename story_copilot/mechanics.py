"""Explicit numeric evidence and campaign-selected arithmetic; no implicit dice rules."""

import math
import re

from .profiles import calculate

_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?!\w|\.\d)")


def numeric_sources(messages, state, rules=()):
    """List supplied values with stable references, never inferred missing inputs."""
    sources = []
    for message in messages[-8:]:
        text = message["text"]
        for i, match in enumerate(_NUMBER.finditer(text)):
            value = float(match[0]) if "." in match[0] else int(match[0])
            sources.append(
                {
                    "id": f"message:{message.get('id', message.get('ordinal'))}:{i}",
                    "value": value,
                    "label": message.get("speaker", "Conversation"),
                    "quote": text[max(0, match.start() - 70) : match.end() + 70],
                }
            )
    player = next(
        (
            m
            for m in reversed(messages)
            if m.get("role") == "player"
            or m.get("speaker_mapping", {}).get("role") == "player"
        ),
        None,
    )
    actor = (
        (player.get("character") or player.get("speaker_mapping", {}).get("character"))
        if player
        else None
    )

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "resources":
                    continue  # Initial sheet resources may have changed during play.
                visit(
                    child, path + "/" + str(key).replace("~", "~0").replace("/", "~1")
                )
        elif type(value) in {int, float} and math.isfinite(value):
            sources.append(
                {
                    "id": "sheet:" + str(actor) + path,
                    "value": value,
                    "label": str(actor) + path,
                    "quote": str(value),
                }
            )

    if actor in state.get("entities", {}):
        visit(state["entities"][actor], "")
    for key, resource in state.get("resources", {}).items():
        value = resource.get("value")
        if (
            actor
            and key.startswith(actor + ":")
            and type(value) in {int, float}
            and math.isfinite(value)
        ):
            sources.append(
                {
                    "id": "resource:" + key,
                    "value": value,
                    "label": key,
                    "quote": str(value),
                }
            )
    for rule in rules:
        for i, match in enumerate(_NUMBER.finditer(rule["text"])):
            value = float(match[0]) if "." in match[0] else int(match[0])
            sources.append(
                {
                    "id": f"rule:{rule['id']}:{i}",
                    "value": value,
                    "label": rule.get("title", "Supplied rule"),
                    "quote": rule["text"],
                }
            )
    return sources[:200]


def grounded_calculation(call, profile, sources):
    if call is None:
        return None
    lookup = {s["id"]: s for s in sources}
    values, evidence = [], []
    for item in call.inputs:
        if (
            item.source not in lookup
            or type(item.value) not in {int, float}
            or lookup[item.source]["value"] != item.value
        ):
            raise ValueError(
                "Every calculation input must match an explicitly supplied numeric source."
            )
        values.append(item.value)
        evidence.append(lookup[item.source])
    result = calculate(profile, call.tool, values)
    return {**result, "evidence": evidence}
