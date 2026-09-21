"""Explicit numeric evidence and campaign-selected arithmetic; no implicit dice rules."""

import math
import re
from copy import deepcopy

from .profiles import calculate

_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?!\w|\.\d)")

_PAST = re.compile(r"\b(?:obsolete|superseded|formerly|previously|(?:old|previous|former)\s+(?:cost|price|balance|total|amount|requirement|rate|value|limit))\b", re.I)
_CURRENT = re.compile(r"\b(?:now|current(?:ly)?|revised|updated|new\s+(?:cost|price|balance|total|amount|value)|instead)\b", re.I)


def number_status(text, match):
    """Conservative explicit obsolescence markers; never infer an unstated value."""
    prefix = re.split(r"[.!?;\n]", text[:match.start()])[-1]
    if re.search(r"\b(?:not|instead of|rather than)\s*$", prefix, re.I):
        return "superseded"
    past = [m.start() for m in _PAST.finditer(prefix)]
    current = [m.start() for m in _CURRENT.finditer(prefix)]
    return "superseded" if past and max(past) > max(current, default=-1) else "supplied"


def calculation_profile(profile):
    """Basic arithmetic is available independently of any particular rule system."""
    profile = deepcopy(profile or {"tools": []})
    existing = {t["name"] for t in profile.get("tools", [])}
    basic = [
        ("arithmetic_sum", "sum", "Sum supplied values."),
        ("arithmetic_difference", "difference", "First supplied value minus the second."),
        ("arithmetic_product", "product", "Multiply supplied values."),
        ("arithmetic_quotient", "quotient", "Divide the first supplied value by the second."),
        ("arithmetic_at_least", "greater_equal", "Whether the first supplied value is at least the second."),
        ("balance_after_cost", "spend", "For a cited rule requiring sufficient resources and spending nothing when unaffordable: inputs are current balance, then cost. Return permission and resulting balance. Do not use for rules allowing debt or partial spending."),
    ]
    profile["tools"] = list(profile.get("tools", [])) + [
        {"name": name, "operation": operation, "description": description}
        for name, operation, description in basic if name not in existing
    ]
    # Preserve the campaign's configured tools when its explicit limit is full.
    profile["tools"] = profile["tools"][:16]
    return profile


def numeric_sources(messages, state, rules=(), *, actors=None):
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
                    "status": number_status(text, match),
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
                    "status": "sheet_value",
                }
            )

    if actor in state.get("entities", {}):
        visit(state["entities"][actor], "")
    subjects = set(actors) if actors is not None else ({actor} if actor else set())
    for key, resource in state.get("resources", {}).items():
        value = resource.get("value")
        if (
            any(key.startswith(name + ":") for name in subjects)
            and type(value) in {int, float}
            and math.isfinite(value)
        ):
            sources.append(
                {
                    "id": "resource:" + key,
                    "value": value,
                    "label": key,
                    "quote": str(value),
                    "status": "current_resource",
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
                    "rule_id": rule["id"],
                    "quote": rule["text"],
                    "status": number_status(rule["text"], match),
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
        if lookup[item.source].get("status") == "superseded" and getattr(call, "scope", "current") != "historical":
            raise ValueError("A current ruling cannot use an explicitly superseded or negated numeric input.")
        evidence.append(lookup[item.source])
    result = calculate(profile, call.tool, values)
    scope = getattr(call, "scope", "current")
    if scope == "historical" and result["operation"] == "spend":
        raise ValueError("Historical numbers cannot authorize a current expenditure.")
    return {**result, "evidence": evidence, "scope": scope}
