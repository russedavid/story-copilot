"""Inspect consequential prose claims without treating creative NPC dialogue as state.

This is a conservative review aid, not a natural-language truth prover. Exact
source quotations and coverage are checked in code; interpretation remains a
model judgment and is evaluated separately.
"""

from copy import deepcopy
import json
import re

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from .store import digest, packed, source_quote


class Support(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    quote: str = Field(min_length=1)


class ClaimCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    verdict: Literal["supported", "unsupported", "not_an_assertion"]
    support: list[Support] = Field(default_factory=list, max_length=6)
    reason: str = Field(min_length=1, max_length=500)


RULE_CLAIM = re.compile(
    r"\b(?:no|zero|none|without|\d+(?:\.\d+)?)\b.{0,65}\b(?:bonus|modifier|penalty|advantage|cost|damage|difficulty)\b"
    r"|\b(?:bonus|modifier|penalty|advantage|cost|damage|difficulty)\b.{0,65}\b(?:no|zero|none|\d+(?:\.\d+)?)\b",
    re.I,
)
NONASSERTION = re.compile(
    r"\?|\b(?:if|can|could|may|might|would|whether|unknown|unspecified|not specified|not established|not known|unclear)\b",
    re.I,
)
ZERO_RULE = re.compile(
    r"\b(?:no|zero|0)\s+(?:\w+\s+){0,2}(?:bonus|modifier|penalty|advantage|cost|damage)\b|\b(?:bonus|modifier|penalty|advantage|cost|damage)\b\s+(?:is|of|equals|=)\s*(?:zero|0|none)\b",
    re.I,
)


def text_fields(answer):
    return {
        "narration": answer.narration,
        "direct_answer": answer.direct_answer,
        "private_notes": answer.private_notes,
        **{f"questions/{i}": text for i, text in enumerate(answer.questions)},
        **{
            f"checks/{i}": check.text for i, check in enumerate(answer.requested_checks)
        },
    }


def players(body):
    names = {
        name
        for name, entity in body.get("state", {}).get("entities", {}).items()
        if isinstance(entity, dict) and "sheet" in entity
    }
    for document in body.get("documents", []):
        if document.get("id") == "table-control":
            try:
                control = json.loads(document["text"])
                names.update(control.get("player_control", {}))
                names.update(control.get("player_characters", []))
            except (ValueError, TypeError):
                pass
    return sorted(names)


def claim_units(body, answer):
    names = players(body)
    actor = re.compile(
        r"(?<!\w)(?:" + "|".join(map(re.escape, [*names, "you", "your"])) + r")(?!\w)",
        re.I,
    )
    units = []
    for field, text in text_fields(answer).items():
        # Keep the exact offsets. Commas and quotes stay together so qualifiers
        # such as 'looking concerned' cannot vanish from the sentence under review.
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+[\"'’”]*|$)", text):
            value = match[0].strip()
            if not value:
                continue
            kinds = []
            if actor.search(value):
                kinds.append("player")
            if RULE_CLAIM.search(value):
                kinds.append("rule")
            if kinds and not value.endswith("?"):
                subject = next(
                    (
                        name
                        for name in [*names, "you"]
                        if re.match(
                            r"^" + re.escape(name) + r"\b(?!\s*[:,])", value, re.I
                        )
                    ),
                    None,
                )
                units.append(
                    {
                        "id": digest(packed([field, match.start(), value]))[:16],
                        "field": field,
                        "text": value,
                        "kinds": kinds,
                        "can_be_nonassertion": bool(NONASSERTION.search(value))
                        and not bool(ZERO_RULE.search(value)),
                        "explicit_zero_rule": bool(ZERO_RULE.search(value)),
                        "literal_player_subject": subject,
                    }
                )
    return units


def sources(body):
    result = []
    for message in body.get("dialogue", []):
        if message.get("id") and message.get("text"):
            result.append(
                {
                    "id": message["id"],
                    "kind": "conversation",
                    "text": message["text"],
                    "speaker": message.get("speaker"),
                    "role": message.get("role"),
                    "visibility": message.get("visibility", "private"),
                }
            )
    for rule in body.get("rules", []):
        result.append(
            {
                "id": rule["id"],
                "kind": "rule",
                "text": rule["text"],
                "visibility": rule.get("visibility", "private"),
            }
        )
    for document in body.get("documents", []):
        metadata = document.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if metadata.get("kind") == "rules":
            result.append(
                {
                    "id": document["id"],
                    "kind": "rule",
                    "text": document["text"],
                    "visibility": document.get("visibility", "private"),
                }
            )
        # Response plans, generated feedback and previous drafts are not evidence.
    for category in ("resources", "facts", "knowledge"):
        for key, value in body.get("state", {}).get(category, {}).items():
            text = packed({key: value})
            if (
                category == "resources"
                and ":" in key
                and type(value.get("value")) in {int, float}
            ):
                actor, attribute = key.split(":", 1)
                text = f"{actor} has {value['value']} {attribute.replace('_', ' ')}."
            result.append(
                {
                    "id": f"state:{category}:{key}",
                    "kind": "observed_state",
                    "text": text,
                    "visibility": value.get(
                        "source_visibility", value.get("visibility", "private")
                    ),
                }
            )
    return list({item["id"]: item for item in result}.values())


def validate_checks(units, checks, evidence):
    by_id = {c.id: c for c in checks}
    if len(by_id) != len(checks) or set(by_id) != {u["id"] for u in units}:
        raise ValueError("Review must assess each required claim exactly once.")
    lookup = {s["id"]: s for s in evidence}
    unsupported = []
    for unit in units:
        check = by_id[unit["id"]]
        if check.verdict == "not_an_assertion":
            if not unit["can_be_nonassertion"]:
                raise ValueError(
                    "A declarative claim needs evidence or an unsupported verdict."
                )
            continue
        if check.verdict == "unsupported":
            unsupported.append(unit)
            continue
        if not check.support:
            raise ValueError("A supported claim needs an exact source quotation.")
        selected = []
        for support in check.support:
            if support.source_id not in lookup:
                raise ValueError("Claim support cites an unavailable source.")
            source = lookup[support.source_id]
            source_quote(source["text"], support.quote)
            selected.append(source)
        if "rule" in unit["kinds"] and not any(s["kind"] == "rule" for s in selected):
            raise ValueError(
                "A numerical rule assertion needs an applicable supplied rule, not an unanswered question or state record."
            )
        if unit.get("explicit_zero_rule") and not any(
            lookup[s.source_id]["kind"] == "rule" and ZERO_RULE.search(s.quote)
            for s in check.support
        ):
            raise ValueError(
                "Zero is a rule value requiring explicit support; an unspecified or absent rule cannot establish zero."
            )
        if unit.get("literal_player_subject"):
            words = lambda text: " ".join(re.findall(r"\w+", text.casefold()))
            statement = words(unit["text"])
            if not any(statement in words(s.quote) for s in check.support):
                raise ValueError(
                    "A declarative player statement must retain the wording of one supporting source; unsupported gestures or feelings cannot be added in paraphrase."
                )
        if unit["field"] == "narration" and all(
            s["visibility"] != "public" for s in selected
        ):
            raise ValueError(
                "Shared narration cannot expose a player fact supported only by private sources."
            )
    return unsupported


def guarded_answer(answer, units):
    """Retain unaffected fields; never return a known-unverified claim as a fallback."""
    result = deepcopy(answer)
    fields = {u["field"] for u in units}
    for field in fields:
        if "/" not in field:
            setattr(result, field, "")
    result.questions = [
        q for i, q in enumerate(result.questions) if f"questions/{i}" not in fields
    ]
    result.requested_checks = [
        check
        for i, check in enumerate(result.requested_checks)
        if f"checks/{i}" not in fields
    ]
    if not any(text_fields(result).values()):
        result.direct_answer = "I could not verify the proposed character or rule claim from the current sources. Confirm that detail before using the draft."
    elif type(result).__name__ == "DirectAnswer" and not result.direct_answer:
        result.direct_answer = "The proposed answer needs a source check before use."
    return type(answer).model_validate(result.model_dump())
