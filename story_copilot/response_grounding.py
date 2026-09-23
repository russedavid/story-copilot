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
    verdict: Literal[
        "supported", "unsupported", "not_an_assertion", "creative_proposal"
    ]
    actor: str = Field(default="", max_length=160)
    support: list[Support] = Field(default_factory=list, max_length=6)
    reason: str = Field(min_length=1, max_length=500)


RULE_CLAIM = re.compile(
    r"\b(?:no|zero|none|without|\d+(?:\.\d+)?)\b.{0,65}\b(?:bonus|modifier|penalty|advantage|cost|damage|difficulty)\b"
    r"|\b(?:bonus|modifier|penalty|advantage|cost|damage|difficulty)\b.{0,65}\b(?:no|zero|none|\d+(?:\.\d+)?)\b",
    re.I,
)
NONASSERTION = re.compile(
    r"\?|\b(?:if|can|could|may|might|would|perhaps|maybe|possible|possibility|possibilities|whether|unknown|unspecified|not specified|not established|not known|unclear)\b",
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


def quote_context(text, position, names):
    """Carry quotation ownership across sentence boundaries without inventing a speaker."""
    opened = []
    for index, char in enumerate(text[:position]):
        if char == "“":
            opened.append(("”", index))
        elif char == "”":
            if opened and opened[-1][0] == char:
                opened.pop()
        elif char == '"':
            if opened and opened[-1][0] == char:
                opened.pop()
            else:
                opened.append(('"', index))
    if not opened:
        return False, False
    prefix = text[max(0, opened[0][1] - 120) : opened[0][1]]
    player_speech = any(
        re.search(
            r"(?<!\w)"
            + re.escape(name)
            + r"\s+(?:\w+\s+)?(?:says|asks|replies|answers|whispers|shouts)\b",
            prefix,
            re.I,
        )
        for name in names
    )
    return True, player_speech


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
        for match in re.finditer(
            r"[^\n]+?(?:[.!?](?!\d)[\"'’”]*(?=\s|$)|$)", text, re.M
        ):
            value = match[0].strip()
            if not value:
                continue
            quoted, player_speech = quote_context(text, match.start(), names)
            quoted = quoted or value.startswith(('"', "“", "'", "‘"))
            first_quote = re.search('[“"]', match[0])
            actor_match = actor.search(match[0])
            if first_quote and actor_match and first_quote.start() < actor_match.start():
                quoted, player_speech = quote_context(
                    text, match.start() + first_quote.start() + 1, names
                )
            kinds = []
            if actor.search(value):
                kinds.append("player")
            if RULE_CLAIM.search(value):
                kinds.append("rule")
            if field == "private_notes":
                kinds.append("private_context")
            if kinds and not value.rstrip("\"'’”").endswith("?"):
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
                if quoted and not player_speech:
                    subject = None
                units.append(
                    {
                        "id": digest(packed([field, match.start(), value]))[:16],
                        "field": field,
                        "text": value,
                        "kinds": kinds,
                        "can_be_nonassertion": (
                            bool(NONASSERTION.search(value))
                            or bool(
                                re.match(
                                    r"^(?:Keep|Maintain|Preserve|Avoid|Leave|Ask|Do not|Don't|Consider|Treat|Remember|Suggestion:|Proposal:)",
                                    value,
                                    re.I,
                                )
                            )
                        )
                        and not bool(ZERO_RULE.search(value)),
                        "explicit_zero_rule": bool(ZERO_RULE.search(value)),
                        "literal_player_subject": subject,
                        "quoted_dialogue": quoted,
                        "quoted_player_speech": player_speech,
                        "preceding_text": text[max(0, match.start() - 240) : match.start()],
                        "player_names": names,
                        "mentioned_players": [
                            name
                            for name in names
                            if re.search(
                                r"(?<!\w)" + re.escape(name) + r"(?!\w)", value, re.I
                            )
                        ],
                    }
                )
    return units


def sources(body):
    result = []
    if body.get("private_facilitator_direction"):
        result.append(
            {
                "id": "facilitator-direction",
                "kind": "direction",
                "text": body["private_facilitator_direction"],
                "visibility": "private",
            }
        )
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
        if document.get("id") == "evidence-investigation":
            try:
                investigation = json.loads(document["text"])
                for observation in investigation.get("observations", []):
                    if observation.get("tool") not in {
                        "recall",
                        "rules",
                    } or not isinstance(observation.get("result"), list):
                        continue
                    for item in observation["result"]:
                        if (
                            isinstance(item, dict)
                            and item.get("id")
                            and item.get("text")
                        ):
                            result.append(
                                {
                                    "id": item["id"],
                                    "kind": "rule"
                                    if observation["tool"] == "rules"
                                    else "conversation",
                                    "text": item["text"],
                                    "visibility": item.get("visibility", "private"),
                                }
                            )
            except (ValueError, TypeError, KeyError):
                pass
            continue
        if document.get("id") in {
            "table-control",
            "response-audience",
            "response-plan",
            "validated-rule-advice",
            "unresolved-source-analysis",
        } or str(document.get("id", "")).startswith("feedback:"):
            continue
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
        elif (
            document.get("id")
            not in {
                "table-control",
                "response-audience",
                "response-plan",
                "validated-rule-advice",
            }
            and not str(document.get("id", "")).startswith("feedback:")
            and document.get("text")
        ):
            result.append(
                {
                    "id": document["id"],
                    "kind": "scenario",
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
            elif category == "facts" and "value" in value:
                text = f"{key}: {value['value']}"
            elif category == "knowledge":
                text = "\n".join(
                    f"{key}:{attribute}: {record['value']}"
                    for attribute, record in value.items()
                    if isinstance(record, dict) and "value" in record
                )
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
        if check.verdict == "creative_proposal":
            quoted = unit.get("quoted_dialogue", False)
            if (
                unit["field"] not in {"narration", "direct_answer"}
                or (unit["field"] == "direct_answer" and not quoted)
                or unit.get("literal_player_subject")
                or unit.get("quoted_player_speech")
                or not check.actor.strip()
                or check.actor.casefold()
                in {p.casefold() for p in unit.get("player_names", [])}
                or (
                    not quoted
                    and check.actor.casefold() not in unit["text"].casefold()
                    and not (
                        re.match(r"^(?:She|He|They|It)\b", unit["text"], re.I)
                        and check.actor.casefold() in unit.get("preceding_text", "").casefold()
                    )
                )
            ):
                raise ValueError(
                    "Creative proposals belong to NPCs or the environment in narration, not player performance or private factual history."
                )
            for support in check.support:
                if support.source_id not in lookup:
                    raise ValueError("Creative support cites an unavailable source.")
                source_quote(lookup[support.source_id]["text"], support.quote)
            continue
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
        if unit["field"] == "narration" and all(
            s["visibility"] != "public" for s in selected
        ):
            raise ValueError(
                "Shared narration cannot expose a player fact supported only by private sources."
            )
    return unsupported


def source_wording(answer, units, checks, evidence):
    """Render verified source wording for attributed facts, preserving other prose.

    A model's unsupported paraphrase cannot acquire new gestures merely because
    its citation exists. The original and each source substitution remain traced.
    """
    result = deepcopy(answer)
    lookup = {s["id"]: s for s in evidence}
    by_id = {c.id: c for c in checks}
    edits = []
    words = lambda text: " ".join(re.findall(r"\w+", text.casefold()))
    for unit in units:
        check = by_id[unit["id"]]
        if (
            not unit.get("literal_player_subject")
            and "private_context" not in unit["kinds"]
        ) or check.verdict != "supported":
            continue
        if any(words(unit["text"]) in words(s.quote) for s in check.support):
            continue
        # Keep one complete source assertion, not a synthetic sentence built by
        # joining fragments from several speakers. Short name-only quotes fail.
        candidates = [
            s
            for s in check.support
            if len(words(s.quote).split()) >= 3
            and (
                not unit.get("literal_player_subject")
                or unit["literal_player_subject"].casefold() == "you"
                or unit["literal_player_subject"].casefold() in s.quote.casefold()
            )
        ]
        if not candidates:
            continue  # The bounded review can remove or clarify this claim.
        support = candidates[0]
        source_quote(lookup[support.source_id]["text"], support.quote)
        replacement = support.quote.strip()
        if replacement[-1:] not in ".!?":
            replacement += "."
        if unit["field"] == "private_notes" and not unit.get("literal_player_subject"):
            label = {
                "conversation": "Recorded speech",
                "rule": "Supplied rule",
                "observed_state": "Working state",
                "scenario": "Scenario",
                "direction": "Facilitator direction",
            }[lookup[support.source_id]["kind"]]
            replacement = f"{label}: “{replacement}”"
        field = unit["field"]
        current = text_fields(result).get(field, "")
        if unit["text"] not in current:
            continue
        revised = current.replace(unit["text"], replacement, 1)
        if field.startswith("questions/"):
            result.questions[int(field.split("/")[1])] = revised
        elif field.startswith("checks/"):
            result.requested_checks[int(field.split("/")[1])].text = revised
        else:
            setattr(result, field, revised)
        edits.append(
            {
                "field": field,
                "original": unit["text"],
                "rendered": replacement,
                "source_id": support.source_id,
            }
        )
    return result, edits


def assess_claims(units, checks, evidence):
    """A bad citation in one note must not invalidate an independently verified answer."""
    approved, rejected, failures = [], [], []
    for unit in units:
        matching = [check for check in checks if check.id == unit["id"]]
        try:
            bad = validate_checks([unit], matching, evidence)
            if bad:
                rejected.append(unit)
            else:
                approved.extend(matching)
        except ValueError as exc:
            rejected.append(unit)
            failures.append({"id": unit["id"], "text": unit["text"], "error": str(exc)})
    unexpected = {check.id for check in checks} - {unit["id"] for unit in units}
    if unexpected:
        failures.append(
            {"error": "Review included unknown claim IDs.", "ids": sorted(unexpected)}
        )
    return approved, rejected, failures


def clean_removed_quotes(text):
    """Drop orphaned dialogue delimiters after deletion; never invent replacement prose."""
    opened, orphaned = [], []
    for index, char in enumerate(text):
        if char == "“":
            opened.append(("”", index))
        elif char == "”":
            if opened and opened[-1][0] == char:
                opened.pop()
            else:
                orphaned.append(index)
        elif char == '"':
            if opened and opened[-1][0] == char:
                opened.pop()
            else:
                opened.append(('"', index))
    orphaned.extend(index for _, index in opened)
    return "".join(char for index, char in enumerate(text) if index not in orphaned)


def remove_claims(answer, units):
    """Remove rejected sentences rather than discarding the rest of a useful field."""
    result = deepcopy(answer)
    for unit in units:
        field = unit["field"]
        value = text_fields(result).get(field, "")
        value = value.replace(unit["text"], "", 1).strip()
        if field.startswith("questions/"):
            result.questions[int(field.split("/")[1])] = value
        elif field.startswith("checks/"):
            result.requested_checks[int(field.split("/")[1])].text = value
        else:
            setattr(result, field, value)
    for field in {unit["field"] for unit in units}:
        value = clean_removed_quotes(text_fields(result).get(field, ""))
        if field.startswith("questions/"):
            result.questions[int(field.split("/")[1])] = value
        elif field.startswith("checks/"):
            result.requested_checks[int(field.split("/")[1])].text = value
        else:
            setattr(result, field, value)
    result.questions = [q for q in result.questions if q.strip()]
    result.requested_checks = [c for c in result.requested_checks if c.text.strip()]
    if not any(text_fields(result).values()):
        result.direct_answer = "I could not verify the proposed character or rule claim from the current sources. Confirm that detail before using the draft."
    elif type(result).__name__ == "DirectAnswer" and not result.direct_answer:
        result.direct_answer = "The proposed answer needs a source check before use."
    return type(answer).model_validate(result.model_dump())


def issue_units(answer, issues):
    """Locate the sentences covered by exact, anchored editorial defects."""
    result = []
    for field, text in text_fields(answer).items():
        for issue in issues:
            try:
                quote = source_quote(text, issue.quote)
            except ValueError:
                continue
            start = text.index(quote)
            end = start + len(quote)
            for match in re.finditer(
                r"[^\n]+?(?:[.!?](?!\d)[\"\'’”]*(?=\s|$)|$)", text, re.M
            ):
                if match.start() < end and match.end() > start:
                    result.append({"field": field, "text": match[0].strip()})
    return result


def literal_checks(units, evidence):
    """Recognize exact source restatements introduced by a repair without a third call."""
    result = []
    for unit in units:
        support = []
        for source in evidence:
            try:
                quote = source_quote(source["text"], unit["text"].rstrip("."))
            except ValueError:
                continue
            candidate = ClaimCheck(
                id=unit["id"],
                verdict="supported",
                support=[Support(source_id=source["id"], quote=quote)],
                reason="Exact source restatement checked by the application.",
            )
            try:
                validate_checks([unit], [candidate], evidence)
            except ValueError:
                continue
            support.append(candidate)
            break
        if support:
            result.extend(support)
    return result


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
