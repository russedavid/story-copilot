"""Adapt the learned evidence policy to live context, with a shared-model fallback."""

from copy import deepcopy
import json
import re
import time
from typing import Literal

from .context import _Counter
from .decisions import Decision
from .rl_environment import EvidenceDecision, INSTRUCTIONS
from .store import packed


class LiveDecision(EvidenceDecision):
    intent: Literal["narrative", "rules", "state", "player_choice"] = "narrative"


SYSTEM = INSTRUCTIONS.replace(
    "Optional intent is state/rules; optional reason is a short rationale.",
    "Include intent in every decision: state, rules, narrative, or player_choice. Optional reason is a short response plan.",
) + """
This is a live, general storytelling conversation. The task is not limited to inventory.
state means factual questions about resources, earlier events or who knows something.
rules means costs, checks, bonuses or adjudication; inspect supplied rules before a ruling.
narrative means responding as an NPC or describing a scene following the players' actual choices.
player_choice means a participant has been asked to choose or reply; invite that person's answer.
Do not ask the user to supply an NPC's reply: NPCs belong to the facilitator. Do not take a player's turn.
For narrative, respond when the current task and scene context suffice. Source facts constrain the draft;
the facilitator may propose consistent new NPC speech and world detail. This is guidance, not an observed event.
For facts and rules, use current_state and the latest dialogue before a starting sheet. Current-state resources
already reflect validated source updates. Absence of a supplied rule is unknown, never zero.
Terminal values/citations are proposals for later verification; they do not change state or become a ruling.
"""


def source_view(value):
    """Keep canonical IDs; bookkeeping revision/hash identifiers are not citations."""
    if isinstance(value, dict):
        return {k: source_view(v) for k, v in value.items()
                if k not in {"revision", "sha256", "source_hash", "source_revisions",
                             "campaign_id", "created", "ingestion_ordinal"}}
    if isinstance(value, list):
        return [source_view(v) for v in value]
    return value


def compact_dialogue(messages):
    """Retain order/uncertainty and canonical citations without long clock hashes."""
    groups = {}
    result = source_view(messages)
    for message in result:
        timing = message.get("chronology", {})
        domain = timing.pop("clock_group", None) or timing.get("source_domain")
        timing.pop("source_domain", None)
        if domain:
            timing["clock_group"] = groups.setdefault(domain, len(groups) + 1)
    return result


def in_learned_scope(task):
    """Conservative routing: the RL curriculum teaches resource/rule questions.

    General fiction, social choices and private knowledge retain the main planner.
    This is an applicability check, not a classifier of the answer or its truth.
    """
    return bool(re.search(
        r"\b(?:how many|how much|afford|calculate|what.{0,30}(?:balance|count|cost)|remaining (?:balance|resources)|enough.{0,30}(?:cells|charges|supplies|resources))\b",
        task, re.I,
    ))


class LivePlanner:
    def __init__(self, model, fallback, *, options=None, context_limit=4096):
        self.model, self.fallback = model, fallback
        self.options = options or {}
        self.budget = min(context_limit, 8192) - 700
        self.history = []
        self.observed = 0
        self.failed = None

    def complete(self, messages, schema, **kwargs):
        started = time.monotonic()
        timeout = kwargs.get("timeout", 75)
        if self.failed is None:
            try:
                brief = json.loads(messages[-1]["content"])
                if not self.history and not in_learned_scope(brief["current_task"]):
                    self.failed = {"reason": "Outside the learned resource/rule-question scope.", "scope": True}
                    raise ValueError(self.failed["reason"])
                if not self.history:
                    body = brief["current_context"]
                    initial = source_view({
                        "task": brief["current_task"],
                        "characters": brief["available_characters"],
                        "inventory_status": "Current state and recent dialogue supersede starting sheets. Earlier dialogue remains available through recall.",
                        "visible_evidence": compact_dialogue(body.get("dialogue", [])),
                        "current_state": body.get("state", {}),
                        "scene_context": body.get("documents", []),
                        "facilitator_direction": body.get("private_facilitator_direction", ""),
                        "context_instruction": body.get("instruction", ""),
                        "rules_available": brief.get("rules_available", []),
                    })
                    counter = _Counter(self.options.get("tokenizer"), self.options.get("token_counter"))
                    # Retain control/knowledge boundaries. If the complete live
                    # brief will not fit, use the larger planner instead of slicing it.
                    if counter.count(initial, SYSTEM) > self.budget:
                        raise ValueError("Live context exceeds the dedicated planner budget.")
                    self.history = [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": packed(source_view(initial))}]
                for observation in brief.get("tool_results", [])[self.observed:]:
                    self.history.append({"role": "user", "content": "Evidence result: " + packed(source_view(observation))})
                self.observed = len(brief.get("tool_results", []))
                counter = _Counter(self.options.get("tokenizer"), self.options.get("token_counter"))
                if counter.count(self.history, "") > self.budget:
                    raise ValueError("Evidence exceeds the dedicated planner budget.")
                output, metrics = self.model.complete(
                    self.history, LiveDecision, max_tokens=320, temperature=0,
                    timeout=min(25, max(1, timeout / 2)),
                    constrain=False,
                )
                self.history.append({"role": "assistant", "content": metrics.get("response") or packed(output.model_dump(exclude_defaults=True))})
                inferred_intent = "intent" not in output.model_fields_set
                if inferred_intent:
                    # The original learned contract predates the live intent
                    # label. Derive only what its typed evidence fields establish.
                    if output.action == "rules" or output.allowed is not None or "rule" in output.missing:
                        output.intent = "rules"
                    elif output.action in {"character", "recall"} or output.value is not None or "balance" in output.missing:
                        output.intent = "state"
                data = output.model_dump(include=set(Decision.model_fields))
                if output.intent in {"state", "rules"} and output.action == "respond":
                    data["reason"] = "Answer from current verified sources; use the rules calculator for numerical outcomes. Planner-supplied values are not authoritative."
                return Decision.model_validate(data), {
                    **metrics, "planner_route": "dedicated", "live_adapter_version": 1,
                    "intent_from_evidence_fields": inferred_intent,
                    "terminal_proposal_not_applied": output.model_dump(exclude=set(Decision.model_fields)),
                }
            except Exception as exc:
                if self.failed is None:
                    self.failed = {"reason": str(exc), "trace": getattr(exc, "trace", {})}
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise ValueError("Planner deadline expired before fallback.")
        output, metrics = self.fallback.complete(messages, schema, **{**kwargs, "timeout": remaining})
        return output, {**metrics, "planner_route": "shared_scope" if self.failed.get("scope") else "shared_fallback",
                        "dedicated_failure": deepcopy(self.failed)}
