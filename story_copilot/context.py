"""CPU-only, source-linked context packing; memories never become game facts.

``pack_context`` handles live campaign snapshots. ``build_context`` adds immutable
SQLite snapshots and revision guards for imported transcript collections. Neither
calls a model, rewrites sources, or accepts proposed events into the state ledger.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from copy import deepcopy

from .state import replay_events
from .store import digest, now, packed

POLICY_VERSION = 4
INSTRUCTION = (
    "Draft the next Facilitator response to the current player input. Respect accepted "
    "state and individual character knowledge. Historical dialogue, claims and "
    "hypotheses are evidence of speech, not automatically true facts. Private "
    "direction, private facts and private documents must not appear in public "
    "narration until discovered. Do not invent player choices or rolls."
)
_STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


class ContextBudgetError(ValueError):
    def __init__(self, message, trace):
        super().__init__(message)
        self.trace = trace


class StaleContextError(ValueError):
    pass


def _terms(text):
    return [
        t
        for t in re.findall(r"[^\W_]+", text.casefold())
        if len(t) > 1 and t not in _STOP
    ]


def _private(item):
    return isinstance(item, dict) and item.get("visibility") in {"private", "private"}


def _visible(value):
    if isinstance(value, dict):
        return {k: _visible(v) for k, v in value.items() if not _private(v)}
    if isinstance(value, list):
        return [_visible(v) for v in value if not _private(v)]
    return deepcopy(value)


class _Counter:
    def __init__(self, tokenizer, counter):
        if tokenizer is not None and counter is not None:
            raise ValueError("Supply a tokenizer or a token counter, not both.")
        self.tokenizer, self.counter = tokenizer, counter
        self.mode = (
            "native-chat-template"
            if tokenizer is not None
            else "injected-token-counter"
            if counter is not None
            else "conservative-utf8-bytes"
        )

    def count(self, body, system):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": packed(body)},
        ]
        if self.tokenizer is not None:
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            return len(ids)
        if self.counter is not None:
            counts = [self.counter(m["content"]) for m in messages]
            if any(type(n) is not int or n < 0 for n in counts):
                raise ValueError("Token counter must return nonnegative integers.")
            return sum(counts) + 32
        # Deliberately pessimistic for byte-based tokenizers, including non-Latin
        # text. This is an estimate, not a measured native-tokenizer count.
        return sum(len(m["content"].encode("utf-8")) for m in messages) + 128


def _turn_view(t):
    return {
        k: t[k]
        for k in (
            "id",
            "ordinal",
            "revision",
            "speaker",
            "role",
            "character",
            "text",
            "visibility",
            "status",
            "order_index",
            "ingestion_ordinal",
            "chronology",
            "speaker_mapping",
            "start",
            "end",
        )
        if k in t
    }


def _exchanges(turns):
    """Preserve recent utterances without pinning hours of one speaker's audio.

    Distinct source records are utterance boundaries unless timing demonstrates
    an adjacent unfinished fragment. Even such fragment runs are bounded at
    source boundaries; a single oversized current message is never truncated.
    """
    blocks = []
    for turn in turns:
        previous = blocks[-1][-1] if blocks else None
        continuous = False
        if (
            previous
            and previous.get("speaker") == turn.get("speaker")
            and previous.get("role") == turn.get("role")
        ):
            prior_clock, clock = (
                previous.get("chronology", {}),
                turn.get("chronology", {}),
            )
            prior_end, start = previous.get("end"), turn.get("start")
            if prior_clock or clock:
                if prior_clock.get("clock_group") and prior_clock.get(
                    "clock_group"
                ) == clock.get("clock_group"):
                    prior_end, start = prior_clock.get("end"), clock.get("start")
                else:
                    prior_end, start = None, None
            timed = (
                isinstance(prior_end, (int, float))
                and isinstance(start, (int, float))
                and 0 <= start - prior_end <= 0.8
            )
            finished = bool(re.search(r"[.!?][\"'’”)]*\s*$", previous["text"]))
            continuous = (
                timed
                and not finished
                and len(blocks[-1]) < 8
                and sum(len(t["text"]) for t in blocks[-1]) + len(turn["text"]) <= 2000
            )
        if continuous:
            blocks[-1].append(turn)
        else:
            blocks.append([turn])
    groups = []
    for block in blocks:
        role = block[0].get("role")
        # A new complete utterance from the same role must not extend a single
        # mandatory exchange forever (monologues and unmapped ASR are common).
        same_role = bool(groups and groups[-1][-1].get("role") == role)
        facilitator_after_reply = (
            role == "facilitator"
            and groups
            and any(t.get("role") != "facilitator" for t in groups[-1])
        )
        if not groups or same_role or facilitator_after_reply:
            groups.append([])
        groups[-1].extend(block)
    return groups


def _rank(items, query):
    """Small-corpus BM25-style ranking, with deterministic recency tie breaks."""
    q = Counter(_terms(query))
    docs = [Counter(_terms(packed(item))) for item in items]
    df = Counter(t for doc in docs for t in doc)
    avg = sum(sum(d.values()) for d in docs) / max(1, len(docs)) or 1
    scores = []
    for index, doc in enumerate(docs):
        length = sum(doc.values())
        score = sum(
            q[t]
            * math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
            * doc[t]
            * 2.2
            / (doc[t] + 1.2 * (0.25 + 0.75 * length / avg))
            for t in q
            if doc[t]
        )
        scores.append((score, index))
    return sorted(scores, reverse=True)


def _excerpt(turn, query, limit=700):
    text = turn["text"]
    if len(text) <= limit:
        start, end = 0, len(text)
    else:
        words = set(_terms(query))
        spans = list(re.finditer(r"[^.!?\n]+[.!?\n]*", text))
        best = max(
            spans,
            key=lambda m: len(words.intersection(_terms(m.group()))),
            default=None,
        )
        start = best.start() if best else 0
        if best and best.end() - start > limit:
            weights = Counter(_terms(query))
            matches = [
                m
                for m in re.finditer(r"[^\W_]+", text[best.start() : best.end()])
                if m.group().casefold() in weights
            ]
            if matches:
                focus = max(matches, key=lambda m: weights[m.group().casefold()])
                start = max(best.start(), best.start() + focus.start() - limit // 3)
                if start > best.start() and text[start - 1].isalnum():
                    boundary = text.find(" ", start, start + limit // 3)
                    if boundary >= start:
                        start = boundary + 1
        end = min(len(text), start + limit)
        # End at a word boundary; exact offsets make truncation unambiguous.
        if end < len(text) and text[end].isalnum():
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
    result = _turn_view(turn)
    result.pop("text", None)
    result.update(
        quote=text[start:end],
        char_start=start,
        char_end=end,
        truncated=(start > 0 or end < len(text)),
        interpretation="source_dialogue_not_established_fact",
    )
    return result


def _state_entries(state):
    result = []
    for category in ("entities", "facts", "knowledge"):
        for key, value in state.get(category, {}).items():
            if category in {"entities", "knowledge"}:
                for attribute, item in value.items():
                    result.append((category, key, attribute, item))
            else:
                result.append((category, key, None, value))
    return result


def _put(state, entry):
    category, key, attribute, item = entry
    if attribute is None:
        state.setdefault(category, {})[key] = item
    else:
        state.setdefault(category, {}).setdefault(key, {})[attribute] = item


def _ref(entry):
    category, key, attr, item = entry
    return {
        "path": [category, key] + ([attr] if attr is not None else []),
        "event": item.get("event") if isinstance(item, dict) else None,
    }


def pack_context(
    *,
    turns,
    state,
    player_input="",
    direction="",
    rules=(),
    documents=(),
    context_limit=16384,
    output_reserve=1800,
    safety_margin=512,
    system_prompt="",
    tokenizer=None,
    token_counter=None,
    public_only=False,
    active_entities=(),
    recent_exchanges=2,
    compact_at=0.85,
):
    """Pack a source snapshot without consuming GPU memory or summarizing facts.

    Resources, pending actions, active entities, the current input/direction and
    newest complete exchanges are mandatory. Inactive facts remain in durable
    state and compete with older source passages for the remaining prompt budget.
    A caller using a live campaign owns snapshot persistence and concurrency checks.
    ``public_only`` requires explicit public visibility on raw turns/documents;
    missing visibility is never assumed to mean that a secret is safe to publish.
    """
    started = time.perf_counter()
    if (
        any(
            type(n) is not int or n < 0
            for n in (context_limit, output_reserve, safety_margin)
        )
        or context_limit <= output_reserve + safety_margin
        or type(recent_exchanges) is not int
        or recent_exchanges < 1
        or not 0 < compact_at <= 1
    ):
        raise ValueError("Invalid context budget or exchange count.")
    counter = _Counter(tokenizer, token_counter)
    available = context_limit - output_reserve - safety_margin
    source_turns = list(turns)
    turns = [
        _turn_view(t)
        for t in source_turns
        if not public_only or t.get("visibility") == "public"
    ]
    state = _visible(state) if public_only else deepcopy(state)
    full_state_hash = digest(packed(state))
    # Starting-sheet resource totals are historical. Showing them beside the
    # current resource ledger caused both narration and editing to restore old
    # balances. Preserve the sheet in storage and explicit character retrieval,
    # but give every packed model context one place for current totals.
    for attributes in state.get("entities", {}).values():
        sheet = attributes.get("sheet", {})
        value = sheet.get("value") if isinstance(sheet, dict) else None
        if isinstance(value, dict) and "resources" in value:
            value.pop("resources")
    rules = [
        deepcopy(r) for r in rules if not public_only or r.get("visibility") == "public"
    ]
    documents = [
        deepcopy(d)
        for d in documents
        if not public_only or d.get("visibility") == "public"
    ]
    direction = "" if public_only else direction
    groups = _exchanges(turns)
    recent = [t for group in groups[-recent_exchanges:] for t in group]
    old = turns[: len(turns) - len(recent)]
    if len(set(_terms(player_input))) >= 2:
        query = " ".join([player_input] * 4 + [direction])
    else:
        query = " ".join([player_input, direction] + [t["text"] for t in recent[-2:]])
    entries = _state_entries(state)
    active = {e.casefold() for e in active_entities}
    # Mentioned entities become active; caller can explicitly pin the whole party.
    for entity in state.get("entities", {}):
        if re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", query, re.IGNORECASE):
            active.add(entity.casefold())
    pinned, optional = [], []
    for entry in entries:
        owner = entry[1].split(":", 1)[0].casefold()
        item = entry[3]
        explicit_pin = isinstance(item, dict) and item.get("pinned") is True
        # Being an active character does not pin every clue they have ever
        # learned. Their identity and live resources stay; old knowledge is
        # retrieved in its original owner-labelled container when relevant.
        live_property = (entry[2] or entry[1].split(":")[-1]).casefold() in {
            "current_scene",
            "current_location",
            "location",
            "active_scene",
        }
        required = (
            explicit_pin
            or (
                live_property
                and (
                    owner in active
                    or owner in {"scene", "world", "session", "current", "party"}
                )
            )
            or (entry[0] == "entities" and owner in active)
        )
        (pinned if required else optional).append(entry)
    base_state = {
        "entities": {},
        "facts": {},
        "knowledge": {},
        "resources": deepcopy(state.get("resources", {})),
        "pending": deepcopy(state.get("pending", {})),
        "resolutions": deepcopy(state.get("resolutions", [])[-4:]),
    }
    for entry in pinned:
        _put(base_state, entry)
    body = {
        "state": base_state,
        "dialogue": recent,
        "new_player_input": player_input,
        "private_facilitator_direction": direction,
        "rules": [],
        "documents": [d for d in documents if d.get("pinned")],
        "historical_claims_and_hypotheses": [],
        "historical_source_passages": [],
        "instruction": INSTRUCTION,
    }
    trace = {
        "policy_version": POLICY_VERSION,
        "token_counter": counter.mode,
        "context_limit": context_limit,
        "output_reserve": output_reserve,
        "safety_margin": safety_margin,
        "prompt_budget": available,
        "public_only": public_only,
        "source_turn_count": len(source_turns),
        "eligible_turn_count": len(turns),
        "retained_turns": [],
        "retrieved_turns": [],
        "matched_source_turns": [],
        "omitted_state": [],
        "omitted_rules": [],
        "omitted_documents": [],
        "excerpted_documents": [],
        "omitted_resolutions": [],
        "compacted": False,
    }
    mandatory_count = counter.count(body, system_prompt)
    trace["mandatory_tokens"] = mandatory_count
    if mandatory_count > available:
        trace["prompt_tokens"] = mandatory_count
        raise ContextBudgetError(
            "Current player input, live resources, unresolved actions or the latest complete exchanges exceed the context budget. Increase the window or shorten explicit input; they were not silently discarded.",
            trace,
        )
    # Keep all context when it fits comfortably; the same representation is used
    # before and after compaction, avoiding a semantic switch in prompt fields.
    complete = deepcopy(body)
    for entry in optional:
        _put(complete["state"], entry)
    complete["dialogue"] = turns
    complete["rules"] = rules
    complete["documents"] = documents
    complete["state"]["resolutions"] = deepcopy(state.get("resolutions", []))
    complete["historical_claims_and_hypotheses"] = deepcopy(
        state.get("claims", [])
    ) + deepcopy(state.get("hypotheses", []))
    full_count = counter.count(complete, system_prompt)
    trace["uncompacted_prompt_tokens"] = full_count
    trace["full_state_sha256"] = full_state_hash
    if full_count <= available * compact_at:
        body = complete
    else:
        trace["compacted"] = True
        # Full accepted state is retained outside this prompt. Add relevant facts
        # first; then rules, older evidence and complete recent exchanges.
        rankable = [{"path": e[:3], "value": e[3]} for e in optional]
        selected_entries = set()

        def add(candidate):
            nonlocal body
            if counter.count(candidate, system_prompt) <= available:
                body = candidate
                return True
            return False

        # Reserve a portion for source retrieval; many older accepted facts should
        # not displace the player's newly relevant question and its evidence.
        state_ceiling = mandatory_count + max(0, available - mandatory_count) * 0.45
        for _, index in _rank(rankable, query):
            candidate = deepcopy(body)
            _put(candidate["state"], optional[index])
            if counter.count(candidate, system_prompt) <= state_ceiling and add(
                candidate
            ):
                selected_entries.add(index)
        trace["omitted_state"] = [
            _ref(e) for i, e in enumerate(optional) if i not in selected_entries
        ]
        ranked_old = _rank(old, query)
        # Retrieve an exchange as a unit: the reply may establish who kept an
        # object, declined an action, or corrected the observation in the hit.
        # Keeping only the matching sentence can silently change its meaning.
        historical_groups = _exchanges(old)
        group_for_index = {}
        cursor = 0
        for group_number, group in enumerate(historical_groups):
            for index in range(cursor, cursor + len(group)):
                group_for_index[index] = group_number
            cursor += len(group)
        selected_groups = set()
        trace["omitted_source_exchanges"] = []

        def retrieve(limit):
            for score, index in ranked_old:
                if score <= 0:
                    break
                group_number = group_for_index[index]
                if group_number in selected_groups:
                    continue
                group = historical_groups[group_number]
                candidate = deepcopy(body)
                candidate["historical_source_passages"].extend(
                    _excerpt(turn, query) for turn in group
                )
                if add(candidate):
                    selected_groups.add(group_number)
                    trace["matched_source_turns"].append(
                        old[index].get("ordinal", old[index].get("id"))
                    )
                    trace["retrieved_turns"].extend(
                        turn.get("ordinal", turn.get("id")) for turn in group
                    )
                else:
                    omitted = [turn.get("ordinal", turn.get("id")) for turn in group]
                    if omitted not in trace["omitted_source_exchanges"]:
                        trace["omitted_source_exchanges"].append(omitted)
                if len(selected_groups) >= limit:
                    break

        # Protect the most relevant source evidence from being crowded out by
        # rules/documents; then spend the remaining budget on additional context.
        retrieve(2)
        for r in rules:
            candidate = deepcopy(body)
            candidate["rules"].append(r)
            if not add(candidate):
                trace["omitted_rules"].append(r.get("id", digest(packed(r))))
        for d in documents:
            if d.get("pinned"):
                continue
            candidate = deepcopy(body)
            candidate["documents"].append(d)
            if add(candidate):
                continue
            # A long scenario document remains durable; retrieve an exact, cited
            # excerpt instead of dropping the whole scenario from a live session.
            excerpt = _excerpt({"text": d.get("text", "")}, query, limit=900)
            partial = {
                **d,
                "text": excerpt["quote"],
                "truncated": excerpt["truncated"],
                "char_start": excerpt["char_start"],
                "char_end": excerpt["char_end"],
                "source_text_sha256": digest(d.get("text", "")),
            }
            candidate = deepcopy(body)
            candidate["documents"].append(partial)
            if add(candidate):
                trace["excerpted_documents"].append(
                    {
                        k: partial[k]
                        for k in ("id", "char_start", "char_end")
                        if k in partial
                    }
                )
            else:
                trace["omitted_documents"].append(d.get("id", digest(packed(d))))
        retrieve(8)
        prior_resolutions = state.get("resolutions", [])[:-4]
        selected_resolutions = set()
        for _, index in _rank(prior_resolutions, query):
            candidate = deepcopy(body)
            candidate["state"]["resolutions"] = [
                r
                for i, r in enumerate(prior_resolutions)
                if i in selected_resolutions | {index}
            ] + state.get("resolutions", [])[-4:]
            if add(candidate):
                selected_resolutions.add(index)
        trace["omitted_resolutions"] = [
            r.get("event", r.get("action"))
            for i, r in enumerate(prior_resolutions)
            if i not in selected_resolutions
        ]
        claims = list(state.get("claims", [])) + list(state.get("hypotheses", []))
        for _, index in _rank(claims, query):
            candidate = deepcopy(body)
            candidate["historical_claims_and_hypotheses"].append(claims[index])
            add(candidate)
        for group in reversed(groups[:-recent_exchanges]):
            candidate = deepcopy(body)
            candidate["dialogue"] = group + candidate["dialogue"]
            if not add(candidate):
                break
        # Spend any remainder on accepted facts; do not discard them merely to
        # achieve a fixed compression ratio.
        for _, index in _rank(rankable, query):
            if index not in selected_entries:
                candidate = deepcopy(body)
                _put(candidate["state"], optional[index])
                if add(candidate):
                    selected_entries.add(index)
        trace["omitted_state"] = [
            _ref(e) for i, e in enumerate(optional) if i not in selected_entries
        ]
    # A retrieved excerpt need not be duplicated when its complete source turn
    # subsequently fits in the retained dialogue. Avoid "compaction" that grows
    # an already-fitting prompt merely by adding retrieval wrappers.
    dialogue_sources = {t.get("id", t.get("ordinal")) for t in body["dialogue"]}
    body["historical_source_passages"] = [
        p
        for p in body["historical_source_passages"]
        if p.get("id", p.get("ordinal")) not in dialogue_sources
    ]
    trace["retrieved_turns"] = [
        p.get("ordinal", p.get("id")) for p in body["historical_source_passages"]
    ]
    if (
        trace["compacted"]
        and full_count <= available
        and counter.count(body, system_prompt) >= full_count
    ):
        body = complete
        trace["compaction_attempted"] = True
        trace["compacted"] = False
        trace["selection_reason"] = (
            "Uncompacted context fits and is no larger than the packed alternative."
        )
        for key in (
            "retrieved_turns",
            "omitted_state",
            "omitted_rules",
            "omitted_documents",
            "excerpted_documents",
            "omitted_resolutions",
        ):
            trace[key] = []
    trace["retained_turns"] = [t.get("ordinal", t.get("id")) for t in body["dialogue"]]
    kept = set(trace["retained_turns"]) | set(trace["retrieved_turns"])
    trace["compacted_turns"] = [
        t.get("ordinal", t.get("id"))
        for t in turns
        if t.get("ordinal", t.get("id")) not in kept
    ]
    trace["prompt_tokens"] = counter.count(body, system_prompt)
    trace["seconds"] = round(time.perf_counter() - started, 6)
    trace["prompt_bytes"] = len(packed(body).encode())
    return {"body": body, "trace": trace}


def _prefix(store, cid, before):
    store.collection(cid)
    all_turns = store.turns(cid)
    if type(before) is not int or not 1 <= before <= len(all_turns) + 1:
        raise ValueError("Choose an existing turn or the next turn after the end.")
    turns = [t for t in all_turns if t["ordinal"] < before]
    events = [e for e in store.events(cid) if e["anchor"] < before]
    refs = {str(t["ordinal"]): t["revision"] for t in turns}
    fingerprint = digest(
        packed(
            {
                "revisions": refs,
                "events": [
                    {k: e[k] for k in ("id", "status", "stale", "revisions")}
                    for e in events
                ],
            }
        )
    )
    return turns, events, refs, fingerprint


def context_is_current(store, cid, context):
    return _prefix(store, cid, context["before"])[3] == context["context_fingerprint"]


def _ensure_memory(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS context_memories (
            id TEXT PRIMARY KEY, collection_id TEXT NOT NULL REFERENCES collections(id),
            before_turn INTEGER NOT NULL, fingerprint TEXT NOT NULL,
            payload TEXT NOT NULL, created TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS context_memory_prefix ON context_memories(collection_id,before_turn);
        CREATE TRIGGER IF NOT EXISTS context_memory_no_update BEFORE UPDATE ON context_memories
        BEGIN SELECT RAISE(ABORT,'Context memories are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS context_memory_no_delete BEFORE DELETE ON context_memories
        BEGIN SELECT RAISE(ABORT,'Context memories are immutable'); END;
    """)


def build_context(store, cid, before, **kwargs):
    """Create a packed imported-transcript context and an immutable audit snapshot."""
    turns, events, refs, fingerprint = _prefix(store, cid, before)
    public_only = kwargs.get("public_only", False)
    state = replay_events(events, before, public_only=public_only)
    # Hypothetical actions are deliberately absent from the authoritative replay.
    # Retain them explicitly as hypotheses, with original evidence and revisions.
    superseded = {
        e["payload"].get("supersedes")
        for e in events
        if e["status"] == "accepted"
        and not e["stale"]
        and (not public_only or e["payload"]["visibility"] == "public")
    }
    hypotheses = [
        {**e["payload"], "event": e["id"], "source_revisions": e["revisions"]}
        for e in events
        if e["status"] == "accepted"
        and not e["stale"]
        and e["id"] not in superseded
        and e["payload"]["stage"] == "hypothetical"
        and (not public_only or e["payload"]["visibility"] == "public")
    ]
    pack_state = {**state, "hypotheses": hypotheses}
    for t in turns:
        metadata = json.loads(t["metadata"])
        if "visibility" in metadata:
            t["visibility"] = metadata["visibility"]
    result = pack_context(turns=turns, state=pack_state, **kwargs)
    snapshot = {
        "policy_version": POLICY_VERSION,
        "collection_id": cid,
        "before": before,
        "context_fingerprint": fingerprint,
        "source_revisions": refs,
        "state_hash": digest(packed(state)),
        "state": state,
        "hypotheses": hypotheses,
        "body": result["body"],
        "trace": {k: v for k, v in result["trace"].items() if k != "seconds"},
    }
    serialized = packed(snapshot)
    memory_id = digest(serialized)
    result["trace"]["memory_bytes"] = len(serialized.encode())
    with store.db() as db:
        _ensure_memory(db)
        db.execute("BEGIN IMMEDIATE")
        if _prefix(store, cid, before)[3] != fingerprint:
            raise StaleContextError(
                "Source or accepted state changed while context was compacted. Retry."
            )
        db.execute(
            "INSERT OR IGNORE INTO context_memories VALUES (?,?,?,?,?,?)",
            (memory_id, cid, before, fingerprint, serialized, now()),
        )
    return {
        **result,
        "before": before,
        "state_hash": snapshot["state_hash"],
        "source_revisions": refs,
        "context_fingerprint": fingerprint,
        "memory_id": memory_id,
    }


def load_memory(store, memory_id, *, require_current=True):
    """Old snapshots remain inspectable; stale memories cannot silently be reused."""
    with store.db() as db:
        _ensure_memory(db)
        row = db.execute(
            "SELECT payload FROM context_memories WHERE id=?", (memory_id,)
        ).fetchone()
    if row is None:
        raise ValueError("Context memory not found.")
    snapshot = json.loads(row[0])
    if require_current and not context_is_current(
        store, snapshot["collection_id"], snapshot
    ):
        raise StaleContextError(
            "This context memory predates corrected source or state decisions."
        )
    return snapshot
