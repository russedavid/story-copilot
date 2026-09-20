from __future__ import annotations

from .store import Store


def replay(store: Store, cid: str, before=None, public_only=False):
    """Materialize accepted evidence strictly before a requested target turn."""
    return replay_events(store.events(cid), before, public_only)


def replay_events(events, before=None, public_only=False):
    state = {
        "entities": {},
        "facts": {},
        "claims": [],
        "resources": {},
        "knowledge": {},
        "pending": {},
        "resolutions": [],
        "applied_events": [],
        "stale_events": [],
    }
    rows = [
        e
        for e in events
        if e["status"] == "accepted"
        and (before is None or e["anchor"] < before)
        and (not public_only or e["payload"]["visibility"] == "public")
    ]
    superseded = {
        e["payload"]["supersedes"]
        for e in rows
        if not e["stale"] and e["payload"].get("supersedes")
    }
    for event in rows:
        if event["stale"]:
            state["stale_events"].append(event["id"])
            continue
        if event["id"] in superseded:
            continue
        p = event["payload"]
        if public_only and p["visibility"] == "private":
            continue
        kind, entity, attribute = p["kind"], p["entity"], p["attribute"]
        key = entity + ":" + attribute
        item = {
            "value": p["value"],
            "event": event["id"],
            "visibility": p["visibility"],
        }
        if kind == "entity":
            state["entities"].setdefault(entity, {})[attribute] = item
        elif kind == "fact":
            state["facts"][key] = item
        elif kind == "claim":
            state["claims"].append({"entity": entity, "attribute": attribute, **item})
        elif kind == "knowledge":
            state["knowledge"].setdefault(entity, {})[attribute] = item
        elif kind == "resource":
            prior = state["resources"].get(key, {"value": None, "known_delta": 0})
            if p["delta"] is not None:
                total = None if prior["value"] is None else prior["value"] + p["delta"]
                known_delta = prior["known_delta"] + p["delta"]
            else:
                total, known_delta = p["value"], 0
            state["resources"][key] = {
                **item,
                "value": total,
                "known_delta": known_delta,
            }
        elif kind == "action":
            if p["stage"] != "hypothetical":
                state["pending"][event["id"]] = {
                    "entity": entity,
                    "attribute": attribute,
                    "stage": p["stage"],
                    **item,
                }
        elif kind == "resolve":
            if p["resolves"] not in state["pending"]:
                continue
            state["pending"].pop(p["resolves"])
            state["resolutions"].append({"action": p["resolves"], **item})
        state["applied_events"].append(event["id"])
    return state
