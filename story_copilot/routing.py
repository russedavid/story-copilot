"""Select task adapters per request without mutating the server's global state."""

import json
import os
from pathlib import Path


TASKS = {"classifier", "storyteller", "rules", "auditor", "player", "planner"}


def selection(task, configuration=None):
    if task not in TASKS:
        raise ValueError("Unknown model task.")
    if configuration is None:
        path = os.environ.get("STORY_MODEL_ROUTING")
        configuration = (
            json.loads(Path(path).read_text())
            if path
            else {"adapters": [], "tasks": {}}
        )
    adapters = configuration["adapters"]
    ids = [a["id"] for a in adapters]
    if any(type(i) is not int for i in ids) or ids != list(range(len(ids))):
        raise ValueError("Adapter inventory must match the ordered server IDs.")
    chosen = configuration.get("tasks", {}).get(task)
    if chosen is not None and (type(chosen) is not int or chosen not in ids):
        raise ValueError("Task points to an unavailable adapter.")
    if task == "auditor" and chosen is not None:
        raise ValueError("Data screening must use the independent untuned base.")
    # Explicit zeros also avoid runtimes where an omitted or empty list preserves
    # an earlier adapter configuration. Never POST a global scale change.
    return [{"id": i, "scale": 1.0 if i == chosen else 0.0} for i in ids], chosen
