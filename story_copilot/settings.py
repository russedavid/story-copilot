"""Workspace-local model settings; credentials stay in the process environment."""

import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from .routing import TASKS, selection

DEFAULT = {
    "backend": "llama.cpp",
    "url": "http://127.0.0.1:8091/v1",
    "model": "local-model",
    "api_key_env": "",
    "context_limit": 16384,
    "output_reserve": 1800,
    "routing": {"adapters": [], "tasks": {}},
}


def validate(settings):
    if not isinstance(settings, dict) or set(settings) - set(DEFAULT):
        raise ValueError("Unknown model settings.")
    settings = {**DEFAULT, **settings}
    url = urlsplit(settings["url"])
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "Use an HTTP(S) API base URL without credentials, query parameters, or a fragment."
        )
    if settings["backend"] not in {"llama.cpp", "chat-completions"}:
        raise ValueError("Choose llama.cpp or a compatible Chat Completions server.")
    if not isinstance(settings["model"], str) or not settings["model"].strip():
        raise ValueError("Supply the server's model identifier.")
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*|", settings["api_key_env"]):
        raise ValueError("Supply an environment variable name, not an API key.")
    if (
        type(settings["context_limit"]) is not int
        or not 4096 <= settings["context_limit"] <= 262144
    ):
        raise ValueError("Use a supported context size from 4096 to 262144 tokens.")
    if type(settings["output_reserve"]) is not int or not 512 <= settings[
        "output_reserve"
    ] <= min(8192, settings["context_limit"] // 2):
        raise ValueError("Reserve 512–8192 output tokens, at most half the context.")
    routing = settings["routing"]
    if (
        not isinstance(routing, dict)
        or set(routing) != {"adapters", "tasks"}
        or not isinstance(routing["adapters"], list)
        or not isinstance(routing["tasks"], dict)
        or set(routing["tasks"]) - TASKS
    ):
        raise ValueError(
            "Routing needs an ordered adapters list and a map of supported tasks."
        )
    for task in TASKS:
        selection(task, routing)
    if settings["backend"] != "llama.cpp" and routing["adapters"]:
        raise ValueError("Per-request LoRA routing requires the llama.cpp backend.")
    return settings


def load(home):
    path = Path(home) / "model-settings.json"
    settings = json.loads(path.read_text()) if path.exists() else dict(DEFAULT)
    # Explicit launch-time settings may override a saved local configuration.
    for field, env in [
        ("url", "STORY_MODEL_URL"),
        ("model", "STORY_MODEL"),
        ("backend", "STORY_MODEL_BACKEND"),
        ("api_key_env", "STORY_API_KEY_ENV"),
    ]:
        if env in os.environ:
            settings[field] = os.environ[env]
    if os.environ.get("STORY_MODEL_ROUTING"):
        settings["routing"] = json.loads(
            Path(os.environ["STORY_MODEL_ROUTING"]).read_text()
        )
    for field, env in [
        ("context_limit", "STORY_CONTEXT_TOKENS"),
        ("output_reserve", "STORY_OUTPUT_TOKENS"),
    ]:
        if env in os.environ:
            settings[field] = int(os.environ[env])
    return validate(settings)


def save(home, settings):
    settings = validate(settings)
    path = Path(home) / "model-settings.json"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        json.dump(settings, stream, indent=2)
    temporary.replace(path)
    return settings
