"""Local model configuration. Saving settings never sends a model request."""

import json

from fasthtml.common import *
from starlette.requests import Request
from starlette.responses import RedirectResponse

from .settings import load, save


def register(app, store, page, csrf, validate):
    @app.route("/settings")
    def model_settings(session):
        settings = load(store.home)
        return page(
            "Model settings",
            H1("Model settings"),
            P(
                "Connect your running model server. The local llama.cpp backend also supports a shared base with task-specific LoRA adapters."
            ),
            Form(
                csrf(session),
                Label("Backend", fr="backend"),
                Select(
                    *[
                        Option(
                            label, value=value, selected=value == settings["backend"]
                        )
                        for value, label in [
                            ("llama.cpp", "Local llama.cpp"),
                            ("chat-completions", "Compatible Chat Completions API"),
                        ]
                    ],
                    name="backend",
                    id="backend",
                ),
                Label("API base URL", fr="url"),
                Input(value=settings["url"], name="url", id="url", required=True),
                Label("Model identifier", fr="model"),
                Input(value=settings["model"], name="model", id="model", required=True),
                Label("API key environment variable (optional)", fr="key"),
                Input(
                    value=settings["api_key_env"],
                    name="api_key_env",
                    id="key",
                    placeholder="STORY_API_KEY",
                ),
                Small(
                    "Set the key in the environment that launches this app. Enter its variable name here; never paste the key. A hosted endpoint receives the private context needed for each request."
                ),
                Label("Server context size per request", fr="context"),
                Input(
                    type="number",
                    value=settings["context_limit"],
                    name="context_limit",
                    id="context",
                    min=4096,
                    max=262144,
                ),
                Label("Tokens reserved for the response", fr="output"),
                Input(
                    type="number",
                    value=settings["output_reserve"],
                    name="output_reserve",
                    id="output",
                    min=512,
                    max=8192,
                ),
                Details(
                    Summary("Task adapters (llama.cpp)"),
                    P(
                        "List every server adapter in order. Task values select its numeric ID; unselected adapters receive an explicit zero scale. The evidence planner uses the untuned base."
                    ),
                    Label("Routing JSON", fr="routing"),
                    Textarea(
                        json.dumps(settings["routing"], indent=2),
                        name="routing",
                        id="routing",
                        rows=12,
                    ),
                ),
                P(Button("Save model settings")),
                action="/settings",
                method="post",
            ),
            P(
                "Launch-time STORY_ environment settings override saved values. Compatible servers must support Chat Completions and JSON Schema responses; provider-specific APIs are not interchangeable."
            ),
        )

    @app.route("/settings", methods=["POST"])
    def save_settings(
        request: Request,
        session,
        backend: str,
        url: str,
        model: str,
        context_limit: int,
        output_reserve: int,
        routing: str,
        api_key_env: str = "",
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            save(
                store.home,
                {
                    "backend": backend,
                    "url": url,
                    "model": model,
                    "api_key_env": api_key_env,
                    "context_limit": context_limit,
                    "output_reserve": output_reserve,
                    "routing": json.loads(routing),
                },
            )
            return RedirectResponse("/settings", status_code=303)
        except (ValueError, KeyError, TypeError) as exc:
            return page(
                "Check model settings",
                H1("Couldn’t save those settings"),
                P(str(exc), role="alert"),
                A("Back to settings", href="/settings"),
            )
