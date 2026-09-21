"""Local model configuration. Saving settings never sends a model request."""

import json

from fasthtml.common import *
from starlette.requests import Request
from starlette.responses import RedirectResponse

from .settings import load, save


def register(app, store, page, csrf, validate):
    @app.route("/settings", methods=["GET"])
    def model_settings(session):
        settings = load(store.home)
        planner = settings.get("planner") or {}
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
                        "List every server adapter in order. Task values select its numeric ID; unselected adapters receive an explicit zero scale. Independent review uses the untuned base."
                    ),
                    Label("Routing JSON", fr="routing"),
                    Textarea(
                        json.dumps(settings["routing"], indent=2),
                        name="routing",
                        id="routing",
                        rows=12,
                    ),
                ),
                Details(
                    Summary("Dedicated decision planner"),
                    Label(Input(type="checkbox", name="planner_enabled", checked=bool(planner)), " Use a separate evidence-decision model"),
                    P("A dedicated planner chooses evidence lookups. The main model handles writing and independent review. If the planner is unavailable or its context is too small, the main model takes over and records the reason."),
                    Label("Planner API URL", fr="planner_url"),
                    Input(name="planner_url", id="planner_url", value=planner.get("url", "http://127.0.0.1:8093/v1")),
                    Label("Planner model identifier", fr="planner_model"),
                    Input(name="planner_model", id="planner_model", value=planner.get("model", "local-planner")),
                    Label("Planner context size", fr="planner_context"),
                    Input(type="number", name="planner_context", id="planner_context", min=4096, max=262144, value=planner.get("context_limit",8192)),
                    Label("Planner backend", fr="planner_backend"),
                    Select(*[Option(v,value=v,selected=v==planner.get("backend","llama.cpp")) for v in ["llama.cpp","chat-completions"]],name="planner_backend",id="planner_backend"),
                    Label("Planner credential environment variable", fr="planner_key"),
                    Input(name="planner_key",id="planner_key",value=planner.get("api_key_env","")),
                    Label("Planner adapter inventory and routing", fr="planner_routing"),
                    Textarea(json.dumps(planner.get("routing",{"adapters":[],"tasks":{}}),indent=2),name="planner_routing",id="planner_routing",rows=8),
                    Small("Use task name planner to select the learned adapter. Include every adapter loaded by that server."),
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
        planner_enabled: str = "",
        planner_url: str = "http://127.0.0.1:8093/v1",
        planner_model: str = "local-planner",
        planner_context: int = 8192,
        planner_backend: str = "llama.cpp",
        planner_key: str = "",
        planner_routing: str = '{"adapters":[],"tasks":{}}',
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
                    "planner": {"backend":planner_backend,"url":planner_url,"model":planner_model,
                                "api_key_env":planner_key,"context_limit":planner_context,
                                "output_reserve":700,"routing":json.loads(planner_routing)} if planner_enabled else None,
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
