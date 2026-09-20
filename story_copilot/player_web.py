"""Local controls for player identities, character assignments, and turn traces."""

import json
from fasthtml.common import *
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from .players import Players
from .settings import load as model_settings
from .store import digest, packed


def register(app, store, *, page, csrf, validate, workers, schedule):
    players = Players(store)
    campaigns = players.campaigns
    app.state.players = players

    def form(session, action, *children):
        return Form(csrf(session), *children, action=action, method="post")

    def error(exc, back):
        return page(
            "Player request",
            H1("Couldn’t complete that"),
            P(str(exc), role="alert"),
            A("Go back", href=back),
        )

    def adapter_select(current=None, field_id="adapter"):
        adapters = model_settings(store.home)["routing"]["adapters"]
        return Select(
            Option(
                "Base model + personality prompt", value="", selected=current is None
            ),
            *[
                Option(
                    f"Adapter {a['id']} · "
                    + str(a.get("candidate", a.get("task", "loaded adapter"))),
                    value=str(a["id"]),
                    selected=a["id"] == current,
                )
                for a in adapters
            ],
            name="adapter_id",
            id=field_id,
        )

    def profile_form(session, profile=None):
        profile = profile or {
            "id": "",
            "name": "",
            "instructions": "",
            "adapter_id": None,
            "revision": 0,
        }
        prefix = "player-" + (profile["id"] or "new")
        return form(
            session,
            "/players/save",
            Hidden(profile["id"], name="identifier"),
            Hidden(str(profile["revision"]), name="revision"),
            Label("Player identity", fr=prefix + "-name"),
            Input(
                name="name", id=prefix + "-name", value=profile["name"], required=True
            ),
            Label("Personality and approach", fr=prefix + "-personality"),
            Textarea(
                profile["instructions"],
                name="instructions",
                id=prefix + "-personality",
                rows=5,
                required=True,
            ),
            Label("Model adapter", fr=prefix + "-adapter"),
            adapter_select(profile["adapter_id"], prefix + "-adapter"),
            P(
                "A player personality can be assigned different characters. A base-only profile uses prompting; select a trained adapter when one is loaded in your model server."
            ),
            P(Button("Save player identity")),
        )

    @app.route("/players", methods=["GET"])
    def player_profiles(session):
        return page(
            "Player agents",
            H1("Player agents"),
            P(
                "Create distinct player personalities, assign their characters, and ask them to contribute one turn at a time. You remain the facilitator."
            ),
            *[
                Article(
                    H2(profile["name"]),
                    P(profile["instructions"]),
                    Small(
                        "Prompted base"
                        if profile["adapter_id"] is None
                        else f"Own adapter · server slot {profile['adapter_id']}"
                    ),
                    Details(
                        Summary("Edit player identity"), profile_form(session, profile)
                    ),
                    cls="card",
                )
                for profile in players.profiles()
            ],
            Article(H2("New player identity"), profile_form(session), cls="card"),
            A("Assign players to a campaign →", href="/campaigns"),
        )

    @app.route("/players/save", methods=["POST"])
    def save_profile(
        request: Request,
        session,
        name: str,
        instructions: str,
        adapter_id: str = "",
        identifier: str = "",
        revision: int = 0,
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            players.save_profile(
                name,
                instructions,
                adapter_id=int(adapter_id) if adapter_id else None,
                identifier=identifier or None,
                expected_revision=revision if identifier else None,
            )
            return RedirectResponse("/players", 303)
        except (ValueError, TypeError) as exc:
            return error(exc, "/players")

    @app.route("/campaigns/{campaign_id}/players", methods=["GET"])
    def assignments(campaign_id: str, session):
        campaign = campaigns.campaign(campaign_id)
        characters = campaigns.characters(campaign_id)
        profiles = players.profiles()
        return page(
            "Assign players",
            A("← Campaign", href=f"/campaigns/{campaign_id}"),
            H1("Players for " + campaign["title"]),
            P(
                "An agent receives its assigned starting sheet, shared documents, shared conversation, and messages addressed to its character. Put facilitator-only secrets in private campaign material."
            ),
            *[
                Article(
                    H2(
                        next(
                            c["name"]
                            for c in characters
                            if c["id"] == binding["character_id"]
                        )
                    ),
                    P(
                        players.profile(binding["profile_id"])["name"]
                        + (" · active" if binding["active"] else " · paused")
                    ),
                    cls="card",
                )
                for binding in players.bindings(campaign_id)
            ],
            form(
                session,
                f"/campaigns/{campaign_id}/players/assign",
                Label("Player identity", fr="profile"),
                Select(
                    *[Option(p["name"], value=p["id"]) for p in profiles],
                    name="profile_id",
                    id="profile",
                    required=True,
                ),
                Label("Character", fr="character"),
                Select(
                    *[Option(c["name"], value=c["id"]) for c in characters],
                    name="character_id",
                    id="character",
                    required=True,
                ),
                Label("Participation", fr="active"),
                Select(
                    Option("Active", value="1"),
                    Option("Paused", value="0"),
                    name="active",
                    id="active",
                ),
                P(Button("Assign player")),
            ),
            P(A("Create or edit player identities", href="/players")),
        )

    @app.route("/campaigns/{campaign_id}/players/assign", methods=["POST"])
    def assign(
        request: Request,
        session,
        campaign_id: str,
        profile_id: str,
        character_id: str,
        active: int = 1,
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            players.assign(campaign_id, profile_id, character_id, active=bool(active))
            return RedirectResponse(f"/campaigns/{campaign_id}/players", 303)
        except ValueError as exc:
            return error(exc, f"/campaigns/{campaign_id}/players")

    def panel_version(session_id):
        cid = campaigns.session(session_id)["campaign_id"]
        bindings = players.bindings(cid)
        return digest(
            packed(
                {
                    "bindings": bindings,
                    "profiles": [
                        (b["profile_id"], players.profile(b["profile_id"])["revision"])
                        for b in bindings
                    ],
                    "runs": [
                        (r["id"], r["status"], r["finished"])
                        for r in players.runs(session_id)
                    ],
                }
            )
        )

    def panel(session_id, session):
        campaign_id = campaigns.session(session_id)["campaign_id"]
        characters = {c["id"]: c for c in campaigns.characters(campaign_id)}
        cards = []
        for binding in players.bindings(campaign_id):
            if not binding["active"]:
                continue
            profile = players.profile(binding["profile_id"])
            runs = players.runs(session_id, binding["id"])
            active = runs and runs[0]["status"] in {"queued", "running"}
            cards.append(
                Div(
                    H3(
                        profile["name"]
                        + " · "
                        + characters[binding["character_id"]]["name"]
                    ),
                    form(
                        session,
                        f"/play/{session_id}/player/{binding['id']}/turn",
                        Button("Take a turn", disabled=active),
                    ),
                    P(runs[0]["status"].replace("_", " ")) if runs else None,
                    P(runs[0]["result"].get("error", ""), role="alert")
                    if runs and runs[0]["result"].get("error")
                    else None,
                    form(
                        session,
                        f"/play/{session_id}/player-run/{runs[0]['id']}/cancel",
                        Button("Cancel turn", cls="secondary"),
                    )
                    if active
                    else None,
                    A(
                        "View this character’s perspective",
                        href=f"/play/{session_id}/player/{binding['id']}/perspective",
                    ),
                    Details(
                        Summary("Turn history and traces"),
                        *[
                            Details(
                                Summary(run["status"] + " · " + run["created"][:19]),
                                P(
                                    run["result"]
                                    .get("player", {})
                                    .get("answer", {})
                                    .get("utterance", "")
                                ),
                                Details(
                                    Summary("Evidence, model and timing"),
                                    Pre(json.dumps(run, indent=2, ensure_ascii=False)),
                                ),
                            )
                            for run in runs[:8]
                        ],
                    ),
                )
            )
        return Article(
            H2("Player agents"),
            P(
                "Requested player turns are recorded as AI participant speech. They do not determine world outcomes; facilitator suggestions remain private drafts."
            ),
            *cards,
            A("Assign or pause players", href=f"/campaigns/{campaign_id}/players"),
            cls="card",
            id="player-panel",
            hx_get=f"/play/{session_id}/players?version={panel_version(session_id)}",
            hx_trigger="every 3s",
            hx_swap="outerHTML",
        )

    app.state.player_panel = panel

    @app.route("/play/{session_id}/players", methods=["GET"])
    def player_panel(session_id: str, session, version: str = ""):
        return (
            Response(status_code=204)
            if version == panel_version(session_id)
            else panel(session_id, session)
        )

    @app.route("/play/{session_id}/player/{binding_id}/turn", methods=["POST"])
    def take_turn(
        request: Request,
        session,
        session_id: str,
        binding_id: str,
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            players.request(
                session_id, binding_id, workers=workers, on_complete=schedule
            )
            return RedirectResponse(f"/play/{session_id}", 303)
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @app.route("/play/{session_id}/player-run/{run_id}/cancel", methods=["POST"])
    def cancel(
        request: Request, session, session_id: str, run_id: str, csrf_token: str = ""
    ):
        try:
            validate(request, session, csrf_token)
            players.cancel(session_id, run_id)
            return RedirectResponse(f"/play/{session_id}", 303)
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @app.route("/play/{session_id}/player/{binding_id}/perspective", methods=["GET"])
    def perspective(session_id: str, binding_id: str):
        try:
            view = players.perspective(session_id, binding_id)
            return page(
                "Character perspective",
                A("← Session", href=f"/play/{session_id}"),
                H1(view["profile"]["name"] + " playing " + view["character"]["name"]),
                P(
                    "This view shows the source material this player can access. It excludes facilitator notes, other characters’ sheets, and private conversation addressed elsewhere."
                ),
                H2("Starting character sheet"),
                Pre(json.dumps(view["character"]["sheet"], indent=2)),
                H2("Shared material"),
                *[
                    Details(Summary(d["title"]), Pre(d["text"]))
                    for d in view["documents"]
                ],
                H2("Visible conversation"),
                *[
                    Article(
                        Strong(m["speaker"] + " · " + m["visibility"]),
                        P(m["text"]),
                        cls="card",
                    )
                    for m in view["messages"]
                ],
            )
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")
