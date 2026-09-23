"""Campaign UI routes, sharing the library's local app and CSRF policy."""

from __future__ import annotations

import json
import math
import os
import secrets
import shlex
from pathlib import Path
from threading import Condition
from urllib.parse import urlencode
from uuid import uuid4

from fasthtml.common import (
    H1,
    H2,
    H3,
    A,
    Article,
    Audio,
    Button,
    Details,
    Div,
    Form,
    Hidden,
    Input,
    Label,
    Option,
    P,
    Pre,
    Select,
    Small,
    Span,
    Strong,
    Summary,
    Textarea,
)
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

from .campaigns import KINDS, MAX_IMPORT_BYTES, Campaigns, imported_text
from .store import digest, packed


def campaign_audio_clip(campaigns, queue, session_id, message_id):
    """Resolve only a retained source chunk attached to this private message.

    Inherited recordings must have a valid message/session ancestry. The source
    may point to another audio-session ID, but never to an arbitrary disk path.
    """
    requested = campaigns.session(session_id)
    message = next(
        (m for m in campaigns.messages(session_id) if m["id"] == message_id), None
    )
    if message is None or message["source"].get("kind") != "live_audio":
        raise ValueError("No captured audio is attached to this session message.")
    source = message["source"]
    chunk_id = source.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("This message has no retained chunk reference.")
    origin = message
    visited = set()
    while origin["source"].get("branched_from_message"):
        if origin["id"] in visited or len(visited) >= 100:
            raise ValueError("Invalid inherited audio ancestry.")
        visited.add(origin["id"])
        parent_id = origin["source"].get("branched_from_session")
        owner = campaigns.session(origin["session_id"])
        if not parent_id or owner["parent_id"] != parent_id:
            raise ValueError(
                "The source recording is not from this session's ancestry."
            )
        parent = campaigns.session(parent_id)
        if parent["campaign_id"] != requested["campaign_id"]:
            raise ValueError("A recording cannot cross campaigns.")
        parent_message = next(
            (
                m
                for m in campaigns.messages(parent_id)
                if m["id"] == origin["source"]["branched_from_message"]
            ),
            None,
        )
        if (
            parent_message is None
            or parent_message["source"].get("chunk_id") != chunk_id
        ):
            raise ValueError(
                "The inherited recording reference does not match its source."
            )
        origin = parent_message
    audio_session = (
        origin["source"].get("transcription", {}).get("session") or origin["session_id"]
    )
    clip = queue.audio_clip(chunk_id, audio_session)
    if clip["audio_sha256"] != source.get("audio_sha256") or clip[
        "channel"
    ] != source.get("channel"):
        raise ValueError("The retained audio differs from this message's provenance.")
    duration = clip["end"] - clip["start"]
    start = source.get("start")
    end = source.get("end")
    finite = lambda value: (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
    timed = finite(start)
    if timed and not clip["start"] - 0.02 <= start <= clip["end"] + 0.02:
        raise ValueError("The message timing falls outside its recorded chunk.")
    seek_start = min(duration, max(0, start - clip["start"])) if timed else 0
    seek_end = (
        min(duration, max(seek_start, end - clip["start"]))
        if timed and finite(end) and end > start
        else None
    )
    return {**clip, "seek_start": seek_start, "seek_end": seek_end, "timed": timed}


def wav_response(data, range_header=None):
    """Small retained WAVs support browser seeking without creating public files."""
    headers = {
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff",
        "Accept-Ranges": "bytes",
    }
    if not range_header:
        return Response(data, media_type="audio/wav", headers=headers)
    try:
        unit, part = range_header.split("=", 1)
        if unit != "bytes" or "," in part:
            raise ValueError()
        first, last = part.split("-", 1)
        if first:
            start = int(first)
            end = min(len(data) - 1, int(last)) if last else len(data) - 1
        else:
            suffix = int(last)
            if suffix <= 0:
                raise ValueError()
            start = max(0, len(data) - suffix)
            end = len(data) - 1
        if start < 0 or start >= len(data) or end < start:
            raise ValueError()
    except (ValueError, AttributeError):
        return Response(
            status_code=416,
            headers={**headers, "Content-Range": f"bytes */{len(data)}"},
        )
    headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
    return Response(
        data[start : end + 1], status_code=206, media_type="audio/wav", headers=headers
    )


class CampaignScheduler:
    """One bounded model slice per submission, fairly requeued on the shared worker.

    Source work lives in persisted run traces. This in-memory reservation prevents
    concurrent jobs for a session; Campaigns also checks its durable running row.
    Failed or non-progressing backlogs stop visibly instead of spinning forever.
    """

    def __init__(self, campaigns, workers, generate, build_context=None):
        self.campaigns, self.workers = campaigns, workers
        self.generate, self.build_context = generate, build_context
        self._condition = Condition()
        self._scheduled, self._dirty = set(), {}
        self.errors = {}

    def is_pending(self, session_id):
        with self._condition:
            return session_id in self._scheduled

    def wait_idle(self, timeout=10):
        """Test/controlled-shutdown helper; never used to block a web request."""
        with self._condition:
            return self._condition.wait_for(lambda: not self._scheduled, timeout)

    def schedule(self, session_id, force=False):
        if self.generate is None or (
            not force and not self.campaigns.session(session_id)["proactive"]
        ):
            return False
        with self._condition:
            if session_id in self._scheduled:
                self._dirty[session_id] = self._dirty.get(session_id, False) or force
                return False
            self._scheduled.add(session_id)
        self._submit(session_id, force)
        return True

    def _submit(self, session_id, force):
        try:
            self.workers.submit(self._work, session_id, force)
        except RuntimeError as exc:
            with self._condition:
                self.errors[session_id] = str(exc)
                self._scheduled.discard(session_id)
                self._dirty.pop(session_id, None)
                self._condition.notify_all()

    def _work(self, session_id, force):
        run, failure = None, None
        with self._condition:
            pending_force = self._dirty.pop(session_id, False)
            force = force or pending_force
        try:
            # A pause while this slice was queued prevents any model request.
            if force or self.campaigns.session(session_id)["proactive"]:
                run_id = self.campaigns.generate(
                    session_id,
                    self.generate,
                    force=force,
                    build_context=self.build_context,
                )
                if run_id:
                    run = next(
                        r for r in self.campaigns.runs(session_id) if r["id"] == run_id
                    )
        except Exception as exc:  # noqa: BLE001 - preserve scheduler failure and release reservation
            failure = str(exc)
        finally:
            analysis_more = bool(
                run
                and run["status"] == "complete"
                and run["result"]
                .get("trace", {})
                .get("classification", {})
                .get("continue_automatically")
            )
            stale = bool(run and run["status"] in {"stale", "superseded"})
            with self._condition:
                if failure:
                    self.errors[session_id] = failure
                else:
                    self.errors.pop(session_id, None)
                changed = self._dirty.pop(session_id, None)
                followup_force = bool(changed)
                proactive = self.campaigns.session(session_id)["proactive"]
                followup = not failure and (
                    followup_force
                    or proactive
                    and (changed is not None or analysis_more or stale)
                )
                if not followup:
                    self._scheduled.discard(session_id)
                    self._condition.notify_all()
            if followup:
                # Retain the session reservation across requeue. Other sessions
                # already queued get their turn before this next bounded slice.
                self._submit(session_id, followup_force)


def register_campaign_routes(
    app, store, *, page, csrf, validate, workers, generate=None, build_context=None
):
    """Register without starting audio or making model calls.

    generate(context) -> {'suggestions': [{kind,title,text,visibility,payload,evidence}],
                          'trace': {...}}
    build_context(snapshot) -> a JSON-serializable model input. The original source
    messages and proposal evidence remain independently navigable in this UI.
    """
    campaigns = Campaigns(store)
    app.state.campaigns = campaigns
    rt = app.route
    scheduler = CampaignScheduler(campaigns, workers, generate, build_context)
    app.state.campaign_scheduler = scheduler
    schedule = scheduler.schedule

    def options(values, current):
        return [
            Option(label, value=value, selected=value == current)
            for value, label in values
        ]

    def visibility_select(current="private"):
        return Select(
            *options(
                [("private", "Facilitator only"), ("public", "Player visible")], current
            ),
            name="visibility",
        )

    def audience_select(campaign_id, current="table"):
        return Select(
            *options(
                [("table", "All players"), ("facilitator", "Facilitator only")]
                + [
                    (c["id"], c["name"] + " only")
                    for c in campaigns.characters(campaign_id)
                ],
                current,
            ),
            name="recipient",
        )

    def post_form(session, action, *content, **kwargs):
        # Associate visible labels with controls, including repeated sheet forms.
        for index, item in enumerate(content[:-1]):
            control = content[index + 1]
            if (
                getattr(item, "tag", None) == "label"
                and "for" not in item.attrs
                and getattr(control, "tag", None) in {"input", "textarea", "select"}
            ):
                control.attrs.setdefault("id", "field-" + uuid4().hex)
                item.attrs["for"] = control.attrs["id"]
        return Form(csrf(session), *content, action=action, method="post", **kwargs)

    def redirect(path):
        return RedirectResponse(path, status_code=303)

    def error(exc, back="/campaigns"):
        return page(
            "Couldn’t save",
            H1("Couldn’t save that"),
            P(str(exc), role="alert"),
            A("Go back", href=back),
        )

    def changed(campaign_id):
        current = campaigns.current_session(campaign_id)
        if current:
            schedule(current["id"])

    @rt("/campaigns")
    def campaign_index(session):
        return page(
            "Campaigns",
            H1("Facilitator’s table"),
            P(
                "Run the game your way. Record what happened, ask for alternatives, and decide what becomes part of the story."
            ),
            Article(
                H2("Try an example"),
                P(
                    "Open an original coastal mystery with two characters already asking for help. Suggestions start when you request them."
                ),
                post_form(session, "/campaigns/example", Button("Try an example")),
                cls="card",
            ),
            *[
                A(
                    H2(c["title"]),
                    Small(c["system"]),
                    href=f"/campaigns/{c['id']}",
                    cls="story",
                )
                for c in campaigns.campaigns()
            ],
            Article(
                H2("New campaign"),
                post_form(
                    session,
                    "/campaigns/create",
                    Label("Campaign name", fr="campaign-title"),
                    Input(name="title", id="campaign-title", required=True),
                    Label("Game system", fr="campaign-system"),
                    Input(name="system", id="campaign-system", value="Custom rules"),
                    Label(
                        "What kind of game do you want to run?", fr="campaign-direction"
                    ),
                    Textarea(
                        name="direction",
                        id="campaign-direction",
                        placeholder="Your priorities, tone, boundaries, and the direction you want the story to take.",
                        rows=4,
                    ),
                    P(Button("Create campaign")),
                ),
                cls="card",
            ),
        )

    @rt("/campaigns/example", methods=["POST"])
    def example(request: Request, session, csrf_token: str = ""):
        try:
            validate(request, session, csrf_token)
            from .demo import ensure_demo

            demo = ensure_demo(campaigns)
            return redirect(f"/play/{demo['session_id']}")
        except ValueError as exc:
            return error(exc)

    @rt("/campaigns/create", methods=["POST"])
    def create_campaign(
        request: Request,
        session,
        title: str,
        system: str = "Custom rules",
        direction: str = "",
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            cid = campaigns.create(title, system, direction)
            return redirect(f"/campaigns/{cid}")
        except ValueError as exc:
            return error(exc)

    def character_form(campaign_id, session, char=None):
        char = char or {
            "id": "",
            "name": "",
            "visibility": "public",
            "revision": 0,
            "sheet": {
                "occupation": "",
                "skills": {},
                "resources": {},
                "notes": "",
            },
        }
        return post_form(
            session,
            f"/campaigns/{campaign_id}/character",
            Hidden(char["id"], name="character_id"),
            Hidden(str(char["revision"]), name="revision"),
            Label("Character name"),
            Input(name="name", value=char["name"], required=True),
            Label("Who may see this sheet?"),
            visibility_select(char["visibility"]),
            Label("Character sheet (JSON)"),
            Textarea(json.dumps(char["sheet"], indent=2), name="sheet", rows=12),
            Small(
                "This is the starting sheet. Record changes during play in the conversation; editing this sheet changes the baseline. Use integer resource totals, or null when unknown. Other fields are yours to define."
            ),
            P(Button("Save character")),
        )

    @rt("/campaigns/{campaign_id}")
    def campaign_detail(campaign_id: str, session):
        try:
            c = campaigns.campaign(campaign_id)
            sessions = campaigns.sessions(campaign_id)
            current = campaigns.current_session(campaign_id)
            chars = campaigns.characters(campaign_id)
            from .rules import active_documents
            documents = campaigns.documents(campaign_id)
            active_ids = {d["id"] for d in active_documents({"documents": documents})}
            from .demo import DISCLOSURE, demo_info

            authored_demo = demo_info(campaigns, campaign_id)
            return page(
                c["title"],
                A("← Campaigns", href="/campaigns"),
                H1(c["title"]),
                P(DISCLOSURE, cls="notice") if authored_demo else None,
                Div(
                    Div(
                        Article(
                            H2("Sessions"),
                            *[
                                P(
                                    A(
                                        f"{s['ordinal']}. {s['title']}",
                                        href=f"/play/{s['id']}",
                                    ),
                                    Small(
                                        " · "
                                        + {
                                            "fresh": "fresh start",
                                            "continue": "continuation",
                                            "branch": "alternative branch",
                                            "legacy": "existing session",
                                        }[s["kind"]]
                                    ),
                                    Span("Current story", cls="badge")
                                    if current and current["id"] == s["id"]
                                    else None,
                                )
                                for s in sessions
                            ],
                            post_form(
                                session,
                                f"/campaigns/{campaign_id}/session",
                                Label("Session title"),
                                Input(
                                    name="title",
                                    value=f"Session {len(sessions) + 1}",
                                    required=True,
                                ),
                                Label("How should this session begin?"),
                                Select(
                                    *options(
                                        [
                                            (
                                                "continue",
                                                "Continue — carry the chosen story forward",
                                            ),
                                            (
                                                "branch",
                                                "Branch — explore an alternative",
                                            ),
                                            (
                                                "fresh",
                                                "Fresh — start without prior play",
                                            ),
                                        ],
                                        "continue" if current else "fresh",
                                    ),
                                    name="mode",
                                ),
                                Label("Source session (for Continue or Branch)"),
                                Select(
                                    Option(
                                        "Choose a source session",
                                        value="",
                                        selected=current is None,
                                    ),
                                    *[
                                        Option(
                                            s["title"]
                                            + (
                                                " · current story"
                                                if current and current["id"] == s["id"]
                                                else " · " + s["kind"]
                                            ),
                                            value=s["id"],
                                            selected=bool(
                                                current and current["id"] == s["id"]
                                            ),
                                        )
                                        for s in sessions
                                    ],
                                    name="parent_id",
                                ),
                                P(
                                    "Continue carries conversation, observed state, resources, and character knowledge into the next session and makes it the current story. Branch copies that history for an alternative without changing the current story. Fresh starts a new current story with the campaign’s shared scenario and starting character sheets, but no prior conversation or session changes."
                                ),
                                P(Button("Open session")),
                            ),
                            cls="card",
                        ),
                        Article(
                            H2("Scenario and reference material"),
                            P(
                                "Scenario secrets are Facilitator-only by default. Player-facing material must be explicitly marked player visible."
                            ),
                            *[
                                Details(
                                    Summary(d["title"], " · ", d["visibility"],
                                            " · superseded" if d["id"] not in active_ids else ""),
                                    Pre(d["text"]),
                                    post_form(
                                        session,
                                        f"/campaigns/{campaign_id}/document/{d['id']}/remove",
                                        Button(
                                            "Remove from active context",
                                            cls="secondary",
                                        ),
                                    ),
                                )
                                for d in documents
                            ],
                            post_form(
                                session,
                                f"/campaigns/{campaign_id}/document",
                                Label("Title"),
                                Input(name="title", required=True),
                                Label("Visibility"),
                                visibility_select(),
                                Label("Material type"),
                                Select(
                                    Option("Scenario / reference", value="reference"),
                                    Option("Rules for this campaign", value="rules"),
                                    name="kind",
                                ),
                                Label("Replaces earlier material (optional)"),
                                Select(Option("New material", value=""),
                                       *[Option(d["title"],value=d["id"]) for d in documents if d["id"] in active_ids],
                                       name="supersedes"),
                                Small("A replacement keeps the earlier text in history and removes it from current retrieval. Choose the same material type."),
                                Label("Paste text, or choose a file"),
                                Textarea(name="text", rows=7),
                                Input(
                                    type="file",
                                    name="upload",
                                    aria_label="Choose scenario or reference file",
                                    accept=".txt,.md,.json,.csv,.pdf",
                                ),
                                Small(
                                    "Text, Markdown, JSON, CSV, or searchable PDF. Files stay in your local data store."
                                ),
                                P(Button("Add material")),
                                enctype="multipart/form-data",
                            ),
                            cls="card",
                        ),
                        cls="",
                    ),
                    Div(
                        Article(
                            H2("Your direction"),
                            post_form(
                                session,
                                f"/campaigns/{campaign_id}/settings",
                                Label("Campaign name"),
                                Input(name="title", value=c["title"], required=True),
                                Label("Facilitator direction (private)"),
                                Textarea(c["direction"], name="direction", rows=5),
                                Label("Style and boundaries (private)"),
                                Textarea(c["style"], name="style", rows=5),
                                P(Button("Save direction")),
                            ),
                            cls="card",
                        ),
                        Article(
                            H2("Rule tools and terminology"),
                            P(
                                "Rules come from the material you attach. Optionally define resource aliases, search aliases, and bounded arithmetic tools for this campaign."
                            ),
                            Details(
                                Summary("Edit rule profile"),
                                post_form(
                                    session,
                                    f"/campaigns/{campaign_id}/profile",
                                    Label("Rule profile (JSON)"),
                                    Textarea(
                                        json.dumps(
                                            campaigns.rule_profile(campaign_id),
                                            indent=2,
                                        ),
                                        name="profile",
                                        rows=12,
                                    ),
                                    Small(
                                        "Tool operations: sum, difference, product, quotient, less, less_equal, equal, greater_equal, greater. Each tool needs name, description, and operation."
                                    ),
                                    P(Button("Save rule profile")),
                                ),
                            ),
                            cls="card",
                        ),
                        Article(
                            H2("Characters"),
                            *[
                                Details(
                                    Summary(ch["name"]),
                                    character_form(campaign_id, session, ch),
                                )
                                for ch in chars
                            ],
                            Details(
                                Summary("Add a character"),
                                character_form(campaign_id, session),
                            ),
                            cls="card",
                        ),
                        Article(
                            H2("People and speakers"),
                            *[
                                P(
                                    Strong(p["name"]),
                                    f" · {p['role']} · {p['speaker']}",
                                    Small(
                                        " → "
                                        + next(
                                            (
                                                ch["name"]
                                                for ch in chars
                                                if ch["id"] == p["character_id"]
                                            ),
                                            "No character assigned",
                                        )
                                    ),
                                )
                                for p in campaigns.participants(campaign_id)
                            ],
                            post_form(
                                session,
                                f"/campaigns/{campaign_id}/participant",
                                Label("Person’s name"),
                                Input(name="name", required=True),
                                Label(
                                    "Exact speaker label (including audio channel prefix, if present)"
                                ),
                                Input(
                                    name="speaker",
                                    required=True,
                                    placeholder="SPEAKER_00, mic:SPEAKER_00, or system:SPEAKER_00",
                                ),
                                Label("Role"),
                                Select(
                                    *options(
                                        [
                                            ("player", "Player"),
                                            ("facilitator", "Facilitator"),
                                            ("unknown", "Unknown"),
                                        ],
                                        "player",
                                    ),
                                    name="role",
                                ),
                                Label("Character"),
                                Select(
                                    Option("Not assigned", value=""),
                                    *[
                                        Option(ch["name"], value=ch["id"])
                                        for ch in chars
                                    ],
                                    name="character_id",
                                ),
                                P(Button("Save speaker mapping")),
                            ),
                            Small(
                                "An audio channel does not prove who is speaking. Your mapping identifies future messages and can ground the actor in existing unknown speech without rewriting its original attribution. Changing a mapping invalidates state proposals that relied on it."
                            ),
                            cls="card",
                        ),
                    ),
                    cls="grid",
                ),
            )
        except ValueError as exc:
            return error(exc)

    @rt("/campaigns/{campaign_id}/profile", methods=["POST"])
    def profile_settings(
        request: Request, session, campaign_id: str, profile: str, csrf_token: str = ""
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.set_rule_profile(campaign_id, json.loads(profile))
            changed(campaign_id)
            return redirect(f"/campaigns/{campaign_id}")
        except (ValueError, TypeError, KeyError) as exc:
            return error(exc, f"/campaigns/{campaign_id}")

    @rt("/campaigns/{campaign_id}/settings", methods=["POST"])
    def settings(
        request: Request,
        session,
        campaign_id: str,
        title: str,
        direction: str = "",
        style: str = "",
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.update(campaign_id, title=title, direction=direction, style=style)
            changed(campaign_id)
            return redirect(f"/campaigns/{campaign_id}")
        except ValueError as exc:
            return error(exc, f"/campaigns/{campaign_id}")

    @rt("/campaigns/{campaign_id}/session", methods=["POST"])
    def new_session(
        request: Request,
        session,
        campaign_id: str,
        title: str,
        parent_id: str = "",
        mode: str = "fresh",
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            if mode not in {"fresh", "continue", "branch"}:
                raise ValueError("Choose Continue, Branch, or Fresh.")
            if parent_id and campaigns.session(parent_id)["campaign_id"] != campaign_id:
                raise ValueError("Choose a session from this campaign.")
            if mode == "continue":
                sid = campaigns.continue_session(
                    campaign_id, title, parent_id=parent_id or None
                )
            elif mode == "branch":
                if not parent_id:
                    raise ValueError("Choose the source session for this branch.")
                sid = campaigns.branch(parent_id, title)
            else:
                sid = campaigns.create_session(campaign_id, title)
            return redirect(f"/play/{sid}")
        except ValueError as exc:
            return error(exc, f"/campaigns/{campaign_id}")

    @rt("/campaigns/{campaign_id}/document", methods=["POST"])
    async def add_document(request: Request, session, campaign_id: str):
        try:
            form = await request.form()
            validate(request, session, str(form.get("csrf_token", "")))
            text = str(form.get("text", ""))
            upload = form.get("upload")
            metadata = {"source": "pasted"}
            if upload is not None and getattr(upload, "filename", ""):
                data = await upload.read(MAX_IMPORT_BYTES + 1)
                if text.strip():
                    raise ValueError("Paste text or upload a file, one at a time.")
                text = imported_text(upload.filename, data)
                metadata = {
                    "source": "uploaded",
                    "filename": Path(upload.filename).name,
                }
            kind = str(form.get("kind", "reference"))
            if kind not in {"rules", "reference"}:
                raise ValueError("Choose rule or reference material.")
            metadata["kind"] = kind
            if str(form.get("supersedes", "")).strip():
                metadata["supersedes"] = str(form["supersedes"])
            campaigns.add_document(
                campaign_id,
                str(form.get("title", "")),
                text,
                visibility=str(form.get("visibility", "private")),
                metadata=metadata,
            )
            changed(campaign_id)
            return redirect(f"/campaigns/{campaign_id}")
        except ValueError as exc:
            return error(exc, f"/campaigns/{campaign_id}")

    @rt("/campaigns/{campaign_id}/document/{document_id}/remove", methods=["POST"])
    def remove_document(
        request: Request,
        session,
        campaign_id: str,
        document_id: str,
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.delete_document(campaign_id, document_id)
            changed(campaign_id)
            return redirect(f"/campaigns/{campaign_id}")
        except ValueError as exc:
            return error(exc, f"/campaigns/{campaign_id}")

    @rt("/campaigns/{campaign_id}/character", methods=["POST"])
    def character(
        request: Request,
        session,
        campaign_id: str,
        name: str,
        sheet: str,
        character_id: str = "",
        revision: int = 0,
        visibility: str = "public",
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.save_character(
                campaign_id,
                name,
                json.loads(sheet),
                character_id=character_id or None,
                expected_revision=revision if character_id else None,
                visibility=visibility,
            )
            changed(campaign_id)
            return redirect(f"/campaigns/{campaign_id}")
        except ValueError as exc:
            return error(exc, f"/campaigns/{campaign_id}")

    @rt("/campaigns/{campaign_id}/participant", methods=["POST"])
    def participant(
        request: Request,
        session,
        campaign_id: str,
        name: str,
        speaker: str,
        role: str = "player",
        character_id: str = "",
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.map_participant(
                campaign_id, name, speaker, role=role, character_id=character_id or None
            )
            changed(campaign_id)
            return redirect(f"/campaigns/{campaign_id}")
        except ValueError as exc:
            return error(exc, f"/campaigns/{campaign_id}")

    def working_state_panel(session_id, state=None):
        state = campaigns.state(session_id) if state is None else state
        return Article(
            H2("Game state from the conversation"),
            *[
                P(Strong(p["entity"]), " · ", p["attribute"], ": ", str(p["value"]))
                for p in state["pending"].values()
            ],
            P("No unresolved actions observed.") if not state["pending"] else None,
            Details(
                Summary("Working state and resources"), Pre(json.dumps(state, indent=2))
            ),
            cls="card",
            id="working-game-state",
            hx_get=f"/play/{session_id}/state?version={digest(packed(state))}",
            hx_trigger="every 5s",
            hx_swap="outerHTML",
        )

    @rt("/play/{session_id}/state")
    def game_state(session_id: str, version: str = ""):
        state = campaigns.state(session_id)
        if version == digest(packed(state)):
            return Response(status_code=204)
        return working_state_panel(session_id, state)

    def proposal_card(proposal, session_id, session):
        observation = bool(proposal.get("observed") or proposal["kind"] == "state")
        return Article(
            Div(
                Strong(proposal["title"]),
                Span(proposal["kind"], cls="badge"),
                Small("From conversation" if observation else "Private suggestion"),
                cls="row",
            ),
            Small(proposal["created"][:19].replace("T", " ") + " UTC"),
            Small(f"As of source #{proposal['payload']['as_of_source']['ordinal']}")
            if proposal["payload"].get("as_of_source")
            else None,
            P(
                "The source changed; this interpretation is no longer active.",
                cls="notice",
            )
            if proposal["stale"]
            else None,
            P(
                proposal["text"],
                style="white-space:pre-wrap",
                data_guidance_text="true",
            ),
            *[
                P(
                    A(
                        "Review or correct source",
                        href=f"/play/{session_id}/message/{e['message_id']}",
                    ),
                    " — ",
                    e["quote"],
                )
                for e in proposal["evidence"]
            ],
            Details(
                Summary("State interpretation"),
                Pre(json.dumps(proposal["payload"], indent=2)),
            )
            if observation
            else None,
            Div(
                Button(
                    "Copy wording",
                    type="button",
                    cls="secondary",
                    onclick="navigator.clipboard.writeText(this.closest('article').querySelector('[data-guidance-text]').textContent)",
                ),
                post_form(
                    session,
                    f"/play/{session_id}/proposal/{proposal['id']}",
                    Hidden("rejected", name="action"),
                    Button("Reject and refresh"),
                )
                if proposal["status"] != "rejected"
                else Small("Rejected"),
                cls="row",
            )
            if not observation
            else None,
            cls="card",
            id="proposal-" + proposal["id"],
        )

    def decision_card(run):
        decision = run["result"].get("trace", {}).get("decision", {})
        if not decision:
            return None
        return Div(
            H3("Evidence decisions"),
            P(decision.get("status", "unknown").replace("_", " ")),
            *[
                Article(
                    Strong(
                        f"{step['number']}. "
                        + step.get("decision", {}).get("action", "failed request")
                    ),
                    P(step.get("decision", {}).get("reason", "")),
                    P(step["error"], role="alert") if step.get("error") else None,
                    Details(
                        Summary("Evidence returned"),
                        Pre(
                            json.dumps(
                                step.get("observation", {}),
                                indent=2,
                                ensure_ascii=False,
                            )
                        ),
                    ),
                )
                for step in decision.get("steps", [])
            ],
        )

    def run_cards(session_id, session):
        return [
            Details(
                Summary(f"{run['status'].title()} · {run['created'][:19]}"),
                P(run["result"].get("error", ""), role="alert")
                if run["result"].get("error")
                else None,
                P(
                    "Context changed during generation; this output was retained for review and was not offered as a current suggestion."
                )
                if run["status"] == "stale"
                else None,
                P(
                    "Private guidance was retained for its original moment while newer conversation arrived. Its wording was not added to the record."
                )
                if run["status"] == "superseded"
                else None,
                decision_card(run),
                Details(
                    Summary("Response editing pass"),
                    P(
                        run["result"]
                        .get("trace", {})
                        .get("response_review", {})
                        .get("status", "not run")
                        .replace("_", " ")
                    ),
                    P(
                        "This is a model editing pass, not an independent quality judgment."
                    ),
                    *[
                        Div(
                            Strong(issue["kind"].replace("_", " ")),
                            P(issue["quote"]),
                            P(issue["reason"]),
                        )
                        for issue in run["result"]
                        .get("trace", {})
                        .get("response_review", {})
                        .get("review", {})
                        .get("issues", [])
                    ],
                ),
                Details(
                    Summary("Context sent to model"),
                    Pre(json.dumps(run["request"], ensure_ascii=False, indent=2)),
                ),
                Details(
                    Summary("Model output and timings"),
                    Pre(json.dumps(run["result"], ensure_ascii=False, indent=2)),
                ),
                A("Download trace JSON", href=f"/play/{session_id}/trace/{run['id']}"),
            )
            for run in campaigns.runs(session_id)[:20]
        ]

    def panel_version(session_id):
        return digest(
            packed(
                {
                    "proposals": [
                        (p["id"], p["status"], p["stale"])
                        for p in campaigns.proposals(session_id)
                    ],
                    "runs": [
                        (r["id"], r["status"]) for r in campaigns.runs(session_id)
                    ],
                }
            )
        )

    def historical_drafts(session_id, runs):
        snapshot = campaigns.snapshot(session_id)
        entries = [
            run
            for run in runs
            if run["status"] == "superseded"
            and run["result"].get("historical_draft", {}).get("items")
            and not all(
                item.get("proposal_id")
                for item in run["result"]["historical_draft"]["items"]
            )
            and run["request"].get("snapshot_guard")
            and campaigns.snapshot_change(run["request"]["snapshot_guard"], snapshot)
            in {"unchanged", "append_only"}
        ][:3]
        if not entries:
            return None
        cards = []
        for run in entries:
            draft = run["result"]["historical_draft"]
            source = draft.get("as_of_source")
            label = (
                f"As of source #{source['ordinal']}"
                if source
                else "As of the session setup"
            )
            for item in draft["items"]:
                cards.append(
                    Article(
                        H3(item["title"]),
                        P(
                            Strong(label),
                            " · private history · newer conversation has arrived",
                        ),
                        A(
                            "Inspect source context",
                            href=f"/play/{session_id}/trace/{run['id']}",
                        ),
                        Pre(item["text"], cls="draft-text"),
                        Button(
                            "Copy wording",
                            type="button",
                            onclick="navigator.clipboard.writeText(this.parentElement.querySelector('.draft-text').textContent)",
                            cls="secondary",
                        ),
                        Small(
                            "This historical draft cannot apply state changes or publish itself."
                        ),
                        cls="card historical-draft",
                    )
                )
        return Div(H2("Drafts while conversation continues"), *cards)

    def suggestion_panel(session_id, session):
        runs = campaigns.runs(session_id)
        active = scheduler.is_pending(session_id) or any(
            r["status"] == "running" for r in runs
        )
        backlog = (
            runs[0]["result"]
            .get("trace", {})
            .get("classification", {})
            .get("backlog", [])
            if runs
            else []
        )
        proposals = campaigns.proposals(session_id)
        guidance = [
            p for p in proposals if p["kind"] != "state" and not p.get("observed")
        ]
        snapshot = campaigns.snapshot(session_id)
        outdated = {
            r["id"]
            for r in runs
            if r["request"].get("snapshot_guard")
            and campaigns.snapshot_change(r["request"]["snapshot_guard"], snapshot)
            == "invalidated"
        }
        latest = next(
            (
                p["run_id"]
                for p in reversed(guidance)
                if p["status"] != "rejected" and p["run_id"] not in outdated
            ),
            None,
        )
        priority = {
            "narration": 0,
            "scene": 1,
            "npc": 2,
            "question": 3,
            "action": 4,
            "rule": 5,
            "note": 6,
        }
        current_guidance = sorted(
            [
                p
                for p in guidance
                if p["run_id"] == latest
                and p["status"] != "rejected"
                and p["run_id"] not in outdated
            ],
            key=lambda p: priority.get(p["kind"], 7),
        )
        return Div(
            H2("Facilitator suggestions"),
            P(
                "The response editing pass could not complete; check this draft against the current conversation. Details are in the trace.",
                cls="notice",
            )
            if runs
            and runs[0]["result"]
            .get("trace", {})
            .get("response_review", {})
            .get("status")
            == "not_reviewed"
            else None,
            P(
                "A suggested check was withheld because its rule citation could not be verified. See the generation trace.",
                cls="notice",
            )
            if runs
            and runs[0]["result"]
            .get("trace", {})
            .get("storyteller", {})
            .get("rejected_checks")
            else None,
            historical_drafts(session_id, runs),
            P(
                "Rules advice could not be verified for this turn. Check the applicable rule and any required roll before resolving the action. The generation trace records the issue.",
                cls="notice",
            )
            if runs
            and runs[0]["result"].get("trace", {}).get("rules", {}).get("status")
            == "failed"
            else None,
            P(
                f"Analyzing {len(backlog)} earlier conversation messages…"
                if backlog
                else "Thinking from the latest table context…",
                role="status",
            )
            if active
            else None,
            P(
                f"{len(backlog)} conversation messages remain to analyze. Inspect the latest trace and retry the request.",
                cls="notice",
            )
            if backlog and not active
            else None,
            P(scheduler.errors.get(session_id, ""), role="alert")
            if session_id in scheduler.errors
            else None,
            P(
                "Model generation is not configured. Conversation, state, and manual suggestions are ready.",
                cls="notice",
            )
            if generate is None
            else None,
            P("No suggestions yet. Add the current situation or request a suggestion.")
            if not current_guidance
            else None,
            *[proposal_card(p, session_id, session) for p in current_guidance],
            Details(
                Summary("Earlier private suggestions"),
                *[
                    proposal_card(p, session_id, session)
                    for p in reversed(guidance)
                    if p["id"] not in {g["id"] for g in current_guidance}
                ],
            ),
            Details(
                Summary("Observed from the conversation"),
                *[
                    proposal_card(p, session_id, session)
                    for p in reversed(proposals)
                    if p.get("observed")
                ],
            ),
            Details(Summary("Generation traces"), *run_cards(session_id, session)),
            id="facilitator-suggestions",
            hx_get=f"/play/{session_id}/suggestions?version={panel_version(session_id)}",
            hx_trigger="every 5s",
            hx_swap="outerHTML",
        )

    @rt("/play/{session_id}")
    def play(session_id: str, session):
        try:
            snapshot = campaigns.snapshot(session_id)
            s, c = snapshot["session"], snapshot["campaign"]
            from .demo import DISCLOSURE, demo_info

            authored_demo = demo_info(campaigns, c["id"])
            return page(
                s["title"],
                A("← " + c["title"], href=f"/campaigns/{c['id']}"),
                H1(s["title"]),
                P(DISCLOSURE, cls="notice") if authored_demo else None,
                Div(
                    post_form(
                        session,
                        f"/play/{session_id}/request",
                        Button("Suggest now", disabled=generate is None),
                    ),
                    post_form(
                        session,
                        f"/play/{session_id}/proactive",
                        Hidden("0" if s["proactive"] else "1", name="enabled"),
                        Button(
                            "Pause automatic suggestions"
                            if s["proactive"]
                            else "Enable automatic suggestions",
                            cls="secondary",
                        ),
                    ),
                    A(
                        "Table record",
                        href=f"/play/{session_id}/public",
                        cls="button secondary",
                    ),
                    cls="row",
                ),
                P(
                    "Suggestions are private and reflect the conversation at that moment. What you and the players actually say updates the game record.",
                    cls="muted",
                ),
                Div(
                    Div(
                        suggestion_panel(session_id, session),
                        app.state.player_panel(session_id, session)
                        if hasattr(app.state, "player_panel")
                        else None,
                    ),
                    Div(
                        Article(
                            H2("At the table"),
                            post_form(
                                session,
                                f"/play/{session_id}/message",
                                Label("Speaker", fr="speaker"),
                                Input(
                                    name="speaker",
                                    id="speaker",
                                    list="known-speakers",
                                    required=True,
                                    placeholder="Person or captured speaker label",
                                ),
                                Label("Role"),
                                Select(
                                    *options(
                                        [
                                            (
                                                "unknown",
                                                "Use speaker mapping / unknown",
                                            ),
                                            ("player", "Player"),
                                            ("facilitator", "Facilitator"),
                                        ],
                                        "unknown",
                                    ),
                                    name="role",
                                ),
                                Label("Who hears this contribution?"),
                                audience_select(s["campaign_id"]),
                                Small(
                                    "Use a character-only message for a whisper or private clue."
                                ),
                                Label("What was said or done?", fr="message"),
                                Textarea(
                                    name="text", id="message", rows=4, required=True
                                ),
                                P(Button("Add conversation")),
                            ),
                            cls="card",
                        ),
                        working_state_panel(session_id, snapshot["state"]),
                        Article(
                            H2("Audio"),
                            P(
                                "Capture is opt-in. Microphone and system audio remain distinct sources; diarization does not establish a character identity."
                            ),
                            P(
                                A(
                                    "Audio status and controls",
                                    href=f"/play/{session_id}/audio",
                                )
                            ),
                            audio_follow_panel(session_id, session),
                            cls="card",
                        ),
                        Details(
                            Summary("Continue in the next session"),
                            P(
                                "Carry this session’s observed story forward, including conversation, resources, and character knowledge."
                            ),
                            post_form(
                                session,
                                f"/play/{session_id}/continue",
                                Label("Next session title"),
                                Input(name="title", required=True),
                                P(Button("Continue this story")),
                            ),
                        ),
                        Details(
                            Summary("Branch this session"),
                            P(
                                "Conversation and decisions are copied independently. Campaign reference material and base character sheets remain shared."
                            ),
                            post_form(
                                session,
                                f"/play/{session_id}/branch",
                                Label("Alternative session title"),
                                Input(name="title", required=True),
                                P(Button("Create independent branch")),
                            ),
                        ),
                    ),
                    cls="grid",
                ),
                conversation_panel(session_id, session),
            )
        except ValueError as exc:
            return error(exc)

    @rt("/play/{session_id}/suggestions")
    def suggestions(session_id: str, session, version: str = ""):
        try:
            campaigns.session(session_id)
            if version == panel_version(session_id):
                return Response(status_code=204)
            return suggestion_panel(session_id, session)
        except ValueError as exc:
            return error(exc)

    def message_card(message, session):
        return Article(
            Div(
                Strong(
                    f"{message.get('order_index', message['ordinal'])} · "
                    + (
                        message.get("source", {}).get("player_name", "Player") + " (AI)"
                        if message.get("source", {}).get("kind") == "player_agent"
                        else message["speaker"]
                    )
                ),
                Small(f"Source #{message['ordinal']}"),
                Span(message["role"], cls="badge"),
                Small(message["character"]),
                Small(
                    "Facilitator only"
                    if message["visibility"] == "private"
                    else "Player visible"
                ),
                cls="row",
            ),
            P(message["text"], style="white-space:pre-wrap"),
            Small("Overlaps earlier speech")
            if message.get("chronology", {}).get("overlaps_prior")
            else None,
            Details(
                Summary("Source"),
                Pre(
                    json.dumps(
                        {
                            "origin": message["source"],
                            "timing": message.get("chronology", {}),
                        },
                        indent=2,
                    )
                ),
            )
            if message["source"] or message.get("chronology")
            else None,
            post_form(
                session,
                f"/play/{message['session_id']}/message/{message['id']}/audio-controls",
                Button("Open source audio", cls="secondary"),
                hx_post=f"/play/{message['session_id']}/message/{message['id']}/audio-controls",
                hx_target="#audio-" + message["id"],
                hx_swap="innerHTML",
            )
            if message.get("source", {}).get("kind") == "live_audio"
            else None,
            Div(id="audio-" + message["id"]),
            A(
                "Review or correct",
                href=f"/play/{message['session_id']}/message/{message['id']}",
            ),
            cls="card",
            id="message-" + message["id"],
        )

    def conversation_panel(session_id, session):
        conversation = campaigns.conversation(session_id)
        messages = conversation["messages"]
        version = digest(
            packed([(m["id"], m["revision"], m["chronology"]) for m in messages])
        )
        return Div(
            H2("Conversation"),
            Details(
                Summary("Conversation timing"),
                *[P(note) for note in conversation["chronology"]["notes"]],
            ),
            P(
                "Some clock domains cannot be compared. Display positions between them are not known event order.",
                cls="notice",
            )
            if conversation["chronology"]["partial_order"]
            else None,
            P(
                "Showing the most recent 150 messages. Evidence links open the exact source message."
            )
            if len(messages) > 150
            else None,
            *[message_card(m, session) for m in messages[-150:]],
            id="table-conversation",
            hx_get=f"/play/{session_id}/conversation?version={version}",
            hx_trigger="every 5s",
            hx_swap="outerHTML",
        )

    @rt("/play/{session_id}/conversation")
    def conversation(session_id: str, session, version: str = ""):
        campaigns.session(session_id)
        messages = campaigns.conversation(session_id)["messages"]
        current = digest(
            packed([(m["id"], m["revision"], m["chronology"]) for m in messages])
        )
        return (
            Response(status_code=204)
            if version == current
            else conversation_panel(session_id, session)
        )

    def recording(session_id, message_id):
        from .live_audio import AudioQueue

        queue = AudioQueue(store.home / "live-audio")
        return campaign_audio_clip(campaigns, queue, session_id, message_id)

    @rt("/play/{session_id}/message/{message_id}/audio-controls", methods=["POST"])
    def audio_controls(
        request: Request,
        session,
        session_id: str,
        message_id: str,
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            clip = recording(session_id, message_id)
            ticket = secrets.token_urlsafe(24)
            access = dict(session.get("audio_access", {}))
            access[session_id + ":" + message_id] = ticket
            session["audio_access"] = dict(list(access.items())[-8:])
            url = f"/play/{session_id}/message/{message_id}/audio?" + urlencode(
                {"ticket": ticket}
            )
            fragment = "#t=" + str(round(clip["seek_start"], 3))
            if clip["seek_end"] is not None:
                fragment += "," + str(round(clip["seek_end"], 3))
            return Div(
                Audio(src=url + fragment, controls=True, preload="none"),
                Small(
                    f"Recorded {clip['channel']} chunk. "
                    + (
                        f"This message starts {clip['seek_start']:.2f} seconds into it. "
                        if clip["timed"]
                        else "Message timing is unavailable; the full chunk is retained. "
                    )
                    + "Press Play to listen."
                ),
            )
        except FileNotFoundError:
            return P(
                "The original audio was pruned; its transcript and source hash remain.",
                role="status",
            )
        except ValueError as exc:
            return P(str(exc), role="alert")

    @rt("/play/{session_id}/message/{message_id}/audio")
    def source_audio(
        request: Request, session, session_id: str, message_id: str, ticket: str = ""
    ):
        try:
            expected = session.get("audio_access", {}).get(
                session_id + ":" + message_id, ""
            )
            if not expected or not secrets.compare_digest(expected, ticket):
                return Response(
                    "Open source audio from the private session first.", status_code=403
                )
            validate(request, session, session.get("csrf", ""))
            clip = recording(session_id, message_id)
            return wav_response(clip["wav_bytes"], request.headers.get("range"))
        except FileNotFoundError:
            return Response("Original audio is no longer retained.", status_code=410)
        except ValueError:
            return Response(
                "Source audio is unavailable for this session.", status_code=404
            )

    @rt("/play/{session_id}/message/{message_id}")
    def source_message(session_id: str, message_id: str, session):
        messages = campaigns.conversation(session_id)["messages"]
        index = next((i for i, m in enumerate(messages) if m["id"] == message_id), None)
        if index is None:
            return error(ValueError("Source message not found."), f"/play/{session_id}")
        return page(
            "Conversation source",
            A("← Session", href=f"/play/{session_id}"),
            H1("Source and surrounding conversation"),
            *[
                message_card(m, session)
                for m in messages[max(0, index - 2) : index + 3]
            ],
            Article(
                H2("Correct this message"),
                post_form(
                    session,
                    f"/play/{session_id}/message/{message_id}/revise",
                    Hidden(messages[index]["revision"], name="revision"),
                    Label("Speaker"),
                    Input(
                        name="speaker", value=messages[index]["speaker"], required=True
                    ),
                    Label("Role"),
                    Select(
                        *options(
                            [
                                ("unknown", "Unknown"),
                                ("player", "Player"),
                                ("facilitator", "Facilitator"),
                            ],
                            messages[index]["role"],
                        ),
                        name="role",
                    ),
                    Label("Character"),
                    Input(name="character", value=messages[index]["character"]),
                    Label("Transcript"),
                    Textarea(
                        messages[index]["text"], name="text", rows=5, required=True
                    ),
                    Label("Who hears this contribution?"),
                    audience_select(
                        campaigns.session(session_id)["campaign_id"],
                        messages[index].get("recipient")
                        or (
                            "table"
                            if messages[index]["visibility"] == "public"
                            else "facilitator"
                        ),
                    ),
                    Label("Correction note"),
                    Input(name="note"),
                    P(Button("Save correction")),
                ),
                Details(
                    Summary("Original captured or entered text"),
                    Pre(messages[index]["original"]),
                ),
                cls="card",
            ),
        )

    @rt("/play/{session_id}/message/{message_id}/revise", methods=["POST"])
    def revise_message(
        request: Request,
        session,
        session_id: str,
        message_id: str,
        revision: str,
        speaker: str,
        text: str,
        role: str = "unknown",
        character: str = "",
        visibility: str = "private",
        note: str = "",
        recipient: str = "",
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.revise_message(
                session_id,
                message_id,
                text=text,
                speaker=speaker,
                role=role,
                character=character,
                visibility=visibility,
                expected_revision=revision,
                note=note,
                recipient=recipient or None,
            )
            schedule(session_id)
            return redirect(f"/play/{session_id}/message/{message_id}")
        except ValueError as exc:
            return error(exc, f"/play/{session_id}/message/{message_id}")

    def audio_follow_panel(session_id, session):
        settings = session.get("audio_follow", {}).get(session_id, {})
        if not settings.get("enabled"):
            return P("Automatic audio import is off.", cls="muted")
        return post_form(
            session,
            f"/play/{session_id}/audio/poll",
            Small(
                "Following completed audio chunks ("
                + settings.get("visibility", "private")
                + ").",
                id="audio-follow-status",
            ),
            hx_post=f"/play/{session_id}/audio/poll",
            hx_trigger="every 5s",
            hx_target="#audio-follow-status",
            hx_swap="innerHTML",
        )

    @rt("/play/{session_id}/public")
    def public_view(session_id: str):
        try:
            snapshot = campaigns.snapshot(session_id, public_only=True)
            return page(
                snapshot["session"]["title"],
                H1(snapshot["session"]["title"]),
                P("Player-visible material"),
                *[
                    Article(
                        H2(d["title"]),
                        P(d["text"], style="white-space:pre-wrap"),
                        cls="card",
                    )
                    for d in snapshot["documents"]
                ],
                H2("Characters"),
                *[
                    Article(
                        H3(c["name"]), Pre(json.dumps(c["sheet"], indent=2)), cls="card"
                    )
                    for c in snapshot["characters"]
                ],
                H2("Table conversation"),
                P(
                    "Clock domains differ; the displayed cross-domain order is uncertain."
                )
                if snapshot["chronology"]["partial_order"]
                else None,
                *[
                    P(Strong(m["speaker"] + ": "), m["text"])
                    for m in snapshot["messages"]
                ],
                H2("Established public state"),
                Pre(json.dumps(snapshot["state"], indent=2)),
            )
        except ValueError as exc:
            return error(exc)

    @rt("/play/{session_id}/message", methods=["POST"])
    def message(
        request: Request,
        session,
        session_id: str,
        speaker: str,
        text: str,
        role: str = "unknown",
        visibility: str = "public",
        recipient: str = "",
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.add_message(
                session_id,
                speaker,
                text,
                role=role,
                visibility=visibility,
                recipient=recipient or None,
                source={"kind": "manual", "review": "user supplied"},
            )
            schedule(session_id)
            return redirect(f"/play/{session_id}")
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @rt("/play/{session_id}/request", methods=["POST"])
    def request_suggestion(
        request: Request, session, session_id: str, csrf_token: str = ""
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.session(session_id)
            if generate is None:
                raise ValueError(
                    "Connect the local model before requesting suggestions."
                )
            schedule(session_id, force=True)
            return redirect(f"/play/{session_id}")
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @rt("/play/{session_id}/proactive", methods=["POST"])
    def proactive(
        request: Request, session, session_id: str, enabled: int, csrf_token: str = ""
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.set_proactive(session_id, bool(enabled))
            if enabled:
                schedule(session_id)
            return redirect(f"/play/{session_id}")
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @rt("/play/{session_id}/proposal/{proposal_id}", methods=["POST"])
    def decision(
        request: Request,
        session,
        session_id: str,
        proposal_id: str,
        action: str,
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            proposal = next(
                (p for p in campaigns.proposals(session_id) if p["id"] == proposal_id),
                None,
            )
            if (
                action != "rejected"
                or proposal is None
                or proposal.get("observed")
                or proposal["kind"] == "state"
            ):
                raise ValueError(
                    "Suggestions are private guidance. Correct the conversation to change the game record."
                )
            campaigns.decide(session_id, proposal_id, "rejected")
            schedule(session_id, force=True)
            return redirect(f"/play/{session_id}")
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @rt("/play/{session_id}/proposal/{proposal_id}/publish", methods=["POST"])
    def publish(
        request: Request,
        session,
        session_id: str,
        proposal_id: str,
        text: str,
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.publish(session_id, proposal_id, text)
            return redirect(f"/play/{session_id}")
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @rt("/play/{session_id}/publication/{publication_id}/withdraw", methods=["POST"])
    def withdraw(
        request: Request,
        session,
        session_id: str,
        publication_id: str,
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.withdraw(session_id, publication_id)
            return redirect(f"/play/{session_id}")
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @rt("/play/{session_id}/undo", methods=["POST"])
    def undo(request: Request, session, session_id: str, csrf_token: str = ""):
        try:
            validate(request, session, csrf_token)
            campaigns.undo(session_id)
            schedule(session_id)
            return redirect(f"/play/{session_id}")
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @rt("/play/{session_id}/continue", methods=["POST"])
    def continue_story(
        request: Request, session, session_id: str, title: str, csrf_token: str = ""
    ):
        try:
            validate(request, session, csrf_token)
            original = campaigns.session(session_id)
            new = campaigns.continue_session(
                original["campaign_id"], title, parent_id=session_id
            )
            return redirect(f"/play/{new}")
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @rt("/play/{session_id}/branch", methods=["POST"])
    def branch(
        request: Request, session, session_id: str, title: str, csrf_token: str = ""
    ):
        try:
            validate(request, session, csrf_token)
            new = campaigns.branch(session_id, title)
            return redirect(f"/play/{new}")
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @rt("/play/{session_id}/manual", methods=["POST"])
    def manual(
        request: Request,
        session,
        session_id: str,
        kind: str,
        title: str,
        text: str,
        payload: str = "{}",
        visibility: str = "private",
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.add_proposal(
                session_id,
                {
                    "kind": kind,
                    "title": title,
                    "text": text,
                    "payload": json.loads(payload),
                    "visibility": visibility,
                },
            )
            return redirect(f"/play/{session_id}")
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @rt("/play/{session_id}/trace/{run_id}")
    def trace(session_id: str, run_id: str):
        found = next((r for r in campaigns.runs(session_id) if r["id"] == run_id), None)
        return JSONResponse(
            found if found else {"error": "Trace not found"},
            status_code=200 if found else 404,
        )

    @rt("/play/{session_id}/trace/{run_id}/recover", methods=["POST"])
    def recover(
        request: Request, session, session_id: str, run_id: str, csrf_token: str = ""
    ):
        try:
            validate(request, session, csrf_token)
            if scheduler.is_pending(session_id):
                raise ValueError(
                    "This app is still processing the request. Wait for it to finish."
                )
            campaigns.recover_run(session_id, run_id)
            return redirect(f"/play/{session_id}")
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @rt("/play/{session_id}/audio")
    def audio(session_id: str, session):
        try:
            campaigns.session(session_id)
            from .live_audio import AudioQueue

            queue = AudioQueue(store.home / "live-audio")
            # Status API is intentionally read-only; opening a page never starts capture.
            status = queue.status(session=session_id)
            remote = os.environ.get("STORY_AUDIO_SSH")
            queue_path = (
                "~/.local/share/story-copilot/live-audio"
                if remote
                else str(store.home / "live-audio")
            )
            args = [
                "python",
                "-m",
                "story_copilot.mac_audio",
                "--queue",
                queue_path,
                "--session",
                session_id,
                "--mic",
                "--system",
            ]
            if remote:
                args += [
                    "--ssh",
                    remote,
                    "--remote-queue",
                    str(store.home / "live-audio"),
                ]
                for env_name, flag in [
                    ("STORY_AUDIO_SSH_CONTROL", "--ssh-control"),
                    ("STORY_AUDIO_REMOTE_PYTHON", "--remote-python"),
                ]:
                    if os.environ.get(env_name):
                        args += [flag, os.environ[env_name]]
            command = shlex.join(args)
            return page(
                "Session audio",
                A("← Session", href=f"/play/{session_id}"),
                H1("Live audio"),
                P(
                    "Start capture explicitly on the Mac. The microphone and system-audio tap are recorded separately. Stop capture with Ctrl+C."
                ),
                Pre(command),
                P(
                    "For microphone only, omit --system. For system audio only, omit --mic."
                ),
                P(
                    "The audio worker transcribes and diarizes queued chunks. Captured audio and traces remain in your private data directory."
                ),
                Pre(json.dumps(status, indent=2)),
                post_form(
                    session,
                    f"/play/{session_id}/audio/follow",
                    Label("New captured conversation is"),
                    visibility_select(),
                    Button("Follow completed chunks", name="enabled", value="1"),
                    Button(
                        "Stop following", name="enabled", value="0", cls="secondary"
                    ),
                ),
                audio_follow_panel(session_id, session),
                post_form(
                    session,
                    f"/play/{session_id}/audio/ingest",
                    Button("Import completed chunks once (Facilitator only)"),
                ),
                *[
                    post_form(
                        session,
                        f"/play/{session_id}/audio/retry",
                        Hidden(item["id"], name="chunk_id"),
                        P(
                            f"{item['channel']} chunk {item['sequence']}: {item['error']}"
                        ),
                        Button("Retry failed chunk", cls="secondary"),
                    )
                    for item in status["errors"]
                ],
            )
        except ImportError:
            return page(
                "Session audio",
                H1("Live audio"),
                P("The local audio capture module is not installed."),
                A("Back to session", href=f"/play/{session_id}"),
            )
        except ValueError as exc:
            return error(exc, f"/play/{session_id}")

    @rt("/play/{session_id}/audio/follow", methods=["POST"])
    def follow_audio(
        request: Request,
        session,
        session_id: str,
        enabled: int,
        visibility: str = "private",
        csrf_token: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.session(session_id)
            if visibility not in {"private", "public"}:
                raise ValueError("Choose Facilitator-only or player-visible capture.")
            settings = dict(session.get("audio_follow", {}))
            settings[session_id] = {"enabled": bool(enabled), "visibility": visibility}
            session["audio_follow"] = settings
            return redirect(f"/play/{session_id}/audio")
        except ValueError as exc:
            return error(exc, f"/play/{session_id}/audio")

    @rt("/play/{session_id}/audio/poll", methods=["POST"])
    def poll_audio(request: Request, session, session_id: str, csrf_token: str = ""):
        try:
            validate(request, session, csrf_token)
            campaigns.session(session_id)
            settings = session.get("audio_follow", {}).get(session_id, {})
            if not settings.get("enabled"):
                return "Automatic audio import is off."
            from .live_audio import AudioQueue, deliver_to_campaign

            queue = AudioQueue(store.home / "live-audio")
            count = deliver_to_campaign(
                queue,
                campaigns,
                session_id,
                session_id,
                visibility=settings.get("visibility", "private"),
            )
            if count:
                schedule(session_id)
            status = queue.status(session=session_id)
            return Small(
                f"Following audio · {count} new chunks imported · {status['counts']}"
            )
        except (ValueError, ImportError) as exc:
            return Small(str(exc), role="alert")

    @rt("/play/{session_id}/audio/retry", methods=["POST"])
    def retry_audio(
        request: Request, session, session_id: str, chunk_id: str, csrf_token: str = ""
    ):
        try:
            validate(request, session, csrf_token)
            campaigns.session(session_id)
            from .live_audio import AudioQueue

            queue = AudioQueue(store.home / "live-audio")
            if not any(
                item["id"] == chunk_id
                for item in queue.status(session=session_id)["errors"]
            ):
                raise ValueError("Select a failed chunk from this session.")
            queue.retry(chunk_id)
            return redirect(f"/play/{session_id}/audio")
        except (ValueError, ImportError) as exc:
            return error(exc, f"/play/{session_id}/audio")

    @rt("/play/{session_id}/audio/ingest", methods=["POST"])
    def ingest_audio(request: Request, session, session_id: str, csrf_token: str = ""):
        try:
            validate(request, session, csrf_token)
            campaigns.session(session_id)
            from .live_audio import AudioQueue, deliver_to_campaign

            queue = AudioQueue(store.home / "live-audio")
            deliver_to_campaign(queue, campaigns, session_id, session_id)
            schedule(session_id)
            return redirect(f"/play/{session_id}")
        except (ValueError, ImportError) as exc:
            return error(exc, f"/play/{session_id}")

    return campaigns
