from __future__ import annotations

import json
import html
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit, quote
from uuid import uuid4

from fasthtml.common import *
from starlette.requests import Request
from starlette.responses import RedirectResponse, FileResponse, PlainTextResponse

from .model import extract, draft, extraction_request, draft_request
from .schema import Event
from .state import replay
from .store import Store, digest
from .training import training_rows, export_dataset

CSS = """
:root{color-scheme:light;font-family:system-ui,-apple-system,sans-serif;color:#352a20;background:#f4eee3}
*{box-sizing:border-box}body{margin:0}a{color:#78502e}header{padding:18px 30px;border-bottom:1px solid #d9cbb8;background:#ede2d0}
header a{font-weight:750;text-decoration:none;font-size:1.25rem}header small{margin-left:20px}main{max-width:1500px;margin:auto;padding:24px}
h1{font:600 2rem Georgia,serif;margin:0 0 12px}h2{font:600 1.4rem Georgia,serif}h3{font-size:1rem;margin:0 0 12px}
.muted,small{color:#756654}.grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(320px,1fr);gap:24px}
.card{background:#fffbf4;border:1px solid #ddd0be;border-radius:10px;padding:18px;margin-bottom:16px}.turn{scroll-margin-top:20px}
.turn p{white-space:pre-wrap;line-height:1.65;margin:12px 0}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.spread{justify-content:space-between}
.badge{font-size:.75rem;padding:4px 8px;background:#e9dfd0;border-radius:30px}.accepted,.approved{background:#d9e7d5}.failed,.stale{background:#f3d6cd}
button,.button{font:inherit;background:#785538;color:white;border:0;border-radius:5px;padding:9px 14px;cursor:pointer;text-decoration:none;display:inline-block}
button.secondary,.button.secondary{background:#e5d8c5;color:#493728}button:disabled{opacity:.5;cursor:not-allowed}input,textarea,select{font:inherit;max-width:100%;border:1px solid #bcaa92;border-radius:4px;padding:8px;background:white;color:#352a20}
textarea{width:100%;line-height:1.5}label{display:block;font-size:.85rem;margin:12px 0 5px}form{margin:0}details{margin:12px 0}summary{cursor:pointer;color:#78502e}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem;line-height:1.5;max-height:520px;overflow:auto;background:#f0e8dc;padding:12px;border-radius:6px}
audio{width:100%;margin:12px 0}.notice{padding:12px;background:#fff0c9;border-radius:6px;margin:12px 0}.metrics{display:flex;gap:25px;flex-wrap:wrap;margin:15px 0 25px}.metric strong{display:block;font-size:1.5rem}
.story{display:block;padding:18px;background:#fffbf4;border:1px solid #ddd0be;border-radius:8px;text-decoration:none;margin-bottom:12px}
@media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:15px}header{padding:15px}header small{display:block;margin:6px 0}}
"""
JS = """document.addEventListener('click',function(e){let b=e.target.closest('[data-seek]');if(b){let a=document.getElementById('episode-audio');if(a){a.currentTime=Number(b.dataset.seek);a.play();}}});"""


def create_app(store=None):
    store = store or Store()
    keypath = store.home / ".session-secret"
    if not keypath.exists():
        keypath.write_text(secrets.token_hex(32))
        keypath.chmod(0o600)
    app = FastHTML(
        hdrs=(Style(CSS), Script(JS)),
        surreal=False,
        htmx=False,
        htmx4=True,
        secret_key=keypath.read_text(),
        session_cookie="story_copilot_session",
        canonical=False,
    )
    rt = app.route
    workers = ThreadPoolExecutor(max_workers=1, thread_name_prefix="facilitator-model")
    app.state.store = store
    experiments = Path(
        os.environ.get("STORY_EXPERIMENTS", store.home.parent / "experiments")
    ).resolve()

    def comparison_files():
        if not experiments.is_dir():
            return {}
        paths = (
            list(experiments.rglob("*.html"))
            + list(experiments.glob("tune-*/validation-loss.json"))
            + list(experiments.glob("live-audio-*/performance.json"))
        )
        return {
            digest(str(p.relative_to(experiments)))[:20]: p
            for p in paths
            if p.resolve().is_relative_to(experiments)
        }

    def token(session):
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(32)
        return session["csrf"]

    def csrf(session):
        return Hidden(token(session), name="csrf_token")

    def validate(request, session, submitted):
        if not secrets.compare_digest(
            session.get("csrf", ""), submitted or ""
        ) or not session.get("csrf"):
            raise ValueError("Reload this page before submitting the form.")
        origin = request.headers.get("origin")
        if origin and origin != "null":
            a, b = urlsplit(origin), urlsplit(str(request.url))
            if (a.scheme, a.netloc) != (b.scheme, b.netloc):
                raise ValueError("Use the local application to submit this form.")
        if origin == "null" and request.headers.get("sec-fetch-site") != "same-origin":
            raise ValueError("Use the local application to submit this form.")
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise ValueError("Use the local application to submit this form.")

    def page(title, *content):
        return (
            Title(title + " · Story Copilot"),
            Header(
                A("Story Copilot", href="/"),
                Small("Your story. Your choices. A second pair of eyes."),
                A(
                    "Campaigns",
                    href="/campaigns",
                    style="font-size:.9rem;margin-left:24px",
                ),
                A(
                    "Player agents",
                    href="/players",
                    style="font-size:.9rem;margin-left:24px",
                ),
                A(
                    "Model settings",
                    href="/settings",
                    style="font-size:.9rem;margin-left:24px",
                ),
                A(
                    "Source library",
                    href="/library",
                    style="font-size:.9rem;margin-left:24px",
                ),
                A(
                    "Model evaluations",
                    href="/experiments",
                    style="font-size:.9rem;margin-left:24px",
                ),
            ),
            Main(*content),
        )

    def failure(exc, cid=None):
        return page(
            "Review needed",
            H1("Couldn’t complete that"),
            P(str(exc)),
            A("Back to review", href=f"/c/{cid}" if cid else "/"),
        )

    def options(values, current):
        return [
            Option(v.replace("_", " ").title(), value=v, selected=v == current)
            for v in values
        ]

    def turn_card(t, cid, session):
        timecode = (
            None
            if t["start"] is None
            else f"{int(t['start'] // 60):02}:{int(t['start'] % 60):02}"
        )
        return Article(
            Div(
                Strong(f"{t['ordinal']} · {t['speaker']}"),
                Span(t["role"], cls="badge"),
                Span(t["status"], cls="badge " + t["status"]),
                Button(
                    "▶ " + timecode,
                    type="button",
                    data_seek=t["start"],
                    cls="secondary",
                )
                if timecode
                else Small("No source timing"),
                cls="row",
            ),
            P(t["text"]),
            Small("Reviewed by " + t["reviewer"])
            if t["status"] in {"approved", "screened"}
            else None,
            Details(
                Summary("Review speaker and text"),
                Form(
                    csrf(session),
                    Hidden(t["revision"], name="revision"),
                    Label("Speaker role", fr=f"role-{t['ordinal']}"),
                    Select(
                        *options(
                            ["unknown", "facilitator", "player", "mixed"], t["role"]
                        ),
                        name="role",
                        id=f"role-{t['ordinal']}",
                    ),
                    Label("Character, if known"),
                    Input(name="character", value=t["character"]),
                    Label("Corrected text"),
                    Textarea(t["text"], name="text", rows=5),
                    Label("Content"),
                    Select(
                        *options(
                            [
                                "gameplay",
                                "narration",
                                "dialogue",
                                "mechanics",
                                "production",
                                "chatter",
                            ],
                            t["category"],
                        ),
                        name="category",
                    ),
                    Label("Review"),
                    Select(
                        *options(
                            ["pending", "approved", "screened", "excluded"], t["status"]
                        ),
                        name="status",
                    ),
                    Label("Review note"),
                    Input(name="note", value=t["note"]),
                    P(Button("Save review")),
                    action=f"/c/{cid}/turn/{t['ordinal']}",
                    method="post",
                ),
                Details(Summary("Original source"), Pre(t["original"])),
            ),
            Details(
                Summary("Word timing and source metadata"),
                Pre(json.dumps(json.loads(t["metadata"]), indent=2)),
            ),
            cls="card turn",
            id=f"turn-{t['ordinal']}",
        )

    def event_card(e, cid, session):
        p = e["payload"]
        status = "stale" if e["stale"] else e["status"]
        return Article(
            Div(
                Strong(f"{p['kind']} · {p['entity']}"),
                Span(status, cls="badge " + status),
                cls="row",
            ),
            P(
                f"{p['attribute']}: {p['value'] if p['delta'] is None else 'delta ' + str(p['delta'])}"
            ),
            Small(f"{p['stage']} · {p['visibility']}"),
            Small(" · reviewed by " + e["reviewer"]) if e["reviewer"] else None,
            *[
                P(
                    A(
                        f"Turn {ref['turn']}",
                        href=f"/c/{cid}?start={max(1, ref['turn'] - 3)}#turn-{ref['turn']}",
                    ),
                    " — ",
                    ref["quote"],
                )
                for ref in p["evidence"]
            ],
            Details(Summary("Proposal details"), Pre(json.dumps(p, indent=2))),
            Details(
                Summary("Correct this proposal"),
                Form(
                    csrf(session),
                    Textarea(json.dumps(p, indent=2), name="payload", rows=12),
                    P(Button("Save corrected proposal")),
                    action=f"/c/{cid}/event/{e['id']}/correct",
                    method="post",
                ),
            ),
            Form(
                csrf(session),
                Button("Accept", name="status", value="accepted", disabled=e["stale"]),
                Button("Reject", name="status", value="rejected", cls="secondary"),
                Button("Reset", name="status", value="pending", cls="secondary"),
                action=f"/c/{cid}/event/{e['id']}",
                method="post",
                cls="row",
            ),
            cls="card",
        )

    def run_panel(cid):
        runs = store.runs(cid)[:10]
        cards = []
        for r in runs:
            result = json.loads(r["result"])
            request = json.loads(r["request"])
            answer = result.get("answer", {})
            cards.append(
                Div(
                    Div(
                        Strong(
                            {
                                "facilitator": "Facilitator draft",
                                "extract": "State extraction",
                                "speaker_roles": "Speaker-role proposal",
                                "target_screen": "Automatic response screen",
                            }.get(r["kind"], r["kind"].replace("_", " ").title())
                        ),
                        Span(r["status"], cls="badge " + r["status"]),
                        cls="row",
                    ),
                    P(result["error"], cls="notice") if result.get("error") else None,
                    P(answer.get("direct_answer"))
                    if answer.get("direct_answer")
                    else None,
                    P(answer["narration"]) if answer.get("narration") else None,
                    Ul(
                        *[
                            Li(x)
                            for x in answer.get("questions", [])
                            + [
                                check["text"] if isinstance(check, dict) else check
                                for check in answer.get("requested_checks", [])
                            ]
                        ]
                    )
                    if answer
                    else None,
                    Details(
                        Summary("Facilitator-private notes"),
                        P(answer.get("private_notes", "")),
                    )
                    if answer.get("private_notes")
                    else None,
                    P(
                        f"{len(result.get('event_ids', []))} event proposals · {result.get('seconds', '—')} seconds"
                    )
                    if r["kind"] == "extract" and r["status"] in {"complete", "partial"}
                    else None,
                    A(
                        "Review proposals",
                        href=f"/c/{cid}#events",
                        cls="button secondary",
                    )
                    if result.get("event_ids")
                    else None,
                    P("Draft only; this response has not changed the recorded game.")
                    if answer
                    else None,
                    Details(
                        Summary("Input, response and timing"),
                        Pre(
                            json.dumps({"request": request, "result": result}, indent=2)
                        ),
                    ),
                    cls="card",
                )
            )
        running = any(r["status"] == "running" for r in runs)
        return Div(
            *cards if cards else [P("No model runs yet.", cls="muted")],
            id="runs",
            hx_get=f"/c/{cid}/runs",
            hx_trigger="every 3s" if running else None,
            hx_swap="outerHTML",
        )

    def state_view(state):
        parts = []
        if state["entities"]:
            parts += [
                H3("Characters and objects"),
                Ul(
                    *[
                        Li(
                            Strong(entity),
                            " — ",
                            "; ".join(
                                f"{key.replace('_', ' ')}: {v['value']}"
                                for key, v in attrs.items()
                            ),
                        )
                        for entity, attrs in state["entities"].items()
                    ]
                ),
            ]
        if state["facts"]:
            parts += [
                H3("Established facts"),
                Ul(
                    *[
                        Li(
                            key.replace("_", " ").replace(":", " · "),
                            " — ",
                            str(v["value"]),
                        )
                        for key, v in state["facts"].items()
                    ]
                ),
            ]
        if state["resources"]:
            parts += [
                H3("Resources"),
                Ul(
                    *[
                        Li(
                            key,
                            " — ",
                            str(v["value"])
                            if v["value"] is not None
                            else f"total unknown; observed change {v['known_delta']:+}",
                        )
                        for key, v in state["resources"].items()
                    ]
                ),
            ]
        if state["pending"]:
            parts += [
                H3("Unresolved actions"),
                Ul(
                    *[
                        Li(v["entity"], " — ", str(v["value"]))
                        for v in state["pending"].values()
                    ]
                ),
            ]
        if state["claims"]:
            parts += [
                H3("Claims, not established facts"),
                Ul(*[Li(v["entity"], " — ", str(v["value"])) for v in state["claims"]]),
            ]
        if state["stale_events"]:
            parts += [
                P(
                    f"{len(state['stale_events'])} events excluded because their sources changed.",
                    cls="notice",
                )
            ]
        return Div(
            *parts
            if parts
            else [P("No accepted state changes at this point.", cls="muted")],
            Details(Summary("Complete state record"), Pre(json.dumps(state, indent=2))),
        )

    @rt("/")
    def home():
        return RedirectResponse("/campaigns", status_code=303)

    @rt("/library")
    def index(session):
        token(session)
        collections = store.collections()
        return page(
            "Library",
            H1("A game you can trace"),
            P(
                A(
                    "Compare diarization outputs",
                    href="/diarization",
                    cls="button secondary",
                )
            )
            if (store.home / "diarization-comparison.json").exists()
            else None,
            P(
                "Review what was said, reconstruct what changed, and teach a Facilitator to respond."
            ),
            *[
                A(
                    H3(c["title"]),
                    P(f"{c['turn_count']:,} turns · {c['split']}"),
                    href=f"/c/{c['id']}",
                    cls="story",
                )
                for c in collections
            ],
            P("Import a transcript with the story-copilot import command to begin.")
            if not collections
            else None,
        )

    @rt("/experiments")
    def experiments_index():
        cards = []
        for key, path in sorted(
            comparison_files().items(),
            key=lambda item: item[1].stat().st_mtime,
            reverse=True,
        ):
            metadata = path.with_suffix(".json") if path.suffix == ".html" else path
            if path.name == "index.html" and (path.parent / "result.json").exists():
                metadata = path.parent / "result.json"
            elif not metadata.exists() and path.name == "review.html":
                metadata = path.parent / "report.json"
            try:
                report = json.loads(metadata.read_text())
            except (OSError, ValueError):
                continue
            if "cases" in report:
                count = len(report["cases"])
                description = (
                    f"{count}/{report.get('expected_cases', count)} cases · "
                    + (
                        "paired response review"
                        if path.suffix == ".html"
                        else "completion loss comparison"
                    )
                )
            elif "steps" in report:
                description = (
                    f"{len(report['steps'])} workflow steps · real-model integration"
                )
            elif "runs" in report and "inventory" in report:
                description = f"{len(report['runs'])} serving configurations · context, latency and GPU memory"
            elif "phases" in report and "hardware" in report:
                description = (
                    "Audio processing and language-model coexistence · measured timings"
                )
            else:
                continue
            cards.append(
                A(
                    H3(path.parent.name + " / " + path.stem),
                    P(description),
                    Small(report.get("status", "recorded")),
                    href=f"/experiments/{key}",
                    cls="story",
                )
            )
        return page(
            "Model evaluations",
            H1("Model evaluations"),
            P(
                "Compare generated responses, inspect source evidence and review timing. Loss and automatic checks are separate from narrative quality."
            ),
            P(
                "New comparisons appear here as they run. Reload to see the latest results."
            ),
            *cards,
            P("No comparisons have been generated yet.") if not cards else None,
        )

    def report_assets(path):
        manifest = path.parent / "result.json"
        if path.name != "index.html" or not manifest.is_file():
            return {}
        report = json.loads(manifest.read_text())
        names = {"result.json", "review.json"}
        names.update(
            result["trace_file"]
            for case in report.get("cases", [])
            for result in case.get("policies", {}).values()
        )
        return {
            name: (path.parent / name).resolve()
            for name in names
            if (path.parent / name).resolve().is_relative_to(path.parent.resolve())
            and (path.parent / name).suffix == ".json"
            and (path.parent / name).is_file()
        }

    @rt("/experiments/{report_id}/trace/{asset_path:path}")
    def experiment_trace(report_id: str, asset_path: str):
        path = comparison_files().get(report_id)
        asset = report_assets(path).get(asset_path) if path else None
        if asset is None:
            return PlainTextResponse("Trace not found.", status_code=404)
        return FileResponse(
            asset, media_type="application/json", headers={"Cache-Control": "no-store"}
        )

    @rt("/experiments/{report_id}")
    def experiment_report(report_id: str):
        path = comparison_files().get(report_id)
        if path is None:
            return PlainTextResponse("Comparison not found.", status_code=404)
        if path.suffix == ".html":
            assets = report_assets(path)
            if assets:
                document = path.read_text()
                for name in assets:
                    document = document.replace(
                        'href="' + html.escape(name, quote=True) + '"',
                        'href="/experiments/'
                        + report_id
                        + "/trace/"
                        + quote(name)
                        + '"',
                    )
                return HTMLResponse(document, headers={"Cache-Control": "no-store"})
            return FileResponse(
                path, media_type="text/html", headers={"Cache-Control": "no-store"}
            )
        report = json.loads(path.read_text())
        if "aggregate" not in report:
            return page(
                "Performance evaluation",
                A("← Evaluations", href="/experiments"),
                H1(path.parent.name),
                P(
                    "Measured performance and source provenance. These timings do not establish transcription or diarization accuracy."
                ),
                Pre(json.dumps(report, indent=2)),
            )
        return page(
            "Completion loss comparison",
            A("← Evaluations", href="/experiments"),
            H1(path.parent.name),
            P(
                report.get(
                    "metric", "Completion loss is not a narrative-quality score."
                )
            ),
            Pre(json.dumps(report.get("aggregate", {}), indent=2)),
            Details(
                Summary("Strict adapter reload"),
                Pre(json.dumps(report.get("reload_validation", {}), indent=2)),
            ),
            Details(
                Summary("Per-case results and provenance"),
                Pre(json.dumps(report.get("cases", []), indent=2)),
            ),
        )

    @rt("/diarization")
    def diarization_comparison(start: int = 0):
        path = store.home / "diarization-comparison.json"
        if not path.exists():
            return PlainTextResponse(
                "No comparison has been generated.", status_code=404
            )
        report = json.loads(path.read_text())
        start = max(0, start)
        cid = report["right_collection_id"]
        turns = store.turns(cid)
        spans = report.get("review_spans", [])
        cards = []
        for span in spans[start : start + 30]:
            turn = next(
                (
                    t
                    for t in turns
                    if t["start"] is not None
                    and t["end"] is not None
                    and t["start"] <= span["start"] <= t["end"]
                ),
                None,
            )
            seconds = span["start"]
            label = f"{int(seconds // 60):02}:{int(seconds % 60):02}"
            cards.append(
                Article(
                    Div(
                        Button(
                            "▶ " + label,
                            type="button",
                            data_seek=seconds,
                            cls="secondary",
                        ),
                        Strong(
                            f"3.1: {span['left_speaker']} · Community-1: {span['right_speaker']}"
                        ),
                        cls="row",
                    ),
                    P(span["context"]),
                    A(
                        "Review this turn",
                        href=f"/c/{cid}?start={max(1, turn['ordinal'] - 2)}#turn-{turn['ordinal']}",
                    )
                    if turn
                    else None,
                    cls="card",
                )
            )
        return page(
            "Speaker comparison",
            A("← Library", href="/"),
            H1("Where speaker assignments differ"),
            P(
                f"{report['disagreed_words']:,} of {report['assigned_by_both']:,} words assigned by both pipelines differ after matching their speaker IDs."
            ),
            P(
                "Differences identify passages to listen to. They do not measure accuracy or establish which model is correct. Community-1 uses its exclusive word assignment; original overlapping speech is retained.",
                cls="notice",
            ),
            Audio(
                controls=True, src=f"/c/{cid}/audio", id="episode-audio", preload="none"
            ),
            Div(
                A(
                    "← Earlier",
                    href=f"/diarization?start={max(0, start - 30)}",
                    cls="button secondary",
                ),
                Strong(
                    f"Passages {start + 1}–{min(start + 30, len(spans))} of {len(spans)}"
                ),
                A(
                    "Later →",
                    href=f"/diarization?start={min(max(0, len(spans) - 1), start + 30)}",
                    cls="button secondary",
                ),
                cls="row spread",
            ),
            *cards,
            Details(
                Summary("Speaker-label matching and coverage"),
                Pre(
                    json.dumps(
                        {
                            k: v
                            for k, v in report.items()
                            if k
                            not in {
                                "disagreements",
                                "review_spans",
                                "left_provenance",
                                "right_provenance",
                            }
                        },
                        indent=2,
                    )
                ),
            ),
        )

    @rt("/c/{cid}")
    def collection(
        cid: str, session, start: int = 1, before: int = 0, message: str = ""
    ):
        try:
            c = store.collection(cid)
        except ValueError as exc:
            return failure(exc)
        turns = store.turns(cid)
        start = max(1, min(start, len(turns)))
        end = min(start + 19, len(turns))
        before = before or end + 1
        rows, skipped = training_rows(store, cid)
        screened_rows, _ = training_rows(store, cid, include_screened=True)
        events = store.events(cid)
        state = replay(store, cid, before=before)
        speakers = sorted({t["speaker"] for t in turns})
        return page(
            c["title"],
            A("← Library", href="/"),
            H1(c["title"]),
            P(message, cls="notice") if message else None,
            Div(
                Div(Strong(f"{len(turns):,}"), Small("source turns"), cls="metric"),
                Div(
                    Strong(
                        str(
                            sum(
                                e["status"] == "accepted" and not e["stale"]
                                for e in events
                            )
                        )
                    ),
                    Small("accepted events"),
                    cls="metric",
                ),
                Div(
                    Strong(str(len(rows))),
                    Small("eligible Facilitator examples"),
                    cls="metric",
                ),
                Div(
                    Strong(
                        str(
                            sum(
                                r["provenance"]["review_level"] == "model-screened"
                                for r in screened_rows
                            )
                        )
                    ),
                    Small("model-screened targets"),
                    cls="metric",
                ),
                cls="metrics",
            ),
            Audio(
                controls=True, src=f"/c/{cid}/audio", id="episode-audio", preload="none"
            )
            if c["audio_path"]
            else None,
            Div(
                Section(
                    Div(
                        A(
                            "← Earlier",
                            href=f"/c/{cid}?start={max(1, start - 20)}",
                            cls="button secondary",
                        ),
                        Strong(f"Turns {start}–{end} of {len(turns)}"),
                        A(
                            "Later →",
                            href=f"/c/{cid}?start={min(len(turns), start + 20)}",
                            cls="button secondary",
                        ),
                        cls="row spread",
                    ),
                    *[turn_card(t, cid, session) for t in turns[start - 1 : end]],
                ),
                Aside(
                    Div(
                        H2("Speaker mapping"),
                        P(
                            "Assign a role to unreviewed turns. Each turn still needs review before training.",
                            cls="muted",
                        ),
                        Form(
                            csrf(session),
                            Select(
                                *[Option(s, value=s) for s in speakers], name="speaker"
                            ),
                            Select(
                                *options(
                                    ["facilitator", "player", "unknown"], "facilitator"
                                ),
                                name="role",
                            ),
                            P(Button("Assign role")),
                            action=f"/c/{cid}/map",
                            method="post",
                        ),
                        cls="card",
                    ),
                    Div(
                        H2("Reconstruct this section"),
                        P("Model suggestions remain pending until reviewed."),
                        Form(
                            csrf(session),
                            Label("First turn"),
                            Input(
                                type="number",
                                name="start",
                                value=start,
                                min=1,
                                max=len(turns),
                            ),
                            Label("Last turn"),
                            Input(
                                type="number",
                                name="end",
                                value=min(start + 9, len(turns)),
                                min=1,
                                max=len(turns),
                            ),
                            P(Button("Propose state changes")),
                            action=f"/c/{cid}/extract",
                            method="post",
                        ),
                        cls="card",
                    ),
                    Div(
                        H2(f"State before turn {before}"),
                        Form(
                            Label("Before turn"),
                            Input(
                                type="number",
                                name="before",
                                value=before,
                                min=1,
                                max=len(turns) + 1,
                            ),
                            Hidden(start, name="start"),
                            Button("Replay"),
                            method="get",
                            action=f"/c/{cid}",
                        ),
                        state_view(state),
                        cls="card",
                    ),
                    Div(
                        H2("Draft the next Facilitator turn"),
                        P(
                            "Uses prior turns, accepted state, and retrieved rules. The recorded target is withheld."
                        ),
                        Form(
                            csrf(session),
                            Label("Before recorded turn"),
                            Input(
                                type="number",
                                name="before",
                                value=before,
                                min=1,
                                max=len(turns) + 1,
                            ),
                            Label("New player action (optional)"),
                            Textarea(name="player_input", rows=2),
                            Label("Private Facilitator direction (optional)"),
                            Textarea(name="direction", rows=2),
                            P(Button("Generate draft")),
                            action=f"/c/{cid}/draft",
                            method="post",
                        ),
                        cls="card",
                    ),
                    Div(
                        H2("Training export"),
                        P(f"{c['split']} split · {len(rows)} eligible examples"),
                        Details(
                            Summary("Exclusion reasons"),
                            Pre(json.dumps(skipped, indent=2)),
                        ),
                        Form(
                            csrf(session),
                            Button("Download reviewed examples", disabled=not rows),
                            method="post",
                            action=f"/c/{cid}/export",
                        ),
                        cls="card",
                    ),
                    H2("Model runs"),
                    run_panel(cid),
                    H2("Event review", id="events"),
                    *[
                        event_card(e, cid, session)
                        for e in events
                        if e["status"] != "rejected"
                    ],
                    Details(
                        Summary(
                            f"Rejected proposals ({sum(e['status'] == 'rejected' for e in events)})"
                        ),
                        *[
                            event_card(e, cid, session)
                            for e in events
                            if e["status"] == "rejected"
                        ],
                    ),
                    P("No event proposals yet.", cls="muted") if not events else None,
                ),
                cls="grid",
            ),
        )

    @rt("/c/{cid}/audio")
    def audio(cid: str):
        path = store.collection(cid)["audio_path"]
        if not path or not Path(path).is_file():
            return PlainTextResponse("Source audio unavailable.", status_code=404)
        return FileResponse(path)

    @rt("/c/{cid}/runs")
    def runs(cid: str):
        return run_panel(cid)

    @rt("/c/{cid}/turn/{ordinal}")
    def post(
        cid: str,
        ordinal: int,
        request: Request,
        session,
        csrf_token: str,
        revision: str,
        text: str,
        role: str,
        status: str,
        category: str,
        character: str = "",
        note: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            store.revise(
                f"{cid}:{ordinal}",
                text=text,
                role=role,
                character=character,
                status=status,
                category=category,
                note=note,
                expected_revision=revision,
            )
            return RedirectResponse(
                f"/c/{cid}?start={max(1, ordinal - 2)}#turn-{ordinal}", 303
            )
        except ValueError as exc:
            return failure(exc, cid)

    @rt("/c/{cid}/map")
    def post(
        cid: str, request: Request, session, csrf_token: str, speaker: str, role: str
    ):
        try:
            validate(request, session, csrf_token)
            for t in store.turns(cid):
                if (
                    t["speaker"] == speaker
                    and t["status"] == "pending"
                    and t["role"] == "unknown"
                ):
                    store.revise(
                        t["id"],
                        text=t["text"],
                        role=role,
                        character=t["character"],
                        category=t["category"],
                        note="Speaker mapping; turn content not yet approved.",
                        expected_revision=t["revision"],
                    )
            return RedirectResponse(f"/c/{cid}", 303)
        except ValueError as exc:
            return failure(exc, cid)

    @rt("/c/{cid}/event/{eid}")
    def post(
        cid: str, eid: str, request: Request, session, csrf_token: str, status: str
    ):
        try:
            validate(request, session, csrf_token)
            store.review_event(cid, eid, status)
            return RedirectResponse(f"/c/{cid}", 303)
        except ValueError as exc:
            return failure(exc, cid)

    @rt("/c/{cid}/extract")
    def post(
        cid: str, request: Request, session, csrf_token: str, start: int, end: int
    ):
        try:
            validate(request, session, csrf_token)
            req = extraction_request(store, cid, start, end)
            rid = store.start_run(cid, "extract", req)
            workers.submit(extract, store, cid, start, end, run_id=rid, request=req)
            return RedirectResponse(f"/c/{cid}?start={start}#runs", 303)
        except ValueError as exc:
            return failure(exc, cid)

    @rt("/c/{cid}/event/{eid}/correct")
    def post(
        cid: str, eid: str, request: Request, session, csrf_token: str, payload: str
    ):
        try:
            validate(request, session, csrf_token)
            prior = next((e for e in store.events(cid) if e["id"] == eid), None)
            if prior is None:
                raise ValueError("Event not found.")
            proposed = json.loads(payload)
            if prior["status"] == "accepted":
                proposed["supersedes"] = eid
            event = Event.model_validate(proposed)
            replacement = store.add_event(cid, event)
            if replacement != eid and prior["status"] == "pending":
                store.review_event(
                    cid,
                    eid,
                    "rejected",
                    note=f"Replaced by corrected proposal {replacement}.",
                )
            return RedirectResponse(f"/c/{cid}", 303)
        except ValueError as exc:
            return failure(exc, cid)

    @rt("/c/{cid}/draft")
    def post(
        cid: str,
        request: Request,
        session,
        csrf_token: str,
        before: int,
        player_input: str = "",
        direction: str = "",
    ):
        try:
            validate(request, session, csrf_token)
            req = draft_request(store, cid, before, player_input, direction)
            rid = store.start_run(cid, "facilitator", req)
            workers.submit(draft, store, cid, before, run_id=rid, request=req)
            return RedirectResponse(f"/c/{cid}?before={before}#runs", 303)
        except ValueError as exc:
            return failure(exc, cid)

    @rt("/c/{cid}/export")
    def post(cid: str, request: Request, session, csrf_token: str):
        try:
            validate(request, session, csrf_token)
            path = store.home / "exports" / f"{cid}-{uuid4().hex[:8]}.jsonl"
            export_dataset(store, cid, path)
            return FileResponse(
                path, filename=path.name, media_type="application/jsonl"
            )
        except ValueError as exc:
            return failure(exc, cid)

    from .settings_web import register

    register(app, store, page, csrf, validate)
    from .campaign_web import register_campaign_routes
    from .copilot import make_copilot

    build_play_context, generate_play = make_copilot(store)
    register_campaign_routes(
        app,
        store,
        page=page,
        csrf=csrf,
        validate=validate,
        workers=workers,
        generate=generate_play,
        build_context=build_play_context,
    )
    from .player_web import register as register_players

    register_players(
        app,
        store,
        page=page,
        csrf=csrf,
        validate=validate,
        workers=workers,
        schedule=app.state.campaign_scheduler.schedule,
    )
    return app
