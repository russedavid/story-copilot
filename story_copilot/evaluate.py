"""Live, original-scenario comparison. Private traces stay outside source control."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import time

from .campaigns import Campaigns
from .copilot import make_copilot
from .demo import ensure_demo
from .settings import load, save
from .store import Store, packed, digest, now

RUBRIC = {
    "current_intent": "Answers the current contributions, including changed choices, rather than advancing an abandoned action.",
    "agency": "Does not speak or choose for a player; NPC reactions belong to the facilitator.",
    "continuity": "Uses corrected source facts and current resources; does not reapply earlier costs on resume.",
    "knowledge": "Does not grant another character private knowledge merely because the facilitator can see it.",
    "rules": "Uses supplied rules and observed values, or asks about missing inputs; invents no dice or automatic outcomes.",
    "usability": "Gives a concise, playable response to the immediate scene.",
}

# Original authored test exchanges; no real-play transcript or reference answers.
STEPS = [
    {
        "id": "opening",
        "title": "Several contributions in one exchange",
        "messages": [],
        "expect": "Address Jo's listening/time question and Rowan's unanswered question to Mara. Do not make Bea answer for Mara or invent a required check.",
    },
    {
        "id": "changed-choice",
        "title": "A player changes their choice",
        "messages": [
            (
                "Facilitator",
                "facilitator",
                "The clicking comes from the maintenance hatch. Mara says she last saw the operator at dusk.",
            ),
            (
                "Ari",
                "player",
                "Actually, I leave the hatch and receiver alone. I want to talk to Mara instead. Does she need help finding the operator?",
            ),
        ],
        "expect": "Follow the changed choice, leave the hatch closed, and propose Mara's response without choosing Jo's next action.",
    },
    {
        "id": "private-knowledge",
        "title": "Knowledge belongs to one character",
        "messages": [
            (
                "Facilitator",
                "facilitator",
                "Only Rowan notices the torn page tucked inside Mara's coat. Jo is looking at the corridor and does not notice it.",
            ),
            (
                "Bea",
                "player",
                "I do not tell Jo about the page yet. I ask Jo whether we should help Mara search outside.",
            ),
        ],
        "expect": "Do not make Jo know about or act on the page; leave Jo free to answer Rowan's invitation.",
    },
    {
        "id": "supported-cost",
        "title": "A supplied rule and current resources",
        "messages": [
            (
                "Ari",
                "player",
                "Jo uses the portable relay tester once. I spend 1 battery charge; Jo now has 2 battery charges. Can I afford a second use under our supplied rule?",
            )
        ],
        "expect": "Recognize two remaining charges and one-charge cost; a second use is possible but has not happened.",
    },
    {
        "id": "corrected-cost",
        "title": "A correction replaces an earlier fact",
        "messages": [],
        "expect": "The corrected source says Jo did not use the tester and still has three charges. Do not retain the superseded expenditure.",
    },
    {
        "id": "unknown-rule",
        "title": "A rule the campaign does not supply",
        "messages": [
            (
                "Bea",
                "player",
                "What exact bonus do I get for helping Jo inspect the hatch? We have not chosen any teamwork rule.",
            )
        ],
        "expect": "Ask for or help the facilitator choose a teamwork rule. Do not invent a numerical bonus or borrow one from a remembered game.",
    },
    {
        "id": "resumed",
        "title": "Continue the next session",
        "messages": [
            (
                "Ari",
                "player",
                "Before we continue, how many battery charges does Jo have? I still have not used the tester. Does Jo know where the missing page went?",
            )
        ],
        "expect": "Resume with three charges and preserve that only Rowan noticed the page; do not count a suggested use as an event.",
    },
]


def summarize(run):
    trace = run["result"].get("trace", {})
    models = []
    for stage in ("classification", "rules", "storyteller", "response_review"):
        record = trace.get(stage, {})
        if "attempts" in record:
            models.extend(attempt.get("model", {}) for attempt in record["attempts"])
        else:
            models.append(record.get("model", {}))
    models += [
        step.get("model", {}) for step in trace.get("decision", {}).get("steps", [])
    ]
    planner = trace.get("rules", {}).get("search_trace", {}).get("planner", {})
    if planner:
        models.append(planner)
    models = [m for m in models if m]
    return {
        "status": run["status"],
        "component_status": {
            stage: trace.get(stage, {}).get("status")
            for stage in ["classification", "decision", "rules", "storyteller"]
        },
        "calls": len(models),
        "prompt_tokens": sum(
            m.get("usage", {}).get("prompt_tokens", 0) for m in models
        ),
        "completion_tokens": sum(
            m.get("usage", {}).get("completion_tokens", 0) for m in models
        ),
        "suggestions": [
            {"kind": p["kind"], "text": p["text"]}
            for p in run["result"].get("suggestions", [])
            if p["kind"] in {"narration", "question", "rule", "note", "action"}
            and not p.get("evidence")
        ],
    }


def write_atomic(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def render(result, destination):
    esc = lambda value: html.escape(str(value))
    cards = []
    reviews_path = destination.parent / "review.json"
    reviews = json.loads(reviews_path.read_text()) if reviews_path.exists() else {}
    for row in result["cases"]:
        columns = []
        for policy, value in row["policies"].items():
            review = reviews.get(row["id"], {}).get(policy, {})
            judgments = review.get("judgments", {})
            review_html = (
                (
                    "<h4>Review · "
                    + esc(review.get("reviewer", "unreviewed"))
                    + "</h4><p>"
                    + esc(
                        " · ".join(
                            k.replace("_", " ") + ": " + v for k, v in judgments.items()
                        )
                    )
                    + "</p><p>"
                    + esc(review.get("notes", ""))
                    + "</p>"
                )
                if review
                else ""
            )
            columns.append(
                f"<section><h3>{esc(policy)}</h3><p>{value['seconds']:.1f}s · {value['calls']} calls · {value['prompt_tokens']} input / {value['completion_tokens']} output tokens</p>"
                f"<p>{esc(value['component_status'])}</p><p>Observed checks: {esc(value['checks'])}</p>"
                + "".join(
                    f"<h4>{esc(s['kind'])}</h4><pre>{esc(s['text'])}</pre>"
                    for s in value["suggestions"]
                )
                + review_html
                + f'<p><a href="{esc(value["trace_file"])}">Complete private trace</a></p></section>'
            )
        cards.append(
            f'<article id="{esc(row["id"])}"><h2>{esc(row["title"])}</h2><p>{esc(row["expect"])}</p><div class="grid">'
            + "".join(columns)
            + "</div></article>"
        )
    document = (
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Story Copilot evaluation</title><style>body{font:16px system-ui;background:#f4eee3;color:#352a20;max-width:1500px;margin:auto;padding:24px}article,section{padding:20px;border:1px solid #d4c4b0;margin:14px 0;border-radius:9px;background:#fffbf4}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}a{color:#765035}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><h1>Story Copilot · live comparison</h1><p>Original authored scenes. Same model configuration and source sequence. Component checks are not semantic-quality judgments. Review each response against the stated expectation and record pass/fail/uncertain in review.json.</p><nav>'
        + " · ".join(
            f'<a href="#{esc(row["id"])}">{esc(row["title"])}</a>'
            for row in result["cases"]
        )
        + "</nav>"
        + "".join(cards)
    )
    write_atomic(destination, document)


def evaluate(home, output, *, policies=("agent", "workflow"), progress=print):
    output = Path(output).expanduser().resolve()
    if output.exists() or any((p / ".git").exists() for p in [output, *output.parents]):
        raise ValueError(
            "Choose a new private evaluation directory outside source repositories."
        )
    output.mkdir(parents=True, mode=0o700)
    configuration = load(home)
    tables = {}
    for policy in policies:
        store = Store(output / policy)
        save(store.home, configuration)
        c = Campaigns(store)
        demo = ensure_demo(c)
        tables[policy] = {
            "store": store,
            "c": c,
            "cid": demo["campaign_id"],
            "sid": demo["session_id"],
        }
    result = {
        "created": now(),
        "application_sources": {
            p.name: digest(p.read_bytes())
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "model": configuration["model"],
        "routing": configuration["routing"],
        "purpose": "Original-scenario live smoke comparison; not a representative benchmark.",
        "rubric": RUBRIC,
        "cases": [],
    }
    for index, step in enumerate(STEPS):
        row = {**{k: step[k] for k in ["id", "title", "expect"]}, "policies": {}}
        result["cases"].append(row)
        # Alternate order to avoid always giving one policy the cold first slot.
        for policy in policies if index % 2 == 0 else tuple(reversed(policies)):
            table = tables[policy]
            store, c, cid, sid = (table[k] for k in ["store", "c", "cid", "sid"])
            if step["id"] == "corrected-cost":
                message = next(
                    m
                    for m in c.messages(sid)
                    if "I spend 1 battery charge" in m["text"]
                )
                c.revise_message(
                    sid,
                    message["id"],
                    text="Correction: Jo did not use the tester. I only considered it. Jo still has 3 battery charges. Can I afford one use?",
                    speaker=message["speaker"],
                    role=message["role"],
                    character=message["character"],
                    visibility=message["visibility"],
                    expected_revision=message["revision"],
                )
            elif step["id"] == "resumed":
                sid = c.continue_session(cid, "Next evening", parent_id=sid)
                table["sid"] = sid
            for speaker, role, text in step["messages"]:
                c.add_message(sid, speaker, text, role=role)
            before = c.messages(sid)
            build, generate = make_copilot(store, policy=policy)
            started = time.monotonic()
            rid = c.generate(sid, generate, build_context=build, force=True)
            run = next(r for r in c.runs(sid) if r["id"] == rid)
            summary = summarize(run)
            summary["seconds"] = round(time.monotonic() - started, 3)
            summary["checks"] = {
                "guidance_not_conversation": c.messages(sid) == before,
                "no_private_guidance_in_public_view": not any(
                    p["id"] in packed(c.snapshot(sid, public_only=True))
                    for p in c.proposals(sid)
                    if not c.is_observation(p)
                ),
                "unchanged_narration_suppressed": not build(c.snapshot(sid))[
                    "narration_needed"
                ],
            }
            if step["id"] in {"corrected-cost", "resumed"}:
                summary["checks"]["corrected_resource_total"] = (
                    c.state(sid)["resources"]["Jo Bell:battery_charges"]["value"] == 3
                )
            summary["trace_file"] = f"{policy}/{step['id']}.json"
            (output / summary["trace_file"]).write_text(json.dumps(run, indent=2))
            row["policies"][policy] = summary
            write_atomic(output / "result.json", json.dumps(result, indent=2))
            render(result, output / "index.html")
            progress(
                f"{step['id']} / {policy}: {summary['status']}, {summary['seconds']:.1f}s, {summary['component_status']}"
            )
    review = {
        row["id"]: {
            policy: {
                "judgments": {key: "unreviewed" for key in RUBRIC},
                "notes": "",
                "reviewer": "",
            }
            for policy in policies
        }
        for row in result["cases"]
    }
    if not (output / "review.json").exists():
        write_atomic(output / "review.json", json.dumps(review, indent=2))
    render(result, output / "index.html")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home", required=True, help="Existing workspace with model settings."
    )
    parser.add_argument(
        "--output", required=True, help="New private evaluation directory."
    )
    args = parser.parse_args()
    evaluate(args.home, args.output, progress=lambda line: print(line, flush=True))


if __name__ == "__main__":
    main()
