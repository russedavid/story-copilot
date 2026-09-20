"""Exercise the real campaign pipeline with an original, inspectable table session.

This is an integration evaluation, not a substitute for narrative review. It uses
the configured local model, creates a labelled evaluation campaign, and preserves
all runs. No podcast material or reference answers are embedded in this module.
"""

import argparse
from html import escape
import json
from pathlib import Path
import time

from .campaigns import Campaigns
from .copilot import make_copilot
from .store import Store, digest, now, packed


def render(report, path):
    panels = []
    for step in report["steps"]:
        outputs = "".join(
            "<h3>"
            + escape(proposal["title"])
            + "</h3><pre>"
            + escape(proposal["text"])
            + "</pre>"
            for proposal in step.get("proposals", [])
        )
        panels.append(
            f"<section><h2>{escape(step['name'])}</h2>"
            f"<p>{escape(packed(step.get('checks', {})))}</p>"
            + outputs
            + f"<details><summary>Inputs, output, evidence and timing</summary>"
            f"<pre>{escape(json.dumps(step, indent=2, ensure_ascii=False))}</pre>"
            "</details></section>"
        )
    path.write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>Facilitator workflow evaluation</title><style>"
        "body{font:16px/1.5 system-ui;background:#f7f2e9;color:#342b23;max-width:1000px;"
        "margin:2rem auto;padding:1rem}section{border-top:1px solid #ad9580;padding:1rem 0}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#744928}</style>"
        "<h1>Real-model workflow evaluation</h1>"
        "<p>Original synthetic conversation. Mechanical checks are separate from "
        "the pending review of narrative quality and player agency.</p>"
        f"<p>Campaign: {escape(report['campaign_id'])}</p>" + "".join(panels)
    )


def evaluate(store, output, *, model_factory=None):
    output = Path(output)
    if output.exists():
        raise ValueError(
            "Choose a new evaluation output; earlier results are immutable."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    campaigns = Campaigns(store)
    cid = campaigns.create(
        "Workflow evaluation — The Station Window",
        direction="Offer short, concrete suggestions. Preserve player decisions. Never narrate a player choosing an unrequested action.",
    )
    sid = campaigns.create_session(
        cid, "Original synthetic evaluation", proactive=False
    )
    campaigns.set_rule_profile(
        cid,
        {
            "resource_aliases": {"hit_points": "HP", "hit_point": "HP"},
            "tools": [
                {
                    "name": "within_limit",
                    "operation": "less_equal",
                    "description": "Compare the explicitly rolled result with the explicitly stated limit.",
                }
            ],
        },
    )
    campaigns.add_document(
        cid,
        "Original fixture rule",
        "In this original fixture, a regular check succeeds when the supplied result is less than or equal to the stated skill limit. A requested check is not itself a resource change.",
        metadata={"kind": "rules"},
    )
    mira = campaigns.save_character(
        cid, "Mira", {"resources": {"HP": 12}, "skills": {"Observation": 60}}
    )
    ivo = campaigns.save_character(
        cid, "Ivo", {"resources": {"HP": 11}, "skills": {"First Aid": 50}}
    )
    campaigns.map_participant(
        cid, "Mira's player", "Mira", role="player", character_id=mira
    )
    campaigns.map_participant(
        cid, "Ivo's player", "Ivo", role="player", character_id=ivo
    )
    campaigns.add_document(
        cid,
        "Public situation",
        "Mira and Ivo are outside a closed railway station. Rain falls. The front door is red.",
        visibility="public",
    )
    campaigns.add_document(
        cid,
        "Private scenario",
        "Behind the station clock is an unclaimed parcel. Its contents are the Facilitator's decision. The characters do not know it exists.",
        visibility="private",
    )
    build, generate = make_copilot(store, model_factory=model_factory)
    report = {
        "created": now(),
        "status": "running",
        "campaign_id": cid,
        "session_id": sid,
        "provenance": "assistant-authored original synthetic interaction; real configured model calls",
        "implementation_sha256": {
            name: digest((Path(__file__).parent / name).read_bytes())
            for name in (
                "system_eval.py",
                "copilot.py",
                "campaigns.py",
                "context.py",
                "runtime_context.py",
                "model.py",
                "rule_advice.py",
                "rules.py",
                "sampling.py",
                "wire_schema.py",
            )
        },
        "quality_review": "pending",
        "steps": [],
    }

    def save():
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        render(report, output.with_suffix(".html"))

    def add(name, checks, **details):
        report["steps"].append({"name": name, "checks": checks, **details})
        save()

    def run(name):
        before = campaigns.state(sid)
        started = time.monotonic()
        rid = campaigns.generate(sid, generate, build_context=build, force=True)
        elapsed = time.monotonic() - started
        record = next((r for r in campaigns.runs(sid) if r["id"] == rid), None)
        proposals = [p for p in campaigns.proposals(sid) if p.get("run_id") == rid]
        # Include the entire trace, even when a component failed. Successful HTTP
        # or JSON decoding alone never counts as a useful assistance result.
        trace = (record or {}).get("result", {}).get("trace", {})
        add(
            name,
            {
                "run_completed": bool(record and record["status"] == "complete"),
                "narration_proposed": any(p["kind"] == "narration" for p in proposals),
                "state_changes_are_source_backed": all(
                    p.get("observed") and p["evidence"]
                    for p in proposals
                    if p["payload"].get("state_changes")
                ),
                "nothing_published_automatically": not campaigns.publications(sid),
            },
            seconds_to_complete_assistance=round(elapsed, 3),
            run=record,
            proposals=proposals,
            state_before=before,
            component_trace=trace,
        )
        return proposals

    campaigns.add_message(
        sid,
        "Facilitator",
        "The station door is closed. You hear a tapping sound from the other side.",
        role="facilitator",
    )
    campaigns.add_message(
        sid,
        "Mira",
        "I listen at the door without opening it. Can I tell where the tapping comes from?",
        role="player",
    )
    campaigns.add_message(
        sid, "Ivo", "I stay outside and watch the empty street.", role="player"
    )
    first = run("Listen without opening the door; respect both players")
    narration = next((p for p in first if p["kind"] == "narration"), None)
    spoken = len(campaigns.messages(sid))
    duplicate = campaigns.generate(sid, generate, build_context=build)
    add(
        "Generated guidance is not speech; unchanged input is suppressed",
        {
            "guidance_did_not_invent_speech": len(campaigns.messages(sid)) == spoken,
            "duplicate_suppressed": duplicate is None,
        },
        duplicate_run_id=duplicate,
    )

    damage_id = campaigns.add_message(
        sid,
        "Facilitator",
        "A loose pane falls. Mira loses two hit points from the falling glass.",
        role="facilitator",
    )
    campaigns.add_message(
        sid,
        "Mira",
        "I step away from the window and ask Ivo to check my cut.",
        role="player",
    )
    damage = run("Explicit damage becomes a reviewable proposal")
    matches = [
        p
        for p in damage
        if any(
            change.get("kind") == "resource"
            and change.get("entity") == "Mira"
            and change.get("delta") == -2
            for change in p.get("payload", {}).get("state_changes", [])
        )
    ]
    hp = campaigns.state(sid)["resources"].get("Mira:HP", {}).get("value")
    add(
        "Spoken resource update without clicking approval",
        {
            "one_supported_damage_proposal": len(matches) == 1,
            "hit_points_updated_from_conversation": hp == 10,
        },
        hp=hp,
    )

    campaigns.add_message(
        sid,
        "Facilitator",
        "Mira, make a regular Observation check. Your Observation skill is 60.",
        role="facilitator",
    )
    campaigns.add_message(
        sid,
        "Mira",
        "I rolled 32. Is that a success at regular difficulty?",
        role="player",
    )
    run("Rules advice from supplied roll inputs")

    campaigns.revise_message(
        sid,
        damage_id,
        text="A loose pane falls. Mira loses one hit point from the falling glass.",
        speaker="Facilitator",
        role="facilitator",
        visibility="public",
        expected_revision=damage_id,
        note="Evaluation correction: the utterance was transcribed incorrectly.",
    )
    revised_hp = campaigns.state(sid)["resources"].get("Mira:HP", {}).get("value")
    add(
        "Correction invalidates outdated observation",
        {"old_damage_removed": revised_hp == 12},
        hp=revised_hp,
    )
    run("Rebuild after source correction")
    add(
        "Corrected speech updates the working state",
        {
            "corrected_hit_points": campaigns.state(sid)["resources"]["Mira:HP"][
                "value"
            ]
            == 11,
        },
    )
    latest = next(
        p for p in reversed(campaigns.proposals(sid)) if p["kind"] == "narration"
    )
    before_refresh = campaigns.state(sid)
    campaigns.decide(
        sid,
        latest["id"],
        "rejected",
        "Assistant evaluation requests a private alternative.",
    )
    count_before = len(campaigns.messages(sid))
    run("Reject and refresh private guidance")
    add(
        "Refreshing suggestions does not change the record",
        {
            "no_added_speech": len(campaigns.messages(sid)) == count_before,
            "no_state_change": campaigns.state(sid) == before_refresh,
            "no_player_suggestions": campaigns.snapshot(sid, public_only=True)[
                "proposals"
            ]
            == [],
        },
    )
    branch = campaigns.branch(sid, "Evaluation branch")
    add(
        "Branch preserves observed state without duplicating effects",
        {
            "resource_values_equal": {
                key: (value.get("value"), value.get("known_delta"))
                for key, value in campaigns.state(branch)["resources"].items()
            }
            == {
                key: (value.get("value"), value.get("known_delta"))
                for key, value in campaigns.state(sid)["resources"].items()
            },
            "source_history_preserved": len(campaigns.messages(branch))
            == len(campaigns.messages(sid)),
        },
        branch_id=branch,
    )
    report["status"] = "complete"
    report["finished"] = now()
    report["summary"] = {
        "checks_passed": sum(
            value is True
            for step in report["steps"]
            for value in step["checks"].values()
        ),
        "checks_total": sum(len(step["checks"]) for step in report["steps"]),
        "note": "Mechanical integration checks only; inspect every response for narrative and semantic quality.",
    }
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home",
        required=True,
        help="Private library; a new labelled evaluation campaign is added.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(Store(args.home), args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result["summary"],
                "campaign_id": result["campaign_id"],
            }
        )
    )


if __name__ == "__main__":
    main()
