"""Original live player-agent checks; references and traces stay in a private workspace."""

import argparse
from copy import deepcopy
import html
import json
from pathlib import Path
import time

from .campaigns import Campaigns
from .model import LocalModel
from .players import Players
from .settings import load, save
from .store import Store, packed


PROFILES = {
    "curious": "Be a patient, curious player. Notice concrete details, connect clues, and ask one useful question when the situation is unclear. Build on other participants contributions. Keep your characters actions and uncertainties clear.",
    "decisive": "Be a practical, decisive player. Choose a concrete next step your character can attempt, explain your approach briefly, and keep the scene moving. Invite cooperation without choosing for others. Ask the facilitator to resolve uncertain outcomes.",
}


def render(report, output):
    esc = lambda value: html.escape(str(value))
    sections = []
    for case in report["cases"]:
        run = case["run"]
        answer = run["result"].get("player", {}).get("answer", {})
        sections.append(
            "<section><h2>"
            + esc(case["profile"] + " · " + case["scenario"])
            + "</h2><p>"
            + esc(case["expect"])
            + "</p><p>"
            + esc(case["checks"])
            + "</p><h3>Player contribution</h3><pre>"
            + esc(answer.get("utterance", "No posted utterance."))
            + "</pre><p>Recipient: "
            + esc(answer.get("recipient", ""))
            + "</p><details><summary>Complete context and trace</summary><pre>"
            + esc(json.dumps(run, indent=2, ensure_ascii=False))
            + "</pre></details></section>"
        )
    output.write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Player-agent workflow evaluation</title><style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:24px;background:#f4eee3;color:#352a20}section{background:#fffbf4;padding:20px;margin:18px 0;border:1px solid #d8c8b4;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Player-agent workflow evaluation</h1><p>Original fictional scenarios. Automated checks cover access, routing and state integrity; the player contributions require separate semantic review.</p>'
        + "".join(sections)
    )


def evaluate(home, output, adapters, *, progress=print):
    output = Path(output).expanduser().resolve()
    if output.exists() or any((p / ".git").exists() for p in [output, *output.parents]):
        raise ValueError("Choose a new private output directory outside repositories.")
    store = Store(output / "workspace")
    save(store.home, load(home))
    c = Campaigns(store)
    players = Players(store)
    report = {
        "purpose": "Original live workflow checks, not a population accuracy estimate.",
        "cases": [],
    }
    for profile, adapter in adapters.items():
        for scenario in ["shared-choice", "private-clue", "source-correction"]:
            cid = c.create(
                "Original canal inspection", direction="FACILITATOR_DIRECTION_ONLY_9d23"
            )
            own = c.save_character(
                cid,
                "Neri",
                {
                    "occupation": "mechanic",
                    "resources": {"lamp_charge": 2},
                    "inventory": ["small lamp", "inspection mirror"],
                },
            )
            other = c.save_character(cid, "Sol", {"notes": "OTHER_SHEET_ONLY_f731"})
            c.add_document(
                cid,
                "Public scene",
                "A closed sluice separates a dry canal from an abandoned glasshouse. The caretaker waits near a handwheel.",
                visibility="public",
            )
            c.add_document(
                cid,
                "Private plot",
                "FACILITATOR_SECRET_ONLY_a864",
                visibility="private",
            )
            pid = players.save_profile(
                profile + " " + scenario, PROFILES[profile], adapter_id=adapter
            )
            binding = players.assign(cid, pid, own)
            sid = c.create_session(cid, scenario)
            c.add_message(
                sid,
                "Facilitator",
                "The caretaker asks whether you want to inspect the handwheel or follow the dry canal. Nothing has been opened or repaired yet.",
                role="facilitator",
            )
            c.add_message(
                sid,
                "Facilitator",
                "OTHER_PRIVATE_ONLY_b732",
                role="facilitator",
                recipient=other,
            )
            if scenario == "private-clue":
                c.add_message(
                    sid,
                    "Facilitator",
                    "Privately, Neri notices a fresh scratch on the handwheel. Sol has not seen the scratch.",
                    role="facilitator",
                    recipient=own,
                )
            before = deepcopy(c.state(sid))
            count = len(c.messages(sid))
            captured = []
            config = load(store.home)
            config["routing"]["tasks"]["player"] = adapter

            class Client:
                def __init__(self, task):
                    self.task = task
                    self.model = LocalModel(task=task, configuration=config)

                def complete(self, messages, schema, **kwargs):
                    captured.append(packed(messages))
                    value = self.model.complete(messages, schema, **kwargs)
                    if scenario == "source-correction" and self.task == "player":
                        first = c.messages(sid)[0]
                        c.revise_message(
                            sid,
                            first["id"],
                            text="The caretaker withdraws the invitation. Please wait without touching the mechanism.",
                            speaker=first["speaker"],
                            role=first["role"],
                            visibility=first["visibility"],
                            expected_revision=first["revision"],
                        )
                    return value

            started = time.monotonic()
            rid = players.request(sid, binding, model_factory=Client)
            run = next(r for r in players.runs(sid) if r["id"] == rid)
            joined = "\n".join(captured)
            checks = {
                "facilitator_secrets_excluded": "FACILITATOR_SECRET_ONLY_a864"
                not in joined
                and "FACILITATOR_DIRECTION_ONLY_9d23" not in joined,
                "other_sheet_and_private_message_excluded": "OTHER_SHEET_ONLY_f731"
                not in joined
                and "OTHER_PRIVATE_ONLY_b732" not in joined,
                "no_world_state_mutation": c.state(sid) == before,
                "correct_adapter_selected": run["result"]
                .get("player", {})
                .get("model", {})
                .get("adapter_id")
                == adapter,
            }
            if scenario == "source-correction":
                checks.update(
                    stale_turn_not_posted=run["status"] == "stale"
                    and len(c.messages(sid)) == count
                )
                expect = "The real model response is discarded because its visible source changed before posting."
            else:
                checks["one_labelled_player_turn"] = (
                    run["status"] == "complete"
                    and len(c.messages(sid)) == count + 1
                    and c.messages(sid)[-1]["source"].get("kind") == "player_agent"
                )
                try:
                    players.request(sid, binding, model_factory=Client)
                    checks["unchanged_turn_suppressed"] = False
                except ValueError:
                    checks["unchanged_turn_suppressed"] = True
                expect = "A concise player contribution reflecting the personality; choose only this character’s actions, leave outcomes to the facilitator, and respect the private clue’s audience."
            report["cases"].append(
                {
                    "profile": profile,
                    "scenario": scenario,
                    "seconds": round(time.monotonic() - started, 3),
                    "expect": expect,
                    "checks": checks,
                    "run": run,
                    "quality_review": "pending",
                }
            )
            (output / "report.json").write_text(json.dumps(report, indent=2))
            render(report, output / "review.html")
            progress(profile + " / " + scenario + ": " + str(checks))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--curious-adapter", type=int, required=True)
    parser.add_argument("--decisive-adapter", type=int, required=True)
    args = parser.parse_args()
    evaluate(
        args.home,
        args.output,
        {"curious": args.curious_adapter, "decisive": args.decisive_adapter},
        progress=lambda line: print(line, flush=True),
    )


if __name__ == "__main__":
    main()
