"""Replay private response/context pairs through the real grounded editing stage."""

import argparse
from html import escape
import json
from pathlib import Path
import time

from .model import LocalModel
from .recorded_audio_eval import private_output
from .response_review import review_response
from .runtime_context import context_options
from .schema import DirectAnswer, NarrationAnswer
from .settings import load
from .store import Store


def evaluate(cases, home, output):
    inputs = json.loads(Path(cases).read_text())
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Supply a nonempty list of reviewed context/response cases.")
    store = Store(home)
    configuration = load(home)
    output = private_output(output)
    report = {
        "kind": "grounded_response_regressions",
        "status": "running",
        "cases": [],
        "limits": "Validation status and abstention are not semantic-quality scores. Review usefulness and source fidelity separately.",
    }
    for case in inputs:
        schema = (
            DirectAnswer
            if not case["answer"].get("narration")
            and case["answer"].get("direct_answer")
            else NarrationAnswer
        )
        original = schema.model_validate(case["answer"])
        started = time.monotonic()
        answer, trace = review_response(
            case["context"],
            original,
            LocalModel(task="auditor", configuration=configuration),
            context_options(store),
        )
        row = {
            "id": case["id"],
            "expect": case["expect"],
            "original": original.model_dump(),
            "answer": answer.model_dump(),
            "trace": trace,
            "seconds": time.monotonic() - started,
            "context": case["context"],
            "quality_review": "pending",
        }
        report["cases"].append(row)
        (output / "report.json").write_text(json.dumps(report, indent=2))
        (output / "review.html").write_text(
            '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            "<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:auto;padding:24px;background:#f4eee3;color:#352a20}section{border-top:1px solid #bca68e}pre{white-space:pre-wrap;overflow-wrap:anywhere}details pre{max-height:34rem;overflow:auto}</style>"
            "<h1>Grounded response regressions</h1><p>A guarded or empty response is not automatically a useful answer. Review each result against its sources.</p>"
            + "".join(
                f"<section><h2>{escape(x['id'])}</h2><p>{escape(x['expect'])}</p>"
                f"<h3>Before</h3><pre>{escape(json.dumps(x['original'], indent=2))}</pre>"
                f"<h3>After</h3><pre>{escape(json.dumps(x['answer'], indent=2))}</pre>"
                f"<details><summary>Source and claim assessments</summary><pre>{escape(json.dumps(x, indent=2))}</pre></details></section>"
                for x in report["cases"]
            )
        )
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "status": trace["status"],
                    "seconds": row["seconds"],
                }
            ),
            flush=True,
        )
    report["status"] = "complete"
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "home", "output"):
        p.add_argument("--" + name, required=True)
    evaluate(**vars(p.parse_args()))
