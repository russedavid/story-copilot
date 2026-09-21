"""Compare base and task adapters on identical requests from a private workflow trace."""

import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import time

from .copilot import COPILOT_SYSTEM, _validate_event
from .model import EXTRACT_SYSTEM, LocalModel
from .rule_advice import checked_advice
from .schema import Extraction, NarrationAnswer
from .settings import load
from .store import packed


def evaluate(home, workflow, output, *, adapters, progress=print):
    output = Path(output).resolve()
    if output.exists() or any((p / '.git').exists() for p in [output, *output.parents]):
        raise ValueError("Choose a new private comparison directory.")
    output.mkdir(parents=True, mode=0o700)
    configuration = load(home)
    source = json.loads(Path(workflow).read_text())
    report = {"purpose": "Paired component development comparison on identical recorded inputs; not held-out population accuracy.",
              "input_report": str(Path(workflow).resolve()), "reviewer": "pending", "cases": []}
    tasks = {"classifier": ["inventory-correction", "affordable"],
             "rules": ["unaffordable", "affordable"],
             "storyteller": ["npc-question", "changed-task"]}
    for task, identifiers in tasks.items():
        for index, identifier in enumerate(identifiers):
            row = next(x for x in source['cases'] if x['id'] == identifier)
            trace = row['run']['result']['trace'][task if task != 'classifier' else 'classification']
            for label in (['base', 'adapter'] if index % 2 == 0 else ['adapter', 'base']):
                config = deepcopy(configuration)
                config['routing']['tasks'][task] = adapters[task] if label == 'adapter' else None
                model = LocalModel(task=task, configuration=config)
                case = {'task': task, 'scenario': identifier, 'candidate': label,
                        'expected': row['expect'], 'quality_review': 'pending'}
                started = time.monotonic()
                try:
                    if task == 'rules':
                        request = trace['request']
                        answer, metrics, calculation, attempts = checked_advice(model, request)
                        case.update(input=request, answer=answer.model_dump(), model=metrics,
                                    calculation=calculation, attempts=attempts)
                    elif task == 'classifier':
                        request = trace['request']
                        answer, metrics = model.complete(
                            [{'role': 'system', 'content': EXTRACT_SYSTEM}, {'role': 'user', 'content': packed(request)}],
                            Extraction, max_tokens=3000, temperature=0)
                        valid, rejected = [], []
                        for event in answer.events:
                            try:
                                checked, _ = _validate_event(event, request['target_turns'], row['run']['request']['state'],
                                                             row['run']['request']['frozen_snapshot'].get('rule_profile'))
                                valid.append(checked.model_dump())
                            except ValueError as exc:
                                rejected.append({'event': event.model_dump(), 'error': str(exc)})
                        case.update(input=request, answer=answer.model_dump(), model=metrics,
                                    valid_events=valid, rejected_events=rejected)
                    else:
                        messages = deepcopy(trace['messages'])
                        messages[0]['content'] = COPILOT_SYSTEM
                        answer, metrics = model.complete(messages, NarrationAnswer, max_tokens=1400, temperature=0.7)
                        case.update(input=messages, answer=answer.model_dump(), model=metrics)
                    case['status'] = 'complete'
                except Exception as exc:
                    case.update(status='failed', error=str(exc), trace=getattr(exc, 'trace', {}))
                case['seconds'] = time.monotonic() - started
                report['cases'].append(case)
                (output / 'report.json').write_text(json.dumps(report, indent=2))
                (output / 'review.html').write_text(
                    '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                    '<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:auto;padding:24px;background:#f4eee3;color:#352a20}section{border-top:1px solid #bca68e}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
                    '<h1>Task-adapter comparison</h1><p>Same input, configured base versus task adapter. Semantic review is separate from schema and source checks.</p>'
                    + ''.join(f'<section><h2>{escape(x["task"])} · {escape(x["scenario"])} · {escape(x["candidate"])}</h2>'
                              f'<p>{escape(x["expected"])}</p><pre>{escape(json.dumps(x.get("answer", x.get("error")), indent=2))}</pre>'
                              f'<details><summary>Complete request and trace</summary><pre>{escape(json.dumps(x, indent=2))}</pre></details></section>'
                              for x in report['cases']))
                progress(f'{task}/{identifier}/{label}: {case["status"]}, {case["seconds"]:.1f}s')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('home', 'workflow', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--classifier', required=True, type=int)
    parser.add_argument('--rules', required=True, type=int)
    parser.add_argument('--storyteller', required=True, type=int)
    args = vars(parser.parse_args())
    adapters = {k: args.pop(k) for k in ('classifier', 'rules', 'storyteller')}
    evaluate(**args, adapters=adapters, progress=lambda line: print(line, flush=True))
