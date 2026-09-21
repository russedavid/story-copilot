"""Check a served policy against authored development tasks after export/quantization."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from .model import LocalModel
from .rl_environment import EvidenceDecision, EvidenceEpisode, make_suite
from .settings import load
from .store import packed


def evaluate(home, output, *, count_per_family=4, start_seed=40000):
    output = Path(output).resolve()
    if output.exists() or any((p / '.git').exists() for p in [output, *output.parents]):
        raise ValueError('Choose a new private output directory.')
    output.mkdir(parents=True, mode=0o700)
    configuration = load(home)
    model = LocalModel(task='planner', configuration=configuration)
    report = {'purpose': 'Greedy serving/export development check; separate from the reserved training evaluation.',
              'configuration': configuration, 'cases': []}
    for case in make_suite(count_per_family, split='validation', start_seed=start_seed):
        episode = EvidenceEpisode(case)
        messages = deepcopy(case['prompt'])
        calls = []
        started = time.monotonic()
        error = None
        while not episode.done:
            try:
                answer, metrics = model.complete(messages, EvidenceDecision, max_tokens=320, temperature=0, timeout=30, constrain=False)
                raw = metrics.get('response') or packed(answer.model_dump(exclude_defaults=True))
                observation = episode.step(raw)
                calls.append(metrics)
                messages.append({'role':'assistant','content':raw})
                if not episode.done:
                    messages.append({'role':'user','content':'Evidence result: '+packed(observation)})
            except Exception as exc:
                error = {'message': str(exc), 'trace': getattr(exc, 'trace', {})}
                break
        row = {'id':case['id'], 'family':case['family'], 'source':case,
               'assessment':episode.assessment(), 'final':episode.final, 'trace':episode.trace,
               'model_calls':calls, 'seconds':time.monotonic()-started, 'error':error}
        report['cases'].append(row)
        report['passes'] = sum(x['assessment']['success'] for x in report['cases'])
        (output/'report.json').write_text(json.dumps(report, indent=2))
        print(f'{case["family"]}: {row["assessment"]["success"]} ({row["seconds"]:.2f}s)', flush=True)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home',required=True);p.add_argument('--output',required=True)
    p.add_argument('--count-per-family',type=int,default=4);p.add_argument('--start-seed',type=int,default=40000)
    evaluate(**vars(p.parse_args()))
