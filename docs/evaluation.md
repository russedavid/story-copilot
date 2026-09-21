# Evaluation

There are two separate questions: does the application preserve its contracts, and does the model give useful guidance?

The CPU test suite checks the first: independent validation of event proposals, exact citations, conversation revisions, correction invalidation, campaign rule isolation, current numeric inputs, private suggestions, bounded evidence decisions, continuation, compaction, and audio ordering. Model responses in these tests are scripted; passing them is not a model-quality score.

For model quality, use original authored scenarios or data you have permission to evaluate. Judge behavior against a rubric rather than a single ideal paragraph. A useful response may have many valid wordings.

The live scenario sequence should include a player's changed choice, a corrected fact, information known to only one character, a request requiring a supplied rule, an unknown rule or input, an unchanged-context request, and resuming the next session. Review whether the guidance answers the current contribution, respects player agency, preserves character knowledge, avoids invented outcomes, retrieves necessary evidence, and remains usable as dialogue. Record failures and uncertainty, not only successful examples.

Compare the default agent with `make_copilot(..., policy="workflow")`. This baseline retains source classification, retrieval, rules advice, narration, and context packing; it is not an intentionally weakened one-shot prompt. Keep model, adapter routing, source sequence, and budgets the same. Report end-to-end latency, calls and token usage alongside judgments. Faster or more elaborate is not automatically better.

Run traces contain private inputs and outputs. Keep them in a separate data directory, never in Git. Publish only deliberately reviewed, non-identifying results and original examples. A small authored suite is a smoke test and error-discovery tool, not a representative benchmark.

Run the live comparison against a configured local workspace:

```sh
python -m story_copilot.evaluate --home /path/to/workspace --output /path/to/new/private/evaluation
```

It creates separate campaign workspaces for the agent and workflow policies, alternates their execution order, and saves each full trace. Open `index.html` for a side-by-side review and record judgments in `review.json`. The run uses your configured model endpoint; it does not download a model or turn on audio. Keep this evaluation directory outside Git.

The response editing pass is part of both policies in the current comparison. Its verdict is an application output, not an evaluation label. An external review may disagree with it. For example, it may correctly remove invented dialogue while being overcautious about a harmless restatement of something a player already said. Keep original and revised drafts available, and judge the final assistance against the source.

## Player agents

Run the original player scenarios with two adapters already loaded and listed in Model settings:

```sh
python -m story_copilot.player_eval --home /path/to/workspace \
  --output /path/to/new/private/player-evaluation \
  --curious-adapter 3 --decisive-adapter 4
```

Use the actual adapter IDs from your server. The six cases cover a shared choice, a character-private clue, and a source correction during generation for each identity. The runner checks what reached the model, which adapter was selected, whether a turn was posted, duplicate suppression, and unchanged world state. Read `review.html` for utterances and full traces. Record semantic judgments in each case's `quality_review` in `report.json`; the renderer displays those judgments separately from automatic checks. Set `STORY_EXPERIMENTS` to the private report directory's parent to browse reports in the app.

For training comparisons, use the companion toolkit's paired benchmark and review shuffled responses before unblinding. A repeated run after changing sampling is a development check. It does not restore the independence of cases already used to choose the change.

## Audio through the full workflow

The authored audio suite covers an NPC question, a corrected inventory, an unaffordable action, a revised rule, hypothetical spending, a private clue, and a changed task. It checks idempotent imports, duplicate suppression, current state, verified cost calculations, and source edits during real generation. The complete traces remain navigable beside each response.

Generate the optional file-only speech fixture on macOS (requires `ffmpeg`):

```sh
python -m story_copilot.prepare_workflow_audio --output /path/to/private/audio-fixtures
```

This writes synthesized speech without playing it or opening a microphone. On the GPU worker, in an environment containing the application's dependencies plus the audio dependencies:

```sh
python -m story_copilot.transcribe_workflow \
  --fixtures /path/to/private/audio-fixtures \
  --output /path/to/private/transcribed --device-index 1
```

The worker uses the actual queue, transcription, alignment, diarization, and chunk-ownership code. An optional `--planner-home /path/to/planner/workspace` overlaps an actual planner call with each speech chunk and records the overlap. Its `seconds` includes waiting for both operations; use the worker's separate timing fields for ASR latency. GPU indices are relative to `CUDA_VISIBLE_DEVICES` when set.

Bring the fixtures and transcription report to the UI machine, then run:

```sh
python -m story_copilot.workflow_eval --home /path/to/workspace \
  --fixtures /path/to/private/audio-fixtures \
  --transcribed /path/to/private/transcribed/report.json \
  --output /path/to/private/workflow-review
```

ASR text is not replaced with the reference script. Fixture identities and audiences are assigned explicitly, as an operator would map speakers; diarization does not establish a person's identity. The rule-revision step explicitly replaces the rule document. It does not claim that a spoken announcement edits the rule library automatically.

The clean speech uses known one-speaker chunks and a shared synthetic schedule. Its word error rate retains number-format differences and cannot establish natural-speech or overlapping-speaker accuracy. End-to-end figures sum measured ASR and assistance time but exclude speech duration, network transit, and scheduling delays. Semantic judgments must read the source, including actual transcription errors; mechanical green checks alone are insufficient.

Compare existing task adapters against the base on identical saved workflow requests:

```sh
python -m story_copilot.adapter_eval --home /path/to/workspace \
  --workflow /path/to/private/workflow-review/report.json \
  --classifier 0 --rules 1 --storyteller 2 \
  --output /path/to/private/adapter-review
```

Use your server's actual adapter IDs. The suite does not train on the evaluation cases or silently select a winner. Judge current intent, source fidelity, rule correctness, player agency, knowledge boundaries, and usable voice separately before setting routing defaults.
