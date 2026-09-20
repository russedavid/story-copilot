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
