# Evaluation

There are two separate questions: does the application preserve its contracts, and does the model give useful guidance?

The CPU test suite checks the first: independent validation of event proposals, exact citations, conversation revisions, correction invalidation, campaign rule isolation, current numeric inputs, private suggestions, bounded evidence decisions, continuation, compaction, and audio ordering. Model responses in these tests are scripted; passing them is not a model-quality score.

For model quality, use original authored scenarios or data you have permission to evaluate. Judge behavior against a rubric rather than a single ideal paragraph. A useful response may have many valid wordings.

The live scenario sequence should include a player's changed choice, a corrected fact, information known to only one character, a request requiring a supplied rule, an unknown rule or input, an unchanged-context request, and resuming the next session. Review whether the guidance answers the current contribution, respects player agency, preserves character knowledge, avoids invented outcomes, retrieves necessary evidence, and remains usable as dialogue. Record failures and uncertainty, not only successful examples.

Compare the default agent with `make_copilot(..., policy="workflow")`. This baseline retains source classification, retrieval, rules advice, narration, and context packing; it is not an intentionally weakened one-shot prompt. Keep model, adapter routing, source sequence, and budgets the same. Report end-to-end latency, calls and token usage alongside judgments. Faster or more elaborate is not automatically better.

Run traces contain private inputs and outputs. Keep them in a separate data directory, never in Git. Publish only deliberately reviewed, non-identifying results and original examples. A small authored suite is a smoke test and error-discovery tool, not a representative benchmark.
