# Development validation

The application is tested at two levels: CPU regressions for its source/state contracts, and live model runs for the assistance people actually receive. A completed request is not a quality verdict.

## Local comparison, 20 September 2026

An original seven-step conversation was run through the agent and fixed-workflow policies at application revision `8165495`. Both used the same quantized 27B base, with all task adapters explicitly disabled, a native tokenizer, and the same 16,384-token preferred context budget. Both included the response editing pass. Execution order alternated by step.

The text model ran on one RTX 3090; a speech worker remained loaded but idle on the second card. No audio was captured or played for this comparison.

| Measure | Agent | Fixed workflow |
| --- | ---: | ---: |
| Scenarios completed | 7 | 7 |
| Model calls | 30 | 27 |
| Median time to assistance | 59.47 s | 52.62 s |
| Observed range | 30.03–86.97 s | 39.31–62.99 s |
| Source, privacy, novelty, and correction checks | 23/23 | 23/23 |

Those 46 checks cover whether suggestions altered the conversation, appeared in the player-visible record, repeated on unchanged context, or lost a corrected resource total. They do **not** establish whether the writing answered every question or respected every character boundary.

Assistant review of all 14 responses found remaining failures: an unanswered NPC question, a time-arithmetic error, confusion about who had asked whom a question, a missing rule described as a zero bonus, and factual answers buried under scene prose. This small development set does not establish an accuracy rate or a general advantage for the agent. The agent also adds latency, although it can end early with a useful clarification.

## Changes driven by the review

The current implementation distinguishes narrative, rules, state, and player-choice intents. A rules intent requires scoped rule advice before a ruling. Missing rule information ends in a clarification. State and rules answers use an explicit answer slot and can omit scene narration. Kind-specific event schemas prevent incompatible resource/stage combinations during constrained generation; application validation still checks every source reference.

Two targeted live follow-ups on already-analyzed conversation snapshots checked these changes. The missing-rule case returned a clarification without a numerical bonus in 23.04 seconds. The state question returned the current resource count and the character's knowledge boundary in 15.27 seconds. These are targeted regressions with reused analysis, **not** fresh-run latency comparisons or held-out quality measurements.

Original and revised drafts, model metadata, token usage, validation failures, and reviewer notes remain in private traces. The repository includes the authored scenario, comparison runner, and review renderer so another operator can run the same procedure with their own model. Generated output, source data, and model weights are not included.

Model editing is fallible and can overcorrect. It is not an independent evaluation label. Continue to inspect responses for missed questions, unnecessary scene development, information leaks, and invented player performance when changing a model or an adapter.

## Player agents, 20 September 2026

Two invented player identities, curious and decisive, now have separate trained LoRAs. A live run used the actual adapters through the player planner, scoped evidence tools, and final utterance path. The six original cases exercised a shared choice, a private clue, and a source change made after the real model generated its answer.

All 34 specified software checks passed: facilitator secrets and other-character material stayed out of model input; requests selected the intended adapter; posted contributions carried AI attribution; unchanged requests were suppressed; and corrected sources discarded stale replies. World state did not change merely because an AI player spoke. End-to-end time ranged from 6.62 to 8.37 seconds. These numbers describe one warm local run, with one request at a time.

Assistant review found the posted turns bounded and usable, although one decisive turn was generic and the personas were not consistently distinct. One discarded draft asked another participant to use a lamp without explicitly arranging the handoff. Source isolation is tested independently of whether a model interprets the material correctly.

The adapters' held-out completion loss improved, but blind review of six response pairs per identity did not show a consistent preference over the base. One decisive response repeated until the output limit with a presence penalty of zero. Restoring the model vendor's non-thinking narrative preset completed all six repeated cases for both candidates. The six application scenarios also retained all 34 checks after that change, taking 6.25–8.46 seconds. These were development regressions, not independent quality results. [Training, comparison protocol, and limitations](https://github.com/russedavid/qwen-ttrpg/blob/main/docs/player-results.md) are recorded separately.
