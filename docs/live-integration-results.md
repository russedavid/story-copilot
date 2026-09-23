# Live model integration findings

The later [response-quality follow-up](response-quality-results.md) tests repairs for the prose failures below, evaluates retained natural conversation, and records the new storyteller comparison. The measurements on this page describe the earlier development pass.

This development pass connected the trained evidence policy to the live application, compared the existing task and player adapters with the base model, and replayed authored speech through actual transcription and assistance. Model outputs were reviewed separately from mechanical checks. Reviews were performed by an assistant and were not calibrated against independent human labels.

## What the replay found

The source sequence includes an NPC question, an inventory correction, an unaffordable action, a replacement rule, hypothetical spending, a character-only clue, and a change of task. The speech recognizer's actual output is imported; the reference script is not substituted for it. Explicit fixture speaker assignments stand in for an operator's identity mapping.

Earlier runs exposed several faults:

- Sentence-sized ASR messages let a trailing qualification replace the complete question. The current contribution now retains a bounded, same-speaker audio chunk, without joining different private recipients.
- A facilitator update could keep the earlier player question as the current trigger. The newest contribution now determines the trigger, while unanswered earlier contributions remain available in context.
- The editing pass restored a starting resource total over a valid correction. Packed model context now keeps current totals in the resource ledger and omits historical resource totals from the embedded starting sheet. The original sheet remains in storage and explicit retrieval.
- A numerical question labelled as ordinary state could skip the rules verifier. Explicit numerical rulings now require the verification stage independently of the planner's label.
- A model put a character-only clue in narration suitable for the whole table. Character-addressed updates now request a private answer with an explicit audience and an empty narration field. This is a presentation boundary; it does not prove every generated sentence is semantically correct.
- A serving grammar prevented the learned policy from emitting its trailing citation field. The policy now uses JSON-object generation with full schema validation afterward, preserving its trained serialization.

The failed runs remain in the private evaluation record. These fixes do not turn model editing into an authoritative judge of narrative quality.

The final eight-step replay passed all 58 mechanical assertions and an additional real-generation stale-source check. Assistance completion took 24.4–62.5 seconds, with a 39.2-second median, excluding ASR and the time spent speaking. These are sequential replay timings, not live-stream throughput. The actor-private response used its private answer format and left shared narration empty.

Strict assistant review passed seven of eight steps. The private-clue response still invented a player looking concerned, despite correctly keeping the clue private and leaving observed state unchanged. That is a remaining prose/agency failure, not a passing result hidden by green structural checks. The application still requires human judgment about generated wording. The broader seven-scene regression also showed that a model can incorrectly paraphrase an unspecified bonus as zero in private notes; absence of a supplied rule must remain unknown.

## Selected model roles

| Work | Selection | Development evidence |
| --- | --- | --- |
| Source classification | Trained classifier adapter | Both paired cases preserved the inventory correction and hypothetical non-spending. The adapter avoided extra recap events and took 10.1/7.3 seconds versus 21.4/10.0 for the base on those same requests. |
| Numerical rules advice | Trained rules adapter, with deterministic verification | Both candidates correctly handled unaffordable and revised affordable costs. Adapter calls took 10.9/13.4 seconds versus 12.6/16.0 for the base. |
| Narration | Base model | The storyteller adapter repeated abandoned numerical advice after the task changed. A short NPC reply alone did not justify selecting it. |
| Editing | Untuned base | Independent of the selected task adapter, but still fallible; its original draft and edits remain visible. |
| Numerical evidence decisions | Q8 serving copy of the trained 4B RL policy | Same structured conclusions and citations as BF16 in 24/24 serving development cases; both passed 23/24 complete contracts. Broader narrative and social decisions retain the main planner. |
| Player turns | Separate curious and decisive adapters, with a base option | All access, routing, duplicate, stale-write, and world-state checks passed in six runs per candidate. An unblinded review preferred adapters in three of four ordinary-turn pairs and the base in one. This is a small subjective comparison. |

These are deployment choices for the tested local setup, not universal adapter rankings. The default repository configuration does not assume these weights exist; bring your own model server and configure its actual inventory.

## Memory and speech

The larger writing model occupies one GPU; the smaller decision model and speech worker share the other. With Q8 serving, two clean synthesized clips of about 35 and 33 seconds completed while actual planner calls overlapped speech processing. Sampled GPU memory peaked at 21,177 MiB on the shared 24 GiB device. Warm speech processing took 2.36 and 2.06 seconds; the first clip also incurred 6.88 seconds of model loading.

This is a bounded coexistence check, not a maximum-capacity guarantee. The speaker count was supplied, the voices were synthesized, and there was no overlapping speech. Neither its word error rate nor its timing establishes natural-table transcription or live-stream accuracy. The application never starts recording merely because it is opened.

See [reproduction commands and review expectations](evaluation.md#audio-through-the-full-workflow), [live planner configuration](live-planner.md), and the separate [training and reserved-evaluation results](https://github.com/russedavid/qwen-ttrpg/blob/main/docs/agent-rl-results.md).
