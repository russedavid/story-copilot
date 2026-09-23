# Response quality and natural-conversation follow-up

This development pass retrained the storyteller, compared three writing candidates, and tested the application's final responses separately. Assistant review judged source fidelity and usefulness; no independent human calibration or acoustic gold is claimed.

## What changed

Earlier runs invented player reactions, described an unspecified modifier as zero, and put invented retrospective events into private notes. Source-grounded review now covers player, numerical-rule, private-note, and other declarative world claims, including short assertions in NPC dialogue. Supported player facts retain verified source wording. An explicit unknown cannot support a definite outcome or absence.

The review retains useful NPC invention and independently supported answer clauses. A bounded repair cannot quietly restore an older rejected draft. When it cannot repair an unsupported certainty, it can return the relevant known and unknown source facts as private guidance. Original responses, exact support, validation failures, source substitutions and repairs remain in the trace. These checks constrain specific failure modes; they are not a proof that arbitrary prose is true.

Factual, numerical and character-private responses use the base writer with deterministic sampling, even if a workspace selects a creative adapter. Creative narration and player turns retain their configured model choices. Guidance remains private and cannot update the world merely by being generated.

## Model decision

The response-focused training run completed on two 24 GB GPUs. All 672 saved adapter tensors passed exact reload checks, and conversion for local serving succeeded. A blinded pilot compared base, previous and new storyteller adapters over 96 generations from eight authored scenario groups, with two turns and two seeds per group.

The new adapter earned 16.5 preference points, compared with 8 for the base and 7.5 for the previous adapter. Strict source-and-task passes were 21/32, 22/32 and 16/32 respectively. Better voice and lower validation loss did not justify replacing the base default. The candidate remains experimental. The [training report and protocol](https://github.com/russedavid/qwen-ttrpg/blob/main/docs/storyteller-results.md) describe the split, labels, metrics and limitations.

Application replays found a further error in both candidate and base writing: an NPC supplied a definite answer to an explicitly unknown event. The editor even cited the unknown statement as proof of certainty. Those failed runs were retained, the release was paused, and broader claim coverage plus the explicit uncertainty check were tested. A separate source-first review experiment did not consistently improve repairs and was not shipped. These are development iterations, not fresh held-out model improvements.

## Full replay and development follow-ups

| Evaluation | Cases | Mechanical assertions | Assistant semantic passes | Seconds to completed assistance | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Saved defect and positive-control replays | 9 | Not a quality score | 9/9 | 4.6–74.3 | 15.4 |
| Authored audio through the application | 8 | 58/58 | 5/8 | 40.6–127.4 | 110.8 |
| Retained natural conversation | 4 | 16/16 | 4/4 | 16.3–110.8 | 70.8 |

The saved-case set includes the actual failures and controls for supported player facts, useful NPC replies, and NPC dialogue addressing a player. Its timing measures editing only. The full workflow covers inventory corrections, replacement rules, hypothetical spending, character-private information and a changed task. It uses actual saved recognizer output from the authored audio fixtures, without replacing it with the reference script. A separate real-generation source-change check tests stale-output rejection.

Timing is sequential local replay, excluding the time spent speaking and ASR. It is not live-stream latency or maximum throughput. The full replay above used the earlier 90-second review deadline. It exposed three usefulness failures despite passing mechanical assertions: two reviews timed out and the last exchange exceeded the preferred review context budget. Those failures remain in the table. The current review uses compact source references, explicit serving headroom, and a bounded 150-second deadline for at most two calls; it never drops source passages to fit. A guarded answer is scored for usefulness, not counted as successful simply because it removed a bad claim.

## Follow-ups after the failed full replay

Replaying the three saved long drafts with compact source references completed editing in 46.0, 41.0 and 77.2 seconds without model timeouts. Scene assistance returned in the first case, and the second retained the unknown companion instead of a generic verification notice. The third still emphasized an old bookkeeping question: valid citations did not make it responsive.

A separate scene-writing contract now leaves the direct factual-answer slot empty, while the reviewer receives the current contribution explicitly. A fresh writer-plus-review check on the changed-task context took 62.7 seconds and supplied the NPC’s concern in narration with no stale direct calculation. Private notes remained repetitive. This was a targeted development follow-up, not a new eight-step replay or an independent held-out score. The original failed responses and all intermediate experiments remain in the private review record.

The public [local-stack launcher](local-stack.md) starts the application, model services and speech worker from pinned public releases with private configuration and assets. Its lifecycle tests exercise readiness, failure cleanup, occupied-port refusal and shutdown with real subprocesses. Application correctness, model quality and deployment readiness remain separate checks.

## Natural speech and its limits

Four overlapping chunks cover 95 unique seconds of retained natural conversation. Fresh ASR and diarization ran without a speaker-count prior. The first chunk took 8.70 seconds including loading; the other three took 1.63, 1.62 and 0.42 seconds. These are processing times for this small sample, not general transcription benchmarks.

The assistance replay imports the freshly recognized words and leaves unmapped speaker identities unknown. It checks incomplete speech, banter, proposed versus completed actions, and whether suggestions remain outside observed dialogue. A prior model transcript is retained only for comparison. Without independently corrected words and speaker boundaries, there is no WER, DER or verified identity score. No recording or playback was initiated for this evaluation.

Source recordings, transcripts, private responses, weights and identifying metadata are not included in the repository. The [evaluation commands](evaluation.md) and [review contract](response-grounding.md) are available for running the procedure with your own material.
