# Learning evidence decisions

The optional `story_copilot.rl_environment` module provides authored practice tasks for a separate decision-policy training experiment. It uses the same read-only character lookup, conversation recall, and campaign-rule retrieval as the running copilot. It does not train on a live workspace or change the default model.

Each episode presents a question, permitted context, and an evidence-call budget. A model can inspect a character, retrieve an earlier exchange, consult a rule, answer with structured facts and citations, or ask for a missing balance or rule. The environment keeps its expected result separate from the model's input. A whole attempt can include three tool calls and a final decision.

Tasks include current and corrected resource counts, affordable and unaffordable actions, missing information, and answers already visible in context. Transfer cases introduce two corrections, another character's later update, or a revised rule cost. These are procedural fixtures with known answers, not representative measurements of real conversations.

The checker verifies the structured conclusion, exact relevant source IDs, and whether those sources were actually seen. A correct guess with unseen citations does not pass. A small discovery reward supports exploration; it is reported separately from task success. Repeated calls, unnecessary calls, invalid actions, and truncated attempts are recorded. Always asking a question or listing every retrieved source cannot earn task success.

The environment shares the production `Decision` actions and adds typed terminal fields for the experiment: `value`, `allowed`, `sources`, and `missing`. Those fields make outcome checks explicit. Free-form rationale and clarification wording still require semantic review. A successful score does not establish narrative quality or justify deploying an adapter.

The companion [Qwen toolkit](https://github.com/russedavid/qwen-ttrpg) contains the GPU runner. Its supervised warm-up and GRPO paths train only a small LoRA, preserve model-generated tokens across tool steps, and mask external evidence out of the learning target. Independent base, supervised-only, and RL evaluations are required before judging improvement. Training data, checkpoints, and review traces belong in a separate private output directory.
