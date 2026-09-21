# A separate evidence planner

The live copilot can use a small learned policy for resource and numerical-rule questions: read a character sheet, recall earlier dialogue, search supplied rules, ask a question, or answer from sufficient evidence. General narration, social choices, and private-knowledge questions retain the main planner because they were outside the small policy’s training curriculum. The main model still classifies observations, writes suggestions, and independently reviews drafts. This is optional: the default installation uses one configured server.

In **Model settings → Dedicated decision planner**, enable the separate endpoint and supply its URL, model alias, context size, and adapter inventory. For example, a server with one learned policy at adapter ID zero uses:

```json
{"adapters": [{"id": 0}], "tasks": {"planner": 0}}
```

List all loaded adapters, even when only one is selected. A server with no adapters uses empty `adapters` and `tasks`. Credentials are environment-variable names, just as for the main server. The model must support the structured decision schema; an arbitrary small language model is not necessarily a useful planner.

The [training toolkit](https://github.com/russedavid/qwen-ttrpg) can export the trained policy and serve it with its saved training chat template. Keep the small model on a separate endpoint; it must never be interpreted as an adapter for a differently sized base. Match the configured context to the server, up to the live bridge's 8,192-token ceiling. An oversized source context, unavailable endpoint, or invalid decision falls back to the main model within the same bounded decision loop. Traces record the route and reason. No fallback silently drops the source context.

Training success on authored numerical tasks does not establish narrative quality. The bridge adds live task categories and preserves the evidence tools, while treating the policy's terminal numbers as unverified proposals. The rules component must ground its inputs in supplied sources; deterministic tools compute the displayed result. One repair attempt is allowed, and both attempts remain in the trace. Failed verification produces a clarification rather than an unchecked numerical ruling.

## Correcting rules

Add a replacement document through **Scenario and reference material → Replaces earlier material**. Choose the same material type. The old document remains visible as superseded history, while current retrieval uses the replacement. Each document has one active replacement chain. A replacement's visibility does not grant a player access to private material.

Basic arithmetic is available without defining a game system. The resource-spending calculator applies only when the cited rule requires sufficient resources and spends nothing if the action is unaffordable; it is not a general rule about debt, partial costs, or outcomes. Calculations never apply resource changes. Only supported observations from the actual conversation can do that.

Spoken announcements do not silently rewrite the supplied rule library. Update the document explicitly when the table adopts a revised rule.

The learned policy uses a JSON-object grammar followed by full schema validation. A property-ordered schema grammar can force a different serialization from training and prevent trailing citation fields. The live bridge accepts the original evidence contract, derives a missing intent label only from typed decision fields, and never treats those terminal values as verified answers. The serving check in `story_copilot.policy_serving_eval` exercises complete tool trajectories after export or quantization.
