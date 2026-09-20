# Player agents

Player identities are separate from characters. Each identity has a personality prompt and an optional LoRA adapter; an assignment connects it to a character in a campaign. Several player adapters can share the same model server and base weights.

Open **Player agents**, create an identity, then use **Assign or pause players** in a session. Choose the character and request **Take a turn** when that participant should respond. A profile with no adapter uses the base model and its personality prompt. Selecting a slot uses that loaded adapter for the final utterance; the read-only evidence planner uses the base.

A requested turn is an AI participant's contribution to the conversation. Its label, model, adapter, source snapshot, tool decisions, and timing remain inspectable. Facilitator suggestions continue to be private drafts. An AI player's words can establish a proposed action or reported claim; they cannot establish a world outcome, resource change, or resolved check without independent facilitator evidence.

## Information each player receives

The application builds the permitted view before retrieval or generation. It includes the assigned starting sheet, shared reference material, shared conversation, messages addressed to that character, and that character's own private speech. It excludes facilitator notes, other character sheets, transport paths, run traces, and the facilitator's inferred state. That last exclusion matters: an inference made with privileged context can contain information the player never received.

Use **Who hears this contribution?** for shared statements, facilitator-only material, or a character-only whisper. Correcting an audience creates a source revision, and continuations preserve the effective recipient. Unlabelled speech cannot reliably establish a private audience; shared audio/text is available to all players. Keep facilitator-only speech private.

**View this character’s perspective** shows the actual permitted source material. Its sheet is a starting point; later changes remain in the visible conversation. Context packing and recall operate only on that permitted history. A player can inspect its sheet, retrieve a previous exchange, look up a shared rule, or speak. Missing information can become a question for the facilitator.

## Turns, corrections, and cancellation

Each assignment can have one queued or running request. A second click returns the existing request, and another turn needs new visible context. The final source check and message insertion share a transaction. If visible evidence, the character, or the identity changes during generation, the obsolete response is retained as a stale trace and is not posted.

Cancel prevents a queued or running turn from being posted and stops further model stages. A model request already in flight may finish before its result is discarded. Plans and prompts are never added to conversation as speech.

## Use a trained personality adapter

Prepare a separate, reviewed dataset for each identity with the companion toolkit's player-data workflow, then train, verify, convert, and load those adapters alongside the common base. See [Qwen TTRPG](https://github.com/russedavid/qwen-ttrpg). In **Model settings**, include every loaded server adapter in order. Preserve the server paths from its routing manifest when available; the client verifies them before inference. Select the appropriate slot on each player identity.

Adapters do not grant additional access. Even an adapter that produces a valid sentence can make a poor choice, confuse a reference, or misread a rule. Evaluate its behavior on held-out conversations and new scenes as well as its reference loss.
