# Story Copilot

A private assistant for running an interactive story. It follows the conversation, keeps track of what changed, and helps the human facilitator decide what to say next.

Players still choose their actions. The facilitator still runs the game. The copilot can recall a previous exchange, inspect a character's knowledge, consult rules you provide, or ask for a missing detail before suggesting a response. Its suggestions never become part of the conversation or the world state just because they were generated. A bounded editing pass checks the draft for player agency, knowledge, and continuity problems, retaining the original for review.

## Start locally

Python 3.10 or newer and a running model server are required. A GPU is not needed on the computer running the web UI; the model server can run on a separate machine.

```sh
git clone https://github.com/russedavid/story-copilot.git
cd story-copilot
python -m venv .venv
source .venv/bin/activate
pip install -e .
story-copilot serve
```

Open **http://127.0.0.1:5022**. Select **Model settings**, enter the server's API base URL and model identifier, and save. The default backend is llama.cpp. A compatible Chat Completions server must support JSON Schema responses; native provider APIs are not automatically interchangeable.

If your server requires authentication, set a key in the app's launch environment and enter the **environment variable name** in Model settings. Keys are not stored in the UI configuration or generation traces. A remote model endpoint receives the private context included in its requests.

For a local llama.cpp server, use an alias matching the model identifier in settings. Configure the app's context limit to fit the server's per-request window. Optional LoRAs are selected per request, sharing a single base model. List every loaded adapter in the routing inventory; all unselected adapters receive an explicit zero scale.

## Try a conversation

1. Choose **Try an example** to open *The Unsent Signal*, an original fictional scenario with two characters. This is authored sample content, not a transcript or a benchmark answer.
2. Select **Suggest now**. Open the generation trace to see the evidence decisions, retrieved sources, validation results, model output, and timing.
3. Add a player contribution or the facilitator's actual reply to the conversation. Suggested prose stays private; copy it if useful, or reject and refresh it for another possibility.
4. Correct a source message when someone changes or clarifies what they said. The affected observations are rebuilt and incompatible in-flight answers are discarded.
5. Use **Continue** for the next session, **Branch** to explore an alternative, or **Fresh** for a new story with the same campaign material.

For your own campaign, add a direction, scenario documents, character sheets, and speaker mappings. Mark rule documents as **Rules for this campaign**. Character sheets accept your own JSON fields; integer resource totals belong under `resources`, with `null` for unknown totals. A campaign can define terminology aliases and bounded arithmetic tools, without requiring a particular game system.

Automatic guidance is optional. Starting the app or opening the example does not capture audio or send model requests. Enable automatic suggestions when ready; unchanged context does not repeatedly generate the same response.

## Conversation, audio, and privacy

Type or paste contributions directly, import your own transcripts into the source library, or use the optional local audio pipeline. Microphone and system audio remain separate sources, and speaker identity can be corrected. [Audio setup](docs/audio.md) describes the additional dependencies and worker process.

Data lives outside the repository in `~/.local/share/story-copilot`. Use `story-copilot --home /path/to/new/workspace serve` for a separate workspace. The app refuses to adopt an existing unidentified database. Keep this localhost service behind an SSH tunnel when using it remotely; it is a single-user application, not an authenticated multiplayer host.

Private notes, suggestions, and character knowledge are kept separate from the player-visible source view. That view is a convenience for the facilitator, not an access-control boundary for remote users. Models can still make mistakes: exact quotation, schema checks, and deterministic calculation validate specific properties, not the truth of an interpretation or the quality of a story.

## Development and evaluation

```sh
pip install -e '.[dev]'
python -m pytest -q
```

The test suite exercises source corrections, dialogue ordering, character knowledge, session continuity, context budgets, audio queues, stale answers, duplicate suppression, rule isolation, and bounded tool use. Scripted model tests verify application behavior; live model evaluation is separate. See [architecture](docs/architecture.md), [evaluation](docs/evaluation.md), and the [recorded development findings](docs/validation.md).

The companion [Qwen adapter toolkit](https://github.com/russedavid/qwen-ttrpg) trains and evaluates task adapters from data you supply. [Conversational Dataset Formatter](https://github.com/russedavid/format_conversation_dataset) prepares reviewed conversational targets with completion-only loss masks. Those projects manage data preparation and model training; this repository contains the interactive application. No training data or model weights are included.
