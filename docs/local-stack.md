# Run the complete local stack

The GPU host can run the public application, model servers and speech worker with a private configuration. Source code comes from pinned checkouts of this repository and [Qwen TTRPG](https://github.com/russedavid/qwen-ttrpg). Campaign data, model files, LoRAs, caches, credentials and generated logs stay outside both checkouts.

Install the application and Qwen toolkit into the application environment. Install the speech dependencies in a separate environment if their requirements differ. Build the documented llama.cpp version separately. Check out the exact revisions you intend to run; changing a checkout under a running process is not an upgrade procedure.

Start the configured services in the foreground:

```sh
python -m story_copilot.local_stack --config /private/local-stack.json
```

The launcher starts services in order, waits for each readiness check, and stops its own process groups if a service exits. Ctrl-C or SIGTERM shuts down the stack. It refuses configured ports that are already occupied and does not stop another process to free them. Each start creates a new private run directory, with service logs and `status.json`; `current.json` under the run root identifies the latest run and supervisor PID. It does not start microphone or system-audio capture.

A private configuration has this shape (replace the paths and model alias):

```json
{
  "run_root": "/srv/story/runtime",
  "services": [
    {
      "name": "model",
      "revision": "PINNED_QWEN_COMMIT",
      "cwd": "/opt/story/repos/qwen-ttrpg",
      "argv": [
        "/opt/story/env/bin/python", "-m", "qwen_ttrpg.serve_models",
        "--runtime", "/opt/story/llama.cpp",
        "--base", "/srv/story/models/base.gguf",
        "--model-alias", "local-story-model",
        "--layout", "single", "--gpu", "0",
        "--context", "32768", "--port", "8091",
        "--output", "{run}/model"
      ],
      "health_url": "http://127.0.0.1:8091/health"
    },
    {
      "name": "speech",
      "revision": "PINNED_APPLICATION_COMMIT",
      "cwd": "/opt/story/repos/story-copilot",
      "argv": [
        "/opt/story/asr-env/bin/python", "-m", "story_copilot.audio_worker",
        "--queue", "/srv/story/workspace/live-audio",
        "--device-index", "1", "--preload"
      ],
      "ready_text": "\"status\": \"ready\""
    },
    {
      "name": "app",
      "revision": "PINNED_APPLICATION_COMMIT",
      "cwd": "/opt/story/repos/story-copilot",
      "argv": [
        "/opt/story/env/bin/python", "-m", "story_copilot.cli",
        "--home", "/srv/story/workspace", "serve", "--port", "5022"
      ],
      "env": {
        "STORY_TOKENIZER": "/srv/story/models/tokenizer",
        "STORY_MAX_CONTEXT_TOKENS": "32768",
        "STORY_MODEL_METADATA": "{run}/model/run.json"
      },
      "health_url": "http://127.0.0.1:5022/"
    }
  ]
}
```

The example serves a base model. Add `--adapter TASK=/path/to/adapter.gguf` or `--candidate NAME=/path/to/adapter.gguf` to the model command for your trained adapters. Preserve their complete ordering, and configure **Model settings** with the actual model alias, endpoint, context limit and adapter inventory. The launcher leaves those workspace settings intact. A loaded candidate is not automatically selected. Speech recognition also needs access to its downloaded model files; use your environment or model cache for any required authentication.

An optional service named `planner` uses the same public `qwen_ttrpg.serve_models` module, a separate port, and the saved policy chat template. Put it before speech and app, and configure the application's separate planner endpoint. Do not assume it fits beside ASR merely because the writing model fits on the other GPU; validate memory with actual concurrent requests. The measured deployment uses a quantized 4B policy and speech on one GPU, and a quantized 27B writing model with small adapters on the other.

The launcher uses Linux/macOS process groups and file locking. It is not a container orchestrator or a multi-user network service. Run it under your process supervisor if it should survive logout. Keep the UI and model endpoints on loopback, and use an SSH tunnel from your own computer.

For an upgrade, finish or cancel active work, stop the owned stack, install the new pinned releases, update the private configuration's paths/revisions, and start it again. Verify a real request and adapter selection, not just HTTP readiness. A rollback selects the prior public revisions with the same private workspace and assets, subject to any documented database migration requirements.
