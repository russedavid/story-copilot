"""Build a local serving command. The companion qwen-ttrpg toolkit manages training and serving runs."""

import json
import math
from pathlib import Path


def command(
    runtime,
    base,
    adapters,
    port=8091,
    layout="single",
    slots=1,
    context=16384,
    tensor_split=None,
):
    if layout not in {"single", "split"} or slots not in {1, 2, 3, 4}:
        raise ValueError("Choose single/split layout and one to four slots.")
    if not 1024 <= context <= 262144:
        raise ValueError("Choose a bounded per-request context window.")
    if tensor_split is not None:
        if layout != "split":
            raise ValueError("A tensor split requires the split layout.")
        try:
            values = [float(value) for value in tensor_split.split(",")]
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                "Use two positive finite GPU proportions, such as 4,1."
            ) from exc
        if len(values) != 2 or any(
            not math.isfinite(value) or value <= 0 for value in values
        ):
            raise ValueError("Use two positive finite GPU proportions, such as 4,1.")
    result = [
        str(Path(runtime) / "build/bin/llama-server"),
        "-m",
        str(base),
        "--alias",
        "local-model",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context * slots),
        "--parallel",
        str(slots),
        "--n-gpu-layers",
        "all",
        "--flash-attn",
        "on",
        "--jinja",
        "--no-context-shift",
        "--chat-template-kwargs",
        json.dumps({"enable_thinking": False}),
    ]
    if layout == "split":
        result += ["--split-mode", "layer", "--tensor-split", tensor_split or "1,1"]
    else:
        result += ["--split-mode", "none", "--main-gpu", "0"]
    if adapters:
        if any(any(c in str(p) for c in [",", ":"]) for p in adapters):
            raise ValueError("Adapter filenames may not contain commas or colons.")
        result += ["--lora-scaled", ",".join(str(p) + ":0.0" for p in adapters)]
    return result
