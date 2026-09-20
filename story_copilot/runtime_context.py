"""Local tokenizer and prompt budgets shared by replay and live play."""

from functools import lru_cache
import os
from pathlib import Path


@lru_cache(maxsize=2)
def _tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(Path(path).expanduser().resolve(strict=True)), local_files_only=True
    )


def context_options(store=None):
    path = os.environ.get("STORY_TOKENIZER")
    from .settings import load, DEFAULT

    settings = load(store.home) if store is not None else DEFAULT
    return {
        "context_limit": int(
            os.environ.get("STORY_CONTEXT_TOKENS", settings["context_limit"])
        ),
        "output_reserve": int(
            os.environ.get("STORY_OUTPUT_TOKENS", settings["output_reserve"])
        ),
        "safety_margin": int(os.environ.get("STORY_CONTEXT_MARGIN", "512")),
        "tokenizer": _tokenizer(path) if path else None,
    }


def _with_budget(builder, kwargs):
    from .context import ContextBudgetError

    preferred = kwargs["context_limit"]
    hard = int(os.environ.get("STORY_MAX_CONTEXT_TOKENS", str(preferred)))
    if hard < preferred:
        raise ValueError("Preferred context budget exceeds the serving window.")
    try:
        result = builder(**kwargs)
    except ContextBudgetError as exc:
        if hard == preferred:
            raise
        needed = (
            exc.trace["mandatory_tokens"]
            + kwargs["output_reserve"]
            + kwargs["safety_margin"]
        )
        if needed > hard:
            raise
        expanded = min(hard, needed + 1024)
        result = builder(**{**kwargs, "context_limit": expanded})
        result["trace"]["expanded_for_current_exchange"] = True
    result["trace"].update(preferred_context_limit=preferred, model_context_limit=hard)
    return result


def pack_with_budget(**kwargs):
    from .context import pack_context

    return _with_budget(pack_context, {**context_options(), **kwargs})


def build_with_budget(store, cid, before, **kwargs):
    from .context import build_context

    return _with_budget(
        lambda **options: build_context(store, cid, before, **options),
        {**context_options(store), **kwargs},
    )
