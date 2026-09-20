"""Explicit task-dependent sampling, with the vendor's narrative defaults."""

# https://huggingface.co/Qwen/Qwen3.8-27B#best-practices
NARRATIVE_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
}


def sampling(task, temperature=None, profile="production"):
    if profile not in {"production", "greedy"}:
        raise ValueError("Unknown sampling profile.")
    if profile == "greedy":
        return {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repeat_penalty": 1.0,
        }
    if task == "storyteller":
        settings = NARRATIVE_SAMPLING.copy()
        if temperature is not None:
            settings["temperature"] = temperature
        return settings
    # Exact source quotation and JSON extraction deliberately avoid a presence
    # penalty: repeating an entity or its quoted words can be necessary.
    return {
        "temperature": 0.2 if temperature is None else temperature,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    }
