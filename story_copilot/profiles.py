"""Campaign-owned terminology and bounded calculation tools; no built-in rule system."""

from copy import deepcopy
import math
import re

from .store import packed

DEFAULT = {"resource_aliases": {}, "query_aliases": {}, "tools": []}
OPERATIONS = {
    "sum",
    "difference",
    "product",
    "quotient",
    "less",
    "less_equal",
    "equal",
    "greater_equal",
    "greater",
}


def validate_profile(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT):
        raise ValueError(
            "A rule profile contains resource_aliases, query_aliases, and tools."
        )
    if len(packed(value)) > 30_000:
        raise ValueError("Keep the rule profile within 30,000 characters.")
    result = {**deepcopy(DEFAULT), **deepcopy(value)}
    for field in ["resource_aliases", "query_aliases"]:
        aliases = result[field]
        if (
            not isinstance(aliases, dict)
            or len(aliases) > 100
            or any(
                not isinstance(k, str)
                or not isinstance(v, str)
                or not k.strip()
                or not v.strip()
                for k, v in aliases.items()
            )
        ):
            raise ValueError(
                "Aliases map nonempty text labels to nonempty text labels."
            )
    names = set()
    if not isinstance(result["tools"], list) or len(result["tools"]) > 16:
        raise ValueError("Configure up to sixteen bounded calculation tools.")
    for tool in result["tools"]:
        if not isinstance(tool, dict) or set(tool) != {
            "name",
            "description",
            "operation",
        }:
            raise ValueError(
                "Each calculation tool needs name, description, and operation."
            )
        if (
            not isinstance(tool["name"], str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", tool["name"])
            or tool["name"] in names
        ):
            raise ValueError(
                "Calculation tool names must be distinct simple identifiers."
            )
        if (
            tool["operation"] not in OPERATIONS
            or not isinstance(tool["description"], str)
            or not tool["description"].strip()
        ):
            raise ValueError(
                "Choose a supported arithmetic/comparison operation and describe its use."
            )
        names.add(tool["name"])
    return result


def calculate(profile, name, values):
    profile = validate_profile(profile)
    tool = next((t for t in profile["tools"] if t["name"] == name), None)
    if tool is None:
        raise ValueError("The campaign has not enabled this calculation tool.")
    if not 1 <= len(values) <= 12 or any(
        type(v) not in {int, float} or not math.isfinite(v) or abs(v) > 1e12
        for v in values
    ):
        raise ValueError("Use one to twelve finite, bounded numeric inputs.")
    operation = tool["operation"]
    if operation not in {"sum", "product"} and len(values) != 2:
        raise ValueError("This operation requires exactly two inputs in order.")
    a, b = (values + [0, 0])[:2]
    if operation == "sum":
        result = sum(values)
    elif operation == "product":
        result = math.prod(values)
    elif operation == "difference":
        result = a - b
    elif operation == "quotient":
        if b == 0:
            raise ValueError("Division by zero is not a valid rule outcome.")
        result = a / b
    elif operation == "less":
        result = a < b
    elif operation == "less_equal":
        result = a <= b
    elif operation == "equal":
        result = a == b
    elif operation == "greater_equal":
        result = a >= b
    else:
        result = a > b
    if type(result) is not bool and (not math.isfinite(result) or abs(result) > 1e18):
        raise ValueError("The calculation result exceeds the supported range.")
    return {"tool": name, "operation": operation, "inputs": values, "result": result}
