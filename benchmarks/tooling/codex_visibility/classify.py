"""Transcript/call classification and derived metrics for Codex visibility.

Classification is deterministic over a frozen case plus already parsed
telemetry.  It performs no process or filesystem orchestration.
"""

from __future__ import annotations

from collections.abc import Mapping


def _output_field(output: object, path: str) -> tuple[bool, object]:
    current = output
    for component in path.split("."):
        if isinstance(current, Mapping) and component in current:
            current = current[component]
        else:
            return False, None
    return True, current


def _substantive_output_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _output_outcome_matches(
    outcome,
    output: object,
) -> bool:
    for field_path in outcome.required_output_fields:
        found, value = _output_field(output, field_path)
        if not found:
            return False
    for field_path, expected_value in outcome.expected_output_values.items():
        found, value = _output_field(output, field_path)
        if not found or value != expected_value:
            return False
    return True


# The full classify_visibility function remains in __init__ because it
# imports from multiple internal modules.  This module owns the pure
# classification helpers that classify_visibility delegates to.

__all__ = [
    "_output_field",
    "_output_outcome_matches",
    "_substantive_output_value",
]
