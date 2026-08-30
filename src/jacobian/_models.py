"""Narrow strict-model primitives shared by unrelated wire owners."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


def canonicalize_json_containers(value: Any) -> Any:
    """Materialize JSON arrays as immutable canonical containers.

    Pydantic validates ``model_validator(mode="before")`` output with Python
    semantics even when its caller supplied strict JSON.  Public mathematical
    values use tuples for every sequence, so owner-local preflight validators
    must return this projection rather than raw JSON arrays.  Mappings are
    copied recursively: callers retain ownership of their transport payload.
    """

    if isinstance(value, list):
        return tuple(canonicalize_json_containers(item) for item in value)
    if isinstance(value, tuple):
        return tuple(canonicalize_json_containers(item) for item in value)
    if isinstance(value, dict):
        return {key: canonicalize_json_containers(item) for key, item in value.items()}
    return value


class StrictModel(BaseModel):
    """Closed immutable model; Pydantic owns nested and JSON validation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


__all__ = ["StrictModel", "canonicalize_json_containers"]
