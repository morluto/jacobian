"""Narrow strict-model primitives shared by unrelated wire owners."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from jacobian.canonical import CanonicalizationError


def canonicalize_json_containers(value: Any) -> Any:
    """Materialize JSON arrays as immutable canonical containers.

    Pydantic validates ``model_validator(mode="before")`` output with Python
    semantics even when its caller supplied strict JSON.  Public mathematical
    values use tuples for every sequence, so owner-local preflight validators
    must return this projection rather than raw JSON arrays.  Mappings are
    copied recursively: callers retain ownership of their transport payload.

    Self-referential containers are rejected as a canonicalization error
    rather than recursing until the Python stack limit.
    """

    return _canonicalize(value, set())


def _canonicalize(value: Any, seen: set[int]) -> Any:
    if isinstance(value, (list, tuple)):
        if id(value) in seen:
            raise CanonicalizationError("cyclic JSON containers are not allowed")
        seen = seen | {id(value)}
        return tuple(_canonicalize(item, seen) for item in value)
    if isinstance(value, dict):
        if id(value) in seen:
            raise CanonicalizationError("cyclic JSON containers are not allowed")
        seen = seen | {id(value)}
        return {key: _canonicalize(item, seen) for key, item in value.items()}
    return value


class StrictModel(BaseModel):
    """Closed immutable model; Pydantic owns nested and JSON validation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


__all__ = ["StrictModel", "canonicalize_json_containers"]
