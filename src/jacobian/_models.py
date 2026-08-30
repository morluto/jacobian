"""Narrow strict-model primitives shared by unrelated wire owners."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

_MAX_CONTAINER_DEPTH = 256


def canonicalize_json_containers(value: Any) -> Any:
    """Materialize JSON arrays as immutable canonical containers.

    Pydantic validates ``model_validator(mode="before")`` output with Python
    semantics even when its caller supplied strict JSON.  Public mathematical
    values use tuples for every sequence, so owner-local preflight validators
    must return this projection rather than raw JSON arrays.  Mappings are
    copied recursively: callers retain ownership of their transport payload.

    A bounded depth guard converts deeply nested or cyclic raw data into a
    ``CanonicalizationError`` before recursion can exhaust the interpreter
    stack.
    """

    from jacobian.canonical import CanonicalizationError

    def _canonicalize(inner: Any, depth: int) -> Any:
        if depth > _MAX_CONTAINER_DEPTH:
            raise CanonicalizationError(
                f"container nesting exceeds {_MAX_CONTAINER_DEPTH} levels"
            )
        if isinstance(inner, list):
            return tuple(_canonicalize(item, depth + 1) for item in inner)
        if isinstance(inner, tuple):
            return tuple(_canonicalize(item, depth + 1) for item in inner)
        if isinstance(inner, dict):
            return {
                key: _canonicalize(item, depth + 1)
                for key, item in inner.items()
            }
        return inner

    return _canonicalize(value, 0)


class StrictModel(BaseModel):
    """Closed immutable model; Pydantic owns nested and JSON validation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


__all__ = ["StrictModel", "canonicalize_json_containers"]
