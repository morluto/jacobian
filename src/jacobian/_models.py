"""Narrow strict-model primitives shared by unrelated wire owners."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from jacobian.canonical import CanonicalizationError

# Containers deeper than this ceiling are left for Pydantic's field validation
# to reject by type rather than recursing the shared preflight helper without
# bound.
_MAX_CONTAINER_DEPTH = 16


def canonicalize_json_containers(value: Any) -> Any:
    """Materialize JSON arrays as immutable canonical containers.

    Pydantic validates ``model_validator(mode="before")`` output with Python
    semantics even when its caller supplied strict JSON.  Public mathematical
    values use tuples for every sequence, so owner-local preflight validators
    must return this projection rather than raw JSON arrays.  Mappings are
    copied so callers retain ownership of their transport payload.

    Traversal is total: self-referential (cyclic) containers raise a
    canonicalization error, and acyclic containers nested deeper than a sane
    canonical ceiling are returned as-is so Pydantic's own field validation
    rejects the adversarial payload by type.  Either way the shared preflight
    helper never recurses into unbounded Python stack.
    """

    return _canonicalize(value, set(), 0)


def _canonicalize(value: Any, seen: set[int], depth: int) -> Any:
    if isinstance(value, (list, tuple)):
        if id(value) in seen:
            raise CanonicalizationError("cyclic JSON containers are not allowed")
        if depth >= _MAX_CONTAINER_DEPTH:
            return value
        seen = seen | {id(value)}
        return tuple(_canonicalize(item, seen, depth + 1) for item in value)
    if isinstance(value, dict):
        if id(value) in seen:
            raise CanonicalizationError("cyclic JSON containers are not allowed")
        if depth >= _MAX_CONTAINER_DEPTH:
            return value
        seen = seen | {id(value)}
        return {
            key: _canonicalize(item, seen, depth + 1) for key, item in value.items()
        }
    return value


class StrictModel(BaseModel):
    """Closed immutable model; Pydantic owns nested and JSON validation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


__all__ = ["StrictModel", "canonicalize_json_containers"]
