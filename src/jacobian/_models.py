"""Narrow strict-model primitives shared by unrelated wire owners."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from jacobian.canonical import CanonicalizationError, CanonicalLimits


def canonicalize_json_containers(value: Any) -> Any:
    """Materialize JSON arrays as immutable canonical containers.

    Pydantic validates ``model_validator(mode="before")`` output with Python
    semantics even when its caller supplied strict JSON.  Public mathematical
    values use tuples for every sequence, so owner-local preflight validators
    must return this projection rather than raw JSON arrays.  Mappings are
    copied recursively: callers retain ownership of their transport payload.
    """

    active_containers: set[int] = set()
    maximum_depth = CanonicalLimits().max_depth

    def project(item: Any, *, depth: int) -> Any:
        if depth > maximum_depth:
            raise CanonicalizationError(
                "JSON nesting exceeds the configured depth limit"
            )
        if not isinstance(item, (list, tuple, dict)):
            return item

        identity = id(item)
        if identity in active_containers:
            raise CanonicalizationError("recursive JSON containers are not supported")
        active_containers.add(identity)
        try:
            if isinstance(item, dict):
                return {
                    key: project(nested, depth=depth + 1)
                    for key, nested in item.items()
                }
            return tuple(project(nested, depth=depth + 1) for nested in item)
        finally:
            active_containers.remove(identity)

    return project(value, depth=0)


class StrictModel(BaseModel):
    """Closed immutable model; Pydantic owns nested and JSON validation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


__all__ = ["StrictModel", "canonicalize_json_containers"]
