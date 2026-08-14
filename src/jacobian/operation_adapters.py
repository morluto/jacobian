"""Strict one-pass parsing for a selected mathematical operation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from jacobian.canonical import (
    CanonicalizationError,
    encode_strict_json,
)


def parse_operation_input[ModelT: BaseModel](
    model: type[ModelT], payload: dict[str, Any]
) -> ModelT:
    """Parse one bounded request into its owning model.

    Canonical JSON is enforced once for size and wire shape, then the selected
    Pydantic model parses it once.  Domain functions receive that model, never
    a repaired dictionary or a second generic validation result.
    """

    try:
        encoded = encode_strict_json(payload)
    except CanonicalizationError as exc:
        raise ValueError("operation request is not valid bounded JSON") from exc
    return model.model_validate_json(encoded, strict=True)


__all__ = ["parse_operation_input"]
