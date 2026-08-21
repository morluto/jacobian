"""Shared passive value types for opaque mathematical labels."""

from __future__ import annotations

import unicodedata
from typing import Annotated

from pydantic import AfterValidator, Field, StringConstraints

MAX_OPAQUE_LABEL_LENGTH = 64


def _require_opaque_label(value: str) -> str:
    if value != value.strip():
        raise ValueError("label must not have leading or trailing whitespace")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError("label must not contain control characters")
    return value


OpaqueLabel = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=MAX_OPAQUE_LABEL_LENGTH,
        strict=True,
    ),
    Field(
        description=(
            "Opaque mathematical label without leading/trailing whitespace or "
            "Unicode control characters."
        )
    ),
    AfterValidator(_require_opaque_label),
]

__all__ = ["MAX_OPAQUE_LABEL_LENGTH", "OpaqueLabel"]
