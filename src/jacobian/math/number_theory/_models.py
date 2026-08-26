"""Neutral canonical values shared by exact integer number-theory families.

Operation requests, bounded admissions, and result contracts live beside their
respective kernels.  This module intentionally owns only the canonical integer
grammar and the owner-namespaced validation error constructor.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import StringConstraints
from pydantic_core import PydanticCustomError

MAX_INTEGER_DIGITS = 256
# Private compatibility spelling for the neutral grammar width.  Family
# contracts must not add their own bounds here.
_MAX_INTEGER_LENGTH = MAX_INTEGER_DIGITS

BoundedInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^-?(?:0|[1-9][0-9]*)$",
        max_length=MAX_INTEGER_DIGITS,
        strict=True,
    ),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable semantic error owned by the number-theory domain."""

    return PydanticCustomError(f"number_theory.{reason}", message)


__all__ = ["MAX_INTEGER_DIGITS", "BoundedInteger"]
