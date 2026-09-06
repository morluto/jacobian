"""Neutral canonical values shared by exact integer number-theory families.

Operation requests, bounded admissions, and result contracts live beside their
respective kernels.  This module intentionally owns only the canonical integer
grammar and the owner-namespaced validation error constructor.
"""

from __future__ import annotations

from typing import Annotated

from pydantic_core import PydanticCustomError

from jacobian._exact import DecimalIntegerEncoding

MAX_INTEGER_DIGITS = 256

BoundedInteger = Annotated[
    int,
    DecimalIntegerEncoding(max_digits=MAX_INTEGER_DIGITS),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable semantic error owned by the number-theory domain."""

    return PydanticCustomError(f"number_theory.{reason}", message)


__all__ = ["MAX_INTEGER_DIGITS", "BoundedInteger"]
