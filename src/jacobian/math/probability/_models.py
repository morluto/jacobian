"""Small shared exact-probability validation helpers.

Operation contracts live beside their mathematical owner. This module only
retains neutral rational bounds shared by finite-distribution, Gaussian, and
graph-reliability families.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer

MAX_INPUT_RATIONAL_DIGITS = 128
MAX_RESULT_RATIONAL_DIGITS = 512


def _validation_error(
    message: str,
    code: str = "probability.model_invariant",
) -> PydanticCustomError:
    """Build one model-invariant rejection.

    ``code`` defaults to the module-wide invariant code so existing callers keep
    their contract. Callers whose rejection is distinguishable by a caller or a
    test should pass a specific code rather than relying on the message text.
    """

    return PydanticCustomError(code, message)


def _require_bounded_fraction(
    value: Fraction,
    *,
    max_digits: int,
    label: str,
) -> None:
    if (
        len(format_canonical_integer(abs(value.numerator))) > max_digits
        or len(format_canonical_integer(value.denominator)) > max_digits
    ):
        raise _validation_error(f"{label} exceeds the {max_digits}-digit bound")


def _require_strictly_increasing(
    values: tuple[CanonicalRational, ...],
    *,
    label: str,
) -> tuple[Fraction, ...]:
    fractions = tuple(value.as_fraction() for value in values)
    if any(left >= right for left, right in pairwise(fractions)):
        raise _validation_error(f"{label} must be strictly increasing")
    return fractions


__all__ = [
    "MAX_INPUT_RATIONAL_DIGITS",
    "MAX_RESULT_RATIONAL_DIGITS",
    "_require_bounded_fraction",
    "_require_strictly_increasing",
    "_validation_error",
]
