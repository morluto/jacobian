"""Canonical passive values for labelled integer affine forms."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer

MAX_AFFINE_COMPONENT_DIGITS = 256
MAX_FORM_ID_LENGTH = 32

AffineComponentInteger = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=MAX_AFFINE_COMPONENT_DIGITS + 1, strict=True),
]

AffineFormId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,31}$",
        max_length=MAX_FORM_ID_LENGTH,
        strict=True,
    ),
]


def _component_digits(value: str) -> int:
    return len(value.lstrip("-"))


class IntegerAffineForm(StrictModel):
    """One labelled nonzero integer affine expression ``a*n+b``."""

    form_id: AffineFormId = Field(
        description=(
            "Stable form label using the grammar [A-Za-z][A-Za-z0-9_.-]{0,31}."
        )
    )
    coefficient: AffineComponentInteger = Field(
        description=(
            "Canonical decimal coefficient a in L(n)=a*n+b, with at most 256 "
            "digits excluding an optional minus sign."
        )
    )
    constant: AffineComponentInteger = Field(
        description=(
            "Canonical decimal constant b in L(n)=a*n+b, with at most 256 "
            "digits excluding an optional minus sign."
        )
    )

    @model_validator(mode="after")
    def require_bounded_nonzero_expression(self) -> Self:
        if (
            _component_digits(self.coefficient) > MAX_AFFINE_COMPONENT_DIGITS
            or _component_digits(self.constant) > MAX_AFFINE_COMPONENT_DIGITS
        ):
            raise ValueError(
                "affine coefficient and constant must each have at most "
                f"{MAX_AFFINE_COMPONENT_DIGITS} digits"
            )
        if (
            parse_canonical_integer(self.coefficient) == 0
            and parse_canonical_integer(self.constant) == 0
        ):
            raise ValueError("integer affine form must not be identically zero")
        return self

    def evaluate(self, parameter: int) -> int:
        """Evaluate the form exactly at one integer parameter."""

        return parse_canonical_integer(
            self.coefficient
        ) * parameter + parse_canonical_integer(self.constant)


__all__ = [
    "MAX_AFFINE_COMPONENT_DIGITS",
    "MAX_FORM_ID_LENGTH",
    "AffineComponentInteger",
    "AffineFormId",
    "IntegerAffineForm",
]
