"""Canonical values for finite families of primitive integer affine forms."""

from __future__ import annotations

from math import gcd
from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.affine_forms.values import IntegerAffineForm

MAX_AFFINE_FORMS = 512
MAX_AFFINE_AGGREGATE_DIGITS = 131_072


def _component_digits(value: str) -> int:
    return len(value.lstrip("-"))


class PrimitiveIntegerAffineForm(IntegerAffineForm):
    """One labelled primitive nonconstant form ``a*n+b`` over the integers."""

    @model_validator(mode="after")
    def require_primitive_nonconstant_form(self) -> Self:
        coefficient = parse_canonical_integer(self.coefficient)
        constant = parse_canonical_integer(self.constant)
        if coefficient == 0:
            raise ValueError("primitive affine form coefficient must be nonzero")
        if gcd(abs(coefficient), abs(constant)) != 1:
            raise ValueError("affine coefficient and constant must be coprime")
        return self


class PrimeAffineTuple(StrictModel):
    """A canonical finite set of distinct primitive integer affine forms."""

    forms: tuple[PrimitiveIntegerAffineForm, ...] = Field(
        min_length=1,
        max_length=MAX_AFFINE_FORMS,
        description=(
            "Nonempty primitive affine-form family. Form IDs and coefficient/constant "
            "pairs must be unique; rows are normalized by form_id. Across all forms, "
            "coefficient and constant magnitudes contain at most 131072 decimal digits."
        ),
    )

    @model_validator(mode="after")
    def require_distinct_canonical_forms(self) -> Self:
        if (
            sum(
                _component_digits(form.coefficient) + _component_digits(form.constant)
                for form in self.forms
            )
            > MAX_AFFINE_AGGREGATE_DIGITS
        ):
            raise ValueError(
                "affine tuple exceeds the aggregate coefficient-digit bound "
                f"{MAX_AFFINE_AGGREGATE_DIGITS}"
            )
        form_ids = tuple(form.form_id for form in self.forms)
        if len(set(form_ids)) != len(form_ids):
            raise ValueError("affine form IDs must be unique")
        coefficient_pairs = tuple(
            (form.coefficient, form.constant) for form in self.forms
        )
        if len(set(coefficient_pairs)) != len(coefficient_pairs):
            raise ValueError("affine forms must be pairwise distinct")
        object.__setattr__(
            self,
            "forms",
            tuple(sorted(self.forms, key=lambda form: form.form_id)),
        )
        return self

    @property
    def form_count(self) -> int:
        return len(self.forms)


__all__ = [
    "MAX_AFFINE_AGGREGATE_DIGITS",
    "MAX_AFFINE_FORMS",
    "PrimeAffineTuple",
    "PrimitiveIntegerAffineForm",
]
