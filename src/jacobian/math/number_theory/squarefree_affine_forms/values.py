"""Canonical values for bounded square-free affine-form families."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel
from jacobian.math.number_theory.affine_forms.values import (
    AffineFormId,
    IntegerAffineForm,
)

MAX_SQUAREFREE_FORMS = 8
MAX_SQUAREFREE_COMPONENT_DIGITS = 8

SquarefreeComponentInteger = Annotated[
    int,
    DecimalIntegerEncoding(max_digits=MAX_SQUAREFREE_COMPONENT_DIGITS),
]


def _validation_error(message: str) -> PydanticCustomError:
    """Return an actionable owner-local validation error."""

    code_by_reason = (
        ("IDs must", "form_id_unique"),
        ("pairwise distinct", "form_duplicate"),
        ("identically zero", "form_zero"),
        ("digits", "component_digit_bound"),
        ("forms must", "family_bound"),
    )
    suffix = next(
        (suffix for phrase, suffix in code_by_reason if phrase in message),
        "family_invariant",
    )
    return PydanticCustomError(f"number_theory.squarefree_affine.{suffix}", message)


def _component_digits(value: int) -> int:
    return len(str(abs(value)))


class SquarefreeAffineForm(IntegerAffineForm):
    """One labelled integer form ``a*n+b`` with bounded exact components.

    Constant forms with ``a=0`` and ``b != 0`` are admitted; they exclude
    either no residue or every residue modulo ``p^2``.
    """

    coefficient: SquarefreeComponentInteger = Field(
        description=(
            "Canonical decimal coefficient a in L(n)=a*n+b, with at most "
            f"{MAX_SQUAREFREE_COMPONENT_DIGITS} digits excluding an optional "
            "minus sign. Zero is admitted for constant forms."
        )
    )
    constant: SquarefreeComponentInteger = Field(
        description=(
            "Canonical decimal constant b in L(n)=a*n+b, with at most "
            f"{MAX_SQUAREFREE_COMPONENT_DIGITS} digits excluding an optional "
            "minus sign."
        )
    )

    @model_validator(mode="after")
    def require_bounded_components(self) -> Self:
        if (
            _component_digits(self.coefficient) > MAX_SQUAREFREE_COMPONENT_DIGITS
            or _component_digits(self.constant) > MAX_SQUAREFREE_COMPONENT_DIGITS
        ):
            raise _validation_error(
                "affine coefficient and constant must each have at most "
                f"{MAX_SQUAREFREE_COMPONENT_DIGITS} digits"
            )
        return self


class SquarefreeAffineFamily(StrictModel):
    """A canonical finite set of distinct bounded integer affine forms."""

    forms: tuple[SquarefreeAffineForm, ...] = Field(
        min_length=1,
        max_length=MAX_SQUAREFREE_FORMS,
        description=(
            "Nonempty family of at most "
            f"{MAX_SQUAREFREE_FORMS} distinct labelled forms L_j(n)=a_j*n+b_j "
            "with (a_j,b_j) != (0,0), unique form IDs, and unique coefficient "
            "pairs. Rows are normalized by form_id; row order is not "
            "mathematical."
        ),
    )

    @model_validator(mode="after")
    def require_distinct_canonical_forms(self) -> Self:
        form_ids = tuple(form.form_id for form in self.forms)
        if len(set(form_ids)) != len(form_ids):
            raise _validation_error("affine form IDs must be unique")
        coefficient_pairs = tuple(
            (form.coefficient, form.constant) for form in self.forms
        )
        if len(set(coefficient_pairs)) != len(coefficient_pairs):
            raise _validation_error("affine forms must be pairwise distinct")
        if any(form.coefficient == 0 and form.constant == 0 for form in self.forms):
            raise _validation_error("affine form must not be identically zero")
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
    "MAX_SQUAREFREE_COMPONENT_DIGITS",
    "MAX_SQUAREFREE_FORMS",
    "AffineFormId",
    "SquarefreeAffineFamily",
    "SquarefreeAffineForm",
]
