"""Typed characteristic-two profiles of integral quadratic forms."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.number_theory.quadratic_forms.integral._models import (
    IntegralQuadraticForm,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular._models import (
    ModularQuadraticPolynomial,
)


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"quadratic_form.parity_profile.{reason}", message)


class ParityProfileRequest(StrictModel):
    """Compute the mod-two coefficient profile of an integral form."""

    form: IntegralQuadraticForm


class ParityProfile(StrictModel):
    """The coefficient reduction to ``Z/2Z`` and basis-vector norm parity."""

    polynomial: ModularQuadraticPolynomial
    all_basis_norms_even: bool

    @model_validator(mode="after")
    def require_mod_two_profile(self) -> Self:
        if self.polynomial.modulus != 2:
            raise _error("modulus", "a parity profile must have modulus two")
        if self.all_basis_norms_even != all(
            residue == 0 for residue in self.polynomial.diagonal_residues
        ):
            raise _error(
                "basis_norms",
                "all_basis_norms_even must agree with the diagonal Q(e_i) residues",
            )
        return self


__all__ = ["ParityProfile", "ParityProfileRequest"]
