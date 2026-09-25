"""Typed request and result for complete modular quadratic fibers."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.number_theory.quadratic_forms.integral.modular._models import (
    ModularCoordinateVector,
    ModularInteger,
    ModularQuadraticPolynomial,
)

MAX_MODULAR_FIBER_OUTPUT_VECTORS = 100_000
MAX_MODULAR_FIBER_OUTPUT_DIGITS = 4_000_000
MAX_MODULAR_FIBER_WORK = 20_000_000


def _fiber_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"quadratic_form.modular.fiber.{reason}", message)


class ModularQuadraticFiberRequest(StrictModel):
    """Enumerate every vector mapping to one target residue."""

    polynomial: ModularQuadraticPolynomial
    target: ModularInteger

    @model_validator(mode="after")
    def require_one_residue_parent(self) -> Self:
        if self.target.modulus != self.polynomial.modulus:
            raise _fiber_error(
                "parent_mismatch",
                "polynomial and target residue moduli must agree",
            )
        return self


class ModularQuadraticFiber(StrictModel):
    """One complete finite fiber, sorted in its declared coordinate axis."""

    polynomial: ModularQuadraticPolynomial
    target: ModularInteger
    vectors: tuple[ModularCoordinateVector, ...] = Field(
        max_length=MAX_MODULAR_FIBER_OUTPUT_VECTORS
    )

    @model_validator(mode="after")
    def require_parent_axis_and_canonical_order(self) -> Self:
        if self.target.modulus != self.polynomial.modulus:
            raise _fiber_error(
                "parent_mismatch",
                "polynomial and target residue moduli must agree",
            )
        modulus_digits = len(str(self.polynomial.modulus))
        axis_chars = sum(len(label) for label in self.polynomial.axis)
        support = sum(value != 0 for value in self.polynomial.diagonal_residues) + len(
            self.polynomial.cross_terms
        )
        work = len(self.vectors) * (len(self.polynomial.axis) + support + 1)
        if work > MAX_MODULAR_FIBER_WORK:
            raise _fiber_error(
                "result_work_bound",
                "fiber membership checks exceed their operation bound",
            )
        coefficient_digits = sum(
            len(str(value)) for value in self.polynomial.diagonal_residues
        ) + sum(len(str(term.coefficient)) for term in self.polynomial.cross_terms)
        result_digits = (
            3 * modulus_digits
            + axis_chars
            + coefficient_digits
            + 16
            * (
                len(self.polynomial.diagonal_residues)
                + len(self.polynomial.cross_terms)
            )
            + 32
            + len(self.vectors)
            * (
                modulus_digits
                + axis_chars
                + len(self.polynomial.axis) * (modulus_digits + 12)
                + 32
            )
        )
        if result_digits > MAX_MODULAR_FIBER_OUTPUT_DIGITS:
            raise _fiber_error(
                "result_output_bound", "complete fiber exceeds its result-value bound"
            )

        previous: tuple[int, ...] | None = None
        for vector in self.vectors:
            if (
                vector.modulus != self.polynomial.modulus
                or vector.axis != self.polynomial.axis
                or len(vector.coordinates) != len(self.polynomial.axis)
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 0 <= value < self.polynomial.modulus
                    for value in vector.coordinates
                )
            ):
                raise _fiber_error(
                    "vector_parent",
                    "every fiber vector must have the polynomial modulus and ordered axis",
                )
            if previous is not None and previous >= vector.coordinates:
                raise _fiber_error(
                    "vector_order",
                    "fiber vectors must be unique and lexicographically ordered",
                )
            previous = vector.coordinates
        return self


__all__ = [
    "MAX_MODULAR_FIBER_OUTPUT_DIGITS",
    "MAX_MODULAR_FIBER_OUTPUT_VECTORS",
    "MAX_MODULAR_FIBER_WORK",
    "ModularQuadraticFiber",
    "ModularQuadraticFiberRequest",
]
