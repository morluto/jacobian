"""Contracts for explicitly supplied finite polynomial Ga-subrepresentations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.derivations._models import PolynomialGaAction
from jacobian.math.polynomials.values import RationalPolynomial

MAX_GA_SUBREPRESENTATION_DIMENSION = 32


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial_ga_subrepresentation.{reason}", message)


class PolynomialGaStableSubrepresentationRequest(StrictModel):
    """An action and a linearly independent, ordered proposed subspace basis."""

    action: PolynomialGaAction
    basis: Annotated[
        tuple[RationalPolynomial, ...],
        Field(min_length=1, max_length=MAX_GA_SUBREPRESENTATION_DIMENSION),
    ]

    @model_validator(mode="after")
    def require_shared_ring(self) -> Self:
        if any(value.variables != self.action.source_variables for value in self.basis):
            raise _error(
                "ordered_ring", "basis polynomials must use the action source ring"
            )
        return self


class PolynomialGaStableSubrepresentation(StrictModel):
    """The supplied basis and its exact polynomial-parameter action matrix.

    Matrix rows index output basis vectors and columns index input basis vectors;
    entries are in QQ[t]. Basis order is structural and survives wire roundtrips.
    """

    action: PolynomialGaAction
    basis: tuple[RationalPolynomial, ...]
    action_matrix: tuple[tuple[RationalPolynomial, ...], ...]

    @model_validator(mode="after")
    def require_axis_shape(self) -> Self:
        size = len(self.basis)
        if not 1 <= size <= MAX_GA_SUBREPRESENTATION_DIMENSION:
            raise _error("dimension", "basis dimension exceeds the admitted bound")
        if any(value.variables != self.action.source_variables for value in self.basis):
            raise _error(
                "ordered_ring", "basis polynomials must use the action source ring"
            )
        if len(self.action_matrix) != size or any(
            len(row) != size for row in self.action_matrix
        ):
            raise _error(
                "matrix_shape", "action matrix must be square on the ordered basis"
            )
        if any(
            entry.variables != (self.action.parameter,)
            for row in self.action_matrix
            for entry in row
        ):
            raise _error(
                "parameter_axis", "action matrix entries must use the parameter axis"
            )
        return self


class PolynomialGaFixedSubspaceRequest(StrictModel):
    """A serialized finite Ga-subrepresentation whose fixed vectors are wanted."""

    subrepresentation: PolynomialGaStableSubrepresentation


class PolynomialGaFixedSubspace(StrictModel):
    """A canonical basis of fixed vectors in the supplied subrepresentation.

    Each coordinate row and corresponding polynomial are one basis vector.
    The coordinate axis is the original ordered subrepresentation basis.
    """

    subrepresentation: PolynomialGaStableSubrepresentation
    coordinates: Annotated[
        tuple[tuple[CanonicalRational, ...], ...],
        Field(max_length=MAX_GA_SUBREPRESENTATION_DIMENSION),
    ]
    basis: Annotated[
        tuple[RationalPolynomial, ...],
        Field(max_length=MAX_GA_SUBREPRESENTATION_DIMENSION),
    ]

    @model_validator(mode="after")
    def require_axis_shape(self) -> Self:
        size = len(self.subrepresentation.basis)
        if len(self.coordinates) != len(self.basis):
            raise _error("basis_shape", "coordinates and fixed basis must align")
        if len(self.basis) > size or any(len(row) != size for row in self.coordinates):
            raise _error(
                "basis_shape", "fixed coordinates must use the supplied basis axis"
            )
        if any(
            value.variables != self.subrepresentation.action.source_variables
            for value in self.basis
        ):
            raise _error(
                "ordered_ring", "fixed basis must retain the action source ring"
            )
        return self
