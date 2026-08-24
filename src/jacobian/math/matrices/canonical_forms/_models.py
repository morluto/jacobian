"""Typed wire contracts for exact canonical-form operations over QQ."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.matrices.values import RationalMatrix, require_matrix_scalar_digits

MAX_CANONICAL_FORM_DIMENSION = 16
MAX_CANONICAL_FORM_SCALAR_DIGITS = 256


class SquareMatrixRequest(StrictModel):
    """One square rational matrix bounded for canonical-form computation."""

    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_bounded_square(self) -> Self:
        rows = len(self.matrix.entries)
        columns = len(self.matrix.entries[0])
        if rows != columns:
            raise ValueError("canonical-form operations require a square matrix")
        if rows > MAX_CANONICAL_FORM_DIMENSION:
            raise ValueError(
                "canonical-form operations are bounded to 16 x 16 matrices"
            )
        require_matrix_scalar_digits(
            self.matrix.entries,
            maximum=MAX_CANONICAL_FORM_SCALAR_DIGITS,
            label="canonical-form matrix",
        )
        return self


class MonicPolynomial(StrictModel):
    """One monic univariate polynomial over QQ, as increasing-degree coefficients."""

    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_CANONICAL_FORM_DIMENSION + 1,
    )

    @model_validator(mode="after")
    def require_monic(self) -> Self:
        if self.coefficients[-1].as_fraction() != 1:
            raise ValueError("polynomial must be monic (leading coefficient = 1)")
        return self


class MinimalPolynomialResult(StrictModel):
    """Exact minimal polynomial of a square rational matrix.

    Retains the source matrix so validation replays the defining relations:
    the claimed polynomial is monic of the claimed degree, annihilates the
    retained matrix, and equals the exact minimal polynomial re-derived
    from that matrix; the characteristic polynomial must be the matrix's.
    Annihilation alone never upgrades a polynomial to minimality.
    """

    matrix: SquareMatrixRequest
    minimal_polynomial: MonicPolynomial
    characteristic_polynomial: MonicPolynomial
    degree: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)
    method: Literal["KRYLOV_NULLSPACE"] = "KRYLOV_NULLSPACE"

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        from jacobian.math.matrices.canonical_forms._replay import (
            _coefficients_of,
            _matrix_entries,
            _matrix_from_request,
            _polynomial_degree,
        )
        from jacobian.math.matrices.canonical_forms.operations import (
            characteristic_polynomial as replay_characteristic,
        )
        from jacobian.math.matrices.canonical_forms.operations import (
            minimal_polynomial as replay_minimal,
        )

        if self.degree != _polynomial_degree(self.minimal_polynomial):
            raise ValueError("degree must equal the minimal-polynomial degree")
        entries = _matrix_entries(self.matrix)
        if _coefficients_of(self.minimal_polynomial) != tuple(replay_minimal(entries)):
            raise ValueError(
                "minimal polynomial must be the exact minimal polynomial of "
                "the retained matrix"
            )
        if _coefficients_of(self.characteristic_polynomial) != tuple(
            replay_characteristic(entries)
        ):
            raise ValueError(
                "characteristic polynomial must be the characteristic "
                "polynomial of the retained matrix"
            )
        matrix = _matrix_from_request(self.matrix)
        evaluated = matrix.zeros(matrix.rows, matrix.cols)
        power = matrix.eye(matrix.rows)
        for coefficient in self.minimal_polynomial.coefficients:
            evaluated = evaluated + power * coefficient.as_fraction()
            power = power * matrix
        if evaluated != matrix.zeros(matrix.rows, matrix.cols):
            raise ValueError("minimal polynomial must annihilate the retained matrix")
        return self


class InvariantFactorEntry(StrictModel):
    """One monic invariant factor from the rational canonical form."""

    factor: MonicPolynomial
    block_size: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)


class RationalCanonicalFormResult(StrictModel):
    """Exact rational (Frobenius) canonical form of a square rational matrix.

    Retains the source matrix so validation replays the invariant-factor
    relations: each block size equals its factor's degree, sizes total the
    matrix dimension, factors divide successively, their product equals the
    characteristic polynomial of the retained matrix, the last factor
    is that matrix's minimal polynomial, and the claimed tuple equals the
    exact invariant-factor tuple re-derived from the retained matrix.
    """

    matrix: SquareMatrixRequest
    invariant_factors: tuple[InvariantFactorEntry, ...] = Field(min_length=1)
    characteristic_polynomial: MonicPolynomial
    minimal_polynomial: MonicPolynomial
    total_block_size: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)
    method: Literal["SMITH_NORMAL_FORM"] = "SMITH_NORMAL_FORM"

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        from jacobian.math.matrices.canonical_forms._replay import (
            _coefficients_of,
            _matrix_entries,
            _poly_from_monic,
            _polynomial_degree,
        )
        from jacobian.math.matrices.canonical_forms.operations import (
            characteristic_polynomial as replay_characteristic,
        )
        from jacobian.math.matrices.canonical_forms.operations import (
            invariant_factors as replay_invariant_factors,
        )
        from jacobian.math.matrices.canonical_forms.operations import (
            minimal_polynomial as replay_minimal,
        )

        entries = _matrix_entries(self.matrix)
        dimension = len(entries)
        if self.total_block_size != sum(
            entry.block_size for entry in self.invariant_factors
        ):
            raise ValueError("total block size must equal the summed block sizes")
        if self.total_block_size != dimension:
            raise ValueError("block sizes must total the matrix dimension")
        for entry in self.invariant_factors:
            if entry.block_size != _polynomial_degree(entry.factor):
                raise ValueError("each block size must equal its factor degree")

        if _coefficients_of(self.minimal_polynomial) != tuple(replay_minimal(entries)):
            raise ValueError(
                "minimal polynomial must be the exact minimal polynomial of "
                "the retained matrix"
            )
        if _coefficients_of(self.characteristic_polynomial) != tuple(
            replay_characteristic(entries)
        ):
            raise ValueError(
                "characteristic polynomial must be the characteristic "
                "polynomial of the retained matrix"
            )

        previous = None
        product = None
        for entry in self.invariant_factors:
            factor = _poly_from_monic(entry.factor)
            if previous is not None:
                _quotient, remainder = factor.div(previous)
                if remainder.as_expr() != 0:
                    raise ValueError("invariant factors must divide successively")
            product = factor if product is None else product * factor
            previous = factor
        characteristic = _poly_from_monic(self.characteristic_polynomial)
        if product is None or product.as_expr() != characteristic.as_expr():
            raise ValueError(
                "invariant factors must multiply to the characteristic polynomial"
            )
        last_factor = _poly_from_monic(self.invariant_factors[-1].factor)
        minimal = _poly_from_monic(self.minimal_polynomial)
        if last_factor.as_expr() != minimal.as_expr():
            raise ValueError("the final invariant factor is the minimal polynomial")
        if tuple(
            _coefficients_of(entry.factor) for entry in self.invariant_factors
        ) != tuple(replay_invariant_factors(entries)):
            raise ValueError(
                "invariant factors must be the exact invariant factors of "
                "the retained matrix"
            )
        return self


class PrimaryDecompositionResult(StrictModel):
    """Primary decomposition of the minimal polynomial into irreducible-power components.

    Retains the source matrix so validation replays the defining relations:
    every component is one monic irreducible power, components are pairwise
    coprime, and their product equals the source minimal polynomial
    re-derived from the retained matrix (for pairwise-coprime components the
    product equals their least common multiple).
    """

    matrix: SquareMatrixRequest
    components: tuple[MonicPolynomial, ...] = Field(min_length=1)
    minimal_polynomial: MonicPolynomial
    method: Literal["FACTOR_LCM"] = "FACTOR_LCM"

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        from jacobian.math.matrices.canonical_forms._replay import (
            _coefficients_of,
            _matrix_entries,
            _poly_from_monic,
        )
        from jacobian.math.matrices.canonical_forms.operations import (
            minimal_polynomial as replay_minimal,
        )

        entries = _matrix_entries(self.matrix)

        if _coefficients_of(self.minimal_polynomial) != tuple(replay_minimal(entries)):
            raise ValueError(
                "minimal polynomial must be the exact minimal polynomial of "
                "the retained matrix"
            )

        component_polys = [_poly_from_monic(component) for component in self.components]
        for component in component_polys:
            _content, factors = component.factor_list()
            if len(factors) != 1:
                raise ValueError(
                    "each primary component must be one monic irreducible power"
                )
        for index, first in enumerate(component_polys):
            for second in component_polys[index + 1 :]:
                if first.gcd(second).degree() >= 1:
                    raise ValueError("primary components must be pairwise coprime")
        product = component_polys[0]
        for component in component_polys[1:]:
            product = product * component
        minimal = _poly_from_monic(self.minimal_polynomial)
        if product.as_expr() != minimal.as_expr():
            raise ValueError(
                "primary components must multiply to the claimed minimal polynomial"
            )
        return self


__all__ = [
    "MAX_CANONICAL_FORM_DIMENSION",
    "MAX_CANONICAL_FORM_SCALAR_DIGITS",
    "InvariantFactorEntry",
    "MinimalPolynomialResult",
    "MonicPolynomial",
    "PrimaryDecompositionResult",
    "RationalCanonicalFormResult",
    "SquareMatrixRequest",
]
