"""Canonical bracket symbols and sparse formal bracket polynomials (#2775).

A bracket is the alternating symbol ``[i j k]`` on three distinct indexed
ground elements.  Ordering is normalized to the increasing triple and the
sign of the permutation is retained as an explicit parity, so an ordered
presentation and its canonical form carry the same mathematical meaning.

A bracket polynomial is a finite rational combination of commutative
monomials in canonical bracket atoms.  It is a polynomial in *formal* bracket
atoms: the value does not assert that the atoms are algebraically independent,
and it is not a coordinate polynomial.
"""

from __future__ import annotations

from typing import Literal, Self, cast

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    ExactInteger,
)
from jacobian._models import StrictModel

MAX_BRACKET_GROUND_SIZE = 12
MAX_BRACKET_FACTORS = 4
MAX_BRACKET_TERMS = 512
MAX_BRACKET_CONTRIBUTIONS = 65_536
MAX_BRACKET_COEFFICIENT_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS
MAX_BRACKET_OUTPUT_CELLS = MAX_BRACKET_TERMS * (MAX_BRACKET_FACTORS + 1)
# Intrinsic allocation envelope for one residual value, measured in retained
# scalar digits and structural slots. Concrete transports own encoded-byte
# ceilings independently.
MAX_BRACKET_RESULT_ALLOCATION_UNITS = 64 * 1024 * 1024


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


class CanonicalBracket(StrictModel):
    """One canonical alternating bracket ``[i j k]`` on an increasing triple."""

    indices: tuple[StrictInt, StrictInt, StrictInt]

    @model_validator(mode="after")
    def require_strictly_increasing(self) -> Self:
        left, middle, right = self.indices
        if not (0 <= left < middle < right < MAX_BRACKET_GROUND_SIZE):
            raise _validation_error(
                "bracket.indices_not_canonical",
                "a canonical bracket requires an increasing triple inside the "
                "admitted ground range",
            )
        return self


def ordered_bracket(ordered: tuple[int, int, int]) -> tuple[CanonicalBracket, int]:
    """Return the canonical bracket and the parity sign of an ordered triple.

    Duplicate indices are an alternating-symbol zero and are rejected rather
    than silently dropped, so a caller cannot lose a degeneracy.
    """

    if len(set(ordered)) != 3:
        raise _validation_error(
            "bracket.duplicate_index_zero",
            "an ordered bracket with a repeated index is the zero alternating symbol",
        )
    if any(index < 0 or index >= MAX_BRACKET_GROUND_SIZE for index in ordered):
        raise _validation_error(
            "bracket.index_out_of_range",
            "ordered bracket indices must lie inside the admitted ground range",
        )
    canonical = cast(tuple[int, int, int], tuple(sorted(ordered)))
    inversions = sum(
        1
        for left_position in range(3)
        for right_position in range(left_position + 1, 3)
        if ordered[left_position] > ordered[right_position]
    )
    sign = -1 if inversions % 2 else 1
    return CanonicalBracket(indices=canonical), sign


class BracketMonomial(StrictModel):
    """A commutative product of distinct canonical bracket atoms.

    Factors are kept in canonical sorted order and repeats are combined by the
    declared multiplicity, so two presentations of the same monomial are equal.
    """

    factors: tuple[tuple[CanonicalBracket, ExactInteger], ...] = Field(
        max_length=MAX_BRACKET_FACTORS
    )

    @model_validator(mode="after")
    def require_canonical_factor_order(self) -> Self:
        keys = tuple(factor.indices for factor, _ in self.factors)
        if keys != tuple(sorted(keys)):
            raise _validation_error(
                "bracket.monomial_factors_not_canonical",
                "bracket-monomial factors must be in canonical sorted order",
            )
        if len(set(keys)) != len(keys):
            raise _validation_error(
                "bracket.monomial_factors_not_combined",
                "repeated bracket factors must be combined into one multiplicity",
            )
        if any(multiplicity < 1 for _, multiplicity in self.factors):
            raise _validation_error(
                "bracket.monomial_multiplicity",
                "a bracket factor multiplicity must be a positive integer",
            )
        return self

    @property
    def total_degree(self) -> int:
        return sum(multiplicity for _, multiplicity in self.factors)


class BracketPolynomialTerm(StrictModel):
    coefficient: CanonicalRational
    monomial: BracketMonomial

    @model_validator(mode="after")
    def require_nonzero_coefficient(self) -> Self:
        if self.coefficient.as_fraction() == 0:
            raise _validation_error(
                "bracket.zero_coefficient_term",
                "a bracket polynomial omits terms with zero coefficient",
            )
        return self


class BracketPolynomial(StrictModel):
    """A canonical sparse formal bracket polynomial over a declared ground size."""

    ground_size: StrictInt = Field(ge=3, le=MAX_BRACKET_GROUND_SIZE)
    terms: tuple[BracketPolynomialTerm, ...] = Field(max_length=MAX_BRACKET_TERMS)

    @model_validator(mode="after")
    def require_canonical_terms(self) -> Self:
        keys = tuple(
            tuple(
                (factor.indices, multiplicity)
                for factor, multiplicity in term.monomial.factors
            )
            for term in self.terms
        )
        if keys != tuple(sorted(keys)):
            raise _validation_error(
                "bracket.polynomial_terms_not_canonical",
                "bracket polynomial terms must be in canonical order",
            )
        if len(set(keys)) != len(keys):
            raise _validation_error(
                "bracket.polynomial_terms_not_combined",
                "equal bracket monomials must be combined into one term",
            )
        for term in self.terms:
            for factor, _ in term.monomial.factors:
                if factor.indices[2] >= self.ground_size:
                    raise _validation_error(
                        "bracket.index_outside_ground",
                        "every bracket index must lie in the declared ground range",
                    )
        return self


class GrassmannPlueckerRelationRequest(StrictModel):
    """One indexed parameter pattern of a rank-3 Grassmann-Pluecker relation."""

    ground_size: StrictInt = Field(ge=3, le=MAX_BRACKET_GROUND_SIZE)
    indices: tuple[StrictInt, ...] = Field(min_length=5, max_length=6)
    family: Literal["FOUR_TERM", "SHARED_INDEX_THREE_TERM"]

    @model_validator(mode="after")
    def require_relation_shape(self) -> Self:
        expected = 6 if self.family == "FOUR_TERM" else 5
        if len(self.indices) != expected:
            raise _validation_error(
                "bracket.relation_index_count",
                f"{self.family} requires exactly {expected} indices",
            )
        if self.ground_size < expected:
            raise _validation_error(
                "bracket.relation_ground_too_small",
                "the declared ground range must contain every relation index",
            )
        if any(index < 0 or index >= self.ground_size for index in self.indices):
            raise _validation_error(
                "bracket.relation_index_outside_ground",
                "every relation index must lie inside the declared ground range",
            )
        if len(set(self.indices)) != len(self.indices):
            raise _validation_error(
                "bracket.relation_indices_not_distinct",
                "a Grassmann-Pluecker relation requires distinct indices",
            )
        return self


class GrassmannPlueckerRelation(StrictModel):
    """A formal relation polynomial bound to its indexed GP source."""

    ground_size: StrictInt = Field(ge=3, le=MAX_BRACKET_GROUND_SIZE)
    indices: tuple[StrictInt, ...] = Field(min_length=5, max_length=6)
    family: Literal["FOUR_TERM", "SHARED_INDEX_THREE_TERM"]
    polynomial: BracketPolynomial

    @model_validator(mode="after")
    def require_relation_shape(self) -> Self:
        expected = 6 if self.family == "FOUR_TERM" else 5
        if len(self.indices) != expected or len(set(self.indices)) != expected:
            raise _validation_error(
                "bracket.relation_index_shape",
                f"{self.family} must retain {expected} distinct indices",
            )
        if not self.polynomial.terms:
            raise _validation_error(
                "bracket.relation_empty",
                "a Grassmann-Pluecker relation must produce at least one bracket term",
            )
        if any(index < 0 or index >= self.ground_size for index in self.indices):
            raise _validation_error(
                "bracket.relation_index_outside_ground",
                "every relation index must lie inside the declared ground range",
            )
        if self.polynomial.ground_size != self.ground_size:
            raise _validation_error(
                "bracket.relation_ground_mismatch",
                "the relation polynomial must share the declared ground range",
            )
        return self


class GrassmannPlueckerRelationResult(GrassmannPlueckerRelation):
    """The canonical formal expression of one Grassmann-Pluecker relation."""


class BracketSyzygyResidualRequest(StrictModel):
    """One target polynomial minus a finite combination of supplied relations."""

    target: BracketPolynomial
    terms: tuple[
        tuple[CanonicalRational, BracketMonomial, GrassmannPlueckerRelation], ...
    ] = Field(
        max_length=128,
        description=(
            "Finite terms (scalar, multiplier monomial, source-bound formal relation); "
            "all polynomials must use the target ground range."
        ),
    )

    @model_validator(mode="after")
    def require_common_ground(self) -> Self:
        for _, multiplier, relation in self.terms:
            if relation.ground_size != self.target.ground_size:
                raise _validation_error(
                    "bracket.syzygy_ground_mismatch",
                    "target and every supplied relation must share one ground range",
                )
            if any(
                factor.indices[2] >= self.target.ground_size
                for factor, _ in multiplier.factors
            ):
                raise _validation_error(
                    "bracket.syzygy_multiplier_index_outside_ground",
                    "every multiplier bracket index must lie in the target ground range",
                )
        return self


__all__ = [
    "MAX_BRACKET_COEFFICIENT_DIGITS",
    "MAX_BRACKET_CONTRIBUTIONS",
    "MAX_BRACKET_FACTORS",
    "MAX_BRACKET_GROUND_SIZE",
    "MAX_BRACKET_OUTPUT_CELLS",
    "MAX_BRACKET_RESULT_ALLOCATION_UNITS",
    "MAX_BRACKET_TERMS",
    "BracketMonomial",
    "BracketPolynomial",
    "BracketPolynomialTerm",
    "BracketSyzygyResidualRequest",
    "CanonicalBracket",
    "GrassmannPlueckerRelation",
    "GrassmannPlueckerRelationRequest",
    "GrassmannPlueckerRelationResult",
    "ordered_bracket",
]
