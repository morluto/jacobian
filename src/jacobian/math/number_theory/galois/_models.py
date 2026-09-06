"""Typed wire contracts for bounded Galois-theory operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.polynomials.values import RationalPolynomial

MAX_FACTOR_DEGREE = 12
MAX_GALOIS_GROUP_DEGREE = 6
MAX_FIELD_ORDER = 251
GaloisCoefficient = Annotated[int, Field(ge=-(10**12), le=10**12, strict=True)]
PositiveFactorDegree = Annotated[
    int,
    Field(ge=1, le=MAX_FACTOR_DEGREE, strict=True),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"galois_theory.{reason}", message)


def _require_prime(value: int) -> None:
    from sympy import isprime

    if not isprime(value):
        raise _validation_error("field_order_not_prime", "field_order must be prime")


def _supported_galois_polynomial(coefficients: tuple[int, ...]) -> None:
    from sympy import Poly, Symbol

    if not 2 <= len(coefficients) <= MAX_GALOIS_GROUP_DEGREE + 1:
        raise _validation_error(
            "degree_bound",
            "Galois computation requires degree one through six",
        )
    if any(
        type(coefficient) is not int or abs(coefficient) > 10**12
        for coefficient in coefficients
    ):
        raise _validation_error(
            "coefficient_bound",
            "Galois coefficients must be integers of magnitude at most 10^12",
        )
    if coefficients[-1] == 0:
        raise _validation_error(
            "leading_coefficient_zero", "leading coefficient must be nonzero"
        )
    polynomial = Poly.from_list(list(reversed(coefficients)), Symbol("x"), domain="QQ")
    if not polynomial.is_irreducible:
        raise _validation_error(
            "polynomial_not_irreducible",
            "SymPy galois_group requires an irreducible polynomial over QQ",
        )


class GaloisFactorRequest(StrictModel):
    """A nonzero, nonconstant polynomial over the prime field ``GF(p)``."""

    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER, strict=True)
    coefficients: tuple[int, ...] = Field(
        min_length=2,
        max_length=MAX_FACTOR_DEGREE + 1,
    )


class FrobeniusCycleRequest(StrictModel):
    """A positive factor-degree partition of a polynomial degree."""

    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER, strict=True)
    polynomial_degree: int = Field(ge=1, le=MAX_FACTOR_DEGREE, strict=True)
    factorization_degrees: tuple[PositiveFactorDegree, ...] = Field(
        min_length=1,
        max_length=MAX_FACTOR_DEGREE,
    )


class _SupportedGaloisPolynomialRequest(StrictModel):
    polynomial: RationalPolynomial = Field(
        description="Univariate polynomial over QQ with integer coefficients of magnitude at most 10^12 and degree one through six."
    )

    @model_validator(mode="after")
    def require_supported_encoding(self) -> Self:
        terms = self.polynomial.polynomial.terms
        if (
            len(self.polynomial.variables) != 1
            or not terms
            or not 1 <= terms[0].exponents[0] <= MAX_GALOIS_GROUP_DEGREE
        ):
            raise _validation_error(
                "degree_bound",
                "Galois computation requires a univariate polynomial of degree one through six",
            )
        if any(
            term.coefficient.den != "1"
            or len(term.coefficient.num.lstrip("-")) > 13
            or abs(parse_canonical_integer(term.coefficient.num)) > 10**12
            for term in terms
        ):
            raise _validation_error(
                "coefficient_bound",
                "Galois coefficients must be integers of magnitude at most 10^12",
            )
        return self

    @property
    def coefficients(self) -> tuple[int, ...]:
        coefficients = [0] * (self.polynomial.polynomial.terms[0].exponents[0] + 1)
        for term in self.polynomial.polynomial.terms:
            coefficients[term.exponents[0]] = parse_canonical_integer(
                term.coefficient.num
            )
        return tuple(coefficients)


class GaloisGroupRequest(_SupportedGaloisPolynomialRequest):
    """An irreducible degree-one-through-six polynomial over ``QQ``."""


class SolvableRequest(_SupportedGaloisPolynomialRequest):
    """The supported SymPy domain for deciding radical solvability."""


class FiniteFieldFactor(StrictModel):
    """One monic irreducible factor with positive multiplicity."""

    coefficients: tuple[int, ...] = Field(
        min_length=2,
        max_length=MAX_FACTOR_DEGREE + 1,
    )
    multiplicity: int = Field(ge=1, le=MAX_FACTOR_DEGREE, strict=True)


class GaloisFactorResult(StrictModel):
    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER, strict=True)
    source_coefficients: tuple[int, ...] = Field(
        min_length=2,
        max_length=MAX_FACTOR_DEGREE + 1,
    )
    unit: int = Field(ge=1, le=MAX_FIELD_ORDER - 1, strict=True)
    factors: tuple[FiniteFieldFactor, ...] = Field(
        min_length=1,
        max_length=MAX_FACTOR_DEGREE,
    )

    @property
    def distinct_factor_count(self) -> int:
        return len(self.factors)

    @property
    def factor_count(self) -> int:
        return sum(factor.multiplicity for factor in self.factors)

    @property
    def is_irreducible(self) -> bool:
        return (
            len(self.factors) == 1
            and self.factors[0].multiplicity == 1
            and len(self.factors[0].coefficients) == len(self.source_coefficients)
        )

    @classmethod
    def _from_kernel(
        cls,
        *,
        field_order: int,
        source_coefficients: tuple[int, ...],
        unit: int,
        factors: tuple[FiniteFieldFactor, ...],
    ) -> Self:
        """Construct output whose complete factorization came from the kernel."""

        return cls.model_construct(
            field_order=field_order,
            source_coefficients=source_coefficients,
            unit=unit,
            factors=factors,
        )

    @model_validator(mode="after")
    def require_structural_consistency(self) -> Self:
        _require_factor_residues(self)
        return self


def _require_factor_residues(result: GaloisFactorResult) -> None:
    prime = result.field_order
    if result.unit >= prime:
        raise _validation_error(
            "unit_not_canonical",
            "factorization unit must be a canonical nonzero residue",
        )
    if any(not 0 <= coefficient < prime for coefficient in result.source_coefficients):
        raise _validation_error(
            "source_coefficients_not_canonical",
            "source coefficients must be canonical field residues",
        )
    for factor in result.factors:
        if any(not 0 <= coefficient < prime for coefficient in factor.coefficients):
            raise _validation_error(
                "factor_coefficients_not_canonical",
                "factor coefficients must be canonical field residues",
            )
        if factor.coefficients[-1] != 1:
            raise _validation_error(
                "factor_not_monic", "finite-field factors must be monic"
            )


class FrobeniusCycleResult(StrictModel):
    cycle_type: tuple[PositiveFactorDegree, ...]

    @property
    def degree(self) -> int:
        return sum(self.cycle_type)

    @property
    def is_irreducible(self) -> bool:
        return self.cycle_type == (self.degree,)

    @model_validator(mode="after")
    def require_canonical_partition(self) -> Self:
        if self.cycle_type != tuple(sorted(self.cycle_type, reverse=True)):
            raise _validation_error(
                "cycle_type_not_canonical",
                "cycle type must be sorted in descending order",
            )
        return self


class GaloisRootAxis(StrictModel):
    """The ordered root positions of one retained source polynomial.

    Positions are canonical indices; the polynomial is the source that gives
    those positions their mathematical meaning.  The shape checks here only
    establish a well-formed axis.  Relation verification belongs to the
    public claim verifiers.
    """

    polynomial: RationalPolynomial
    indices: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_GALOIS_GROUP_DEGREE,
    )

    @model_validator(mode="after")
    def require_structural_axis(self) -> Self:
        terms = self.polynomial.polynomial.terms
        if len(self.polynomial.variables) != 1 or not terms:
            raise _validation_error(
                "root_axis_requires_univariate_polynomial",
                "root axis requires a nonempty univariate polynomial",
            )
        degree = terms[0].exponents[0]
        if len(self.indices) != degree or self.indices != tuple(range(degree)):
            raise _validation_error(
                "root_axis_not_canonical",
                "root positions must be the canonical degree-sized index axis",
            )
        return self


class FinitePermutationGroup(StrictModel):
    """A composable permutation group on one polynomial root axis."""

    root_axis: GaloisRootAxis
    generators: tuple[tuple[StrictInt, ...], ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_permutations_on_axis(self) -> Self:
        expected = tuple(range(len(self.root_axis.indices)))
        if any(
            len(generator) != len(expected) or tuple(sorted(generator)) != expected
            for generator in self.generators
        ):
            raise _validation_error(
                "generator_not_permutation",
                "every generator must permute the complete root axis",
            )
        return self


class GaloisGroupResult(StrictModel):
    group: FinitePermutationGroup
    group_name: str
    order: int = Field(ge=1)
    degree: int = Field(ge=1, le=MAX_GALOIS_GROUP_DEGREE)
    is_solvable: bool

    @property
    def polynomial(self) -> RationalPolynomial:
        """Return the canonical source retained by the root axis."""

        return self.group.root_axis.polynomial

    @classmethod
    def _from_kernel(
        cls,
        *,
        group: FinitePermutationGroup,
        group_name: str,
        order: int,
        degree: int,
        is_solvable: bool,
    ) -> Self:
        """Construct output whose group properties came from the kernel."""

        return cls.model_construct(
            group=group,
            group_name=group_name,
            order=order,
            degree=degree,
            is_solvable=is_solvable,
        )

    @model_validator(mode="after")
    def require_group_degree(self) -> Self:
        if self.degree != len(self.group.root_axis.indices):
            raise _validation_error(
                "group_degree_mismatch",
                "group root axis must match the polynomial degree",
            )
        return self


class SolvableResult(StrictModel):
    solvable_by_radicals: bool
    group: FinitePermutationGroup

    @property
    def polynomial(self) -> RationalPolynomial:
        """Return the canonical source retained by the root axis."""

        return self.group.root_axis.polynomial

    @classmethod
    def _from_kernel(
        cls,
        *,
        solvable_by_radicals: bool,
        group: FinitePermutationGroup,
    ) -> Self:
        """Construct output whose solvability came from the kernel."""

        return cls.model_construct(
            solvable_by_radicals=solvable_by_radicals,
            group=group,
        )


__all__ = [
    "FiniteFieldFactor",
    "FinitePermutationGroup",
    "FrobeniusCycleRequest",
    "FrobeniusCycleResult",
    "GaloisFactorRequest",
    "GaloisFactorResult",
    "GaloisGroupRequest",
    "GaloisGroupResult",
    "GaloisRootAxis",
    "SolvableRequest",
    "SolvableResult",
]
