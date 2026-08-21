"""Typed wire contracts for bounded Galois-theory operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian._models import StrictModel

MAX_FACTOR_DEGREE = 12
MAX_GALOIS_GROUP_DEGREE = 6
MAX_FIELD_ORDER = 251
GaloisCoefficient = Annotated[int, Field(ge=-(10**12), le=10**12, strict=True)]
RootIdentifier = Annotated[
    str,
    StringConstraints(pattern=r"^root_[0-9]+$", strict=True),
]
PositiveFactorDegree = Annotated[
    int,
    Field(ge=1, le=MAX_FACTOR_DEGREE, strict=True),
]


def _require_prime(value: int) -> None:
    from sympy import isprime

    if not isprime(value):
        raise ValueError("field_order must be prime")


def _supported_galois_polynomial(coefficients: tuple[int, ...]) -> None:
    from sympy import Poly, Symbol

    if coefficients[-1] == 0:
        raise ValueError("leading coefficient must be nonzero")
    polynomial = Poly.from_list(list(reversed(coefficients)), Symbol("x"), domain="QQ")
    if not polynomial.is_irreducible:
        raise ValueError(
            "SymPy galois_group requires an irreducible polynomial over QQ"
        )


class GaloisFactorRequest(StrictModel):
    """A nonzero, nonconstant polynomial over the prime field ``GF(p)``."""

    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER, strict=True)
    coefficients: tuple[int, ...] = Field(
        min_length=2,
        max_length=MAX_FACTOR_DEGREE + 1,
    )

    @model_validator(mode="after")
    def require_supported_polynomial(self) -> Self:
        _require_prime(self.field_order)
        if any(
            not 0 <= coefficient < self.field_order for coefficient in self.coefficients
        ):
            raise ValueError("coefficients must be canonical field residues")
        if self.coefficients[-1] == 0:
            raise ValueError(
                "factorization requires a nonzero polynomial with canonical degree"
            )
        return self


class FrobeniusCycleRequest(StrictModel):
    """A positive factor-degree partition of a polynomial degree."""

    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER, strict=True)
    polynomial_degree: int = Field(ge=1, le=MAX_FACTOR_DEGREE, strict=True)
    factorization_degrees: tuple[PositiveFactorDegree, ...] = Field(
        min_length=1,
        max_length=MAX_FACTOR_DEGREE,
    )

    @model_validator(mode="after")
    def require_positive_partition(self) -> Self:
        from collections import Counter

        from sympy import divisors, mobius

        _require_prime(self.field_order)
        if sum(self.factorization_degrees) != self.polynomial_degree:
            raise ValueError("factorization degrees must sum to polynomial degree")
        for degree, count in Counter(self.factorization_degrees).items():
            available = (
                sum(
                    int(mobius(divisor)) * self.field_order ** (degree // divisor)
                    for divisor in divisors(degree)
                )
                // degree
            )
            if count > available:
                raise ValueError(
                    "factorization pattern exceeds the available distinct "
                    f"degree-{degree} irreducible factors over the field"
                )
        return self


class _SupportedGaloisPolynomialRequest(StrictModel):
    coefficients: tuple[GaloisCoefficient, ...] = Field(
        min_length=2,
        max_length=MAX_GALOIS_GROUP_DEGREE + 1,
    )

    @model_validator(mode="after")
    def require_backend_domain(self) -> Self:
        _supported_galois_polynomial(self.coefficients)
        return self


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


def _multiply_mod(
    left: tuple[int, ...], right: tuple[int, ...], prime: int
) -> tuple[int, ...]:
    result = [0] * (len(left) + len(right) - 1)
    for left_degree, left_coefficient in enumerate(left):
        for right_degree, right_coefficient in enumerate(right):
            result[left_degree + right_degree] = (
                result[left_degree + right_degree]
                + left_coefficient * right_coefficient
            ) % prime
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result)


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
    distinct_factor_count: int = Field(ge=1, le=MAX_FACTOR_DEGREE, strict=True)
    factor_count: int = Field(ge=1, le=MAX_FACTOR_DEGREE, strict=True)
    is_irreducible: bool
    method: str = "SYMPY_FACTOR_MOD_P"

    @model_validator(mode="after")
    def require_reconstruction_certificate(self) -> Self:
        _require_prime(self.field_order)
        _require_factor_residues(self)
        if self.distinct_factor_count != len(self.factors):
            raise ValueError("distinct_factor_count must equal the number of factors")
        total = sum(factor.multiplicity for factor in self.factors)
        if self.factor_count != total:
            raise ValueError("factor_count must include factor multiplicities")
        if _reconstruct_factorization(self) != self.source_coefficients:
            raise ValueError("factorization must reconstruct the source modulo p")
        if self.is_irreducible != _factorization_is_irreducible(self):
            raise ValueError(
                "irreducibility must agree with the complete factorization"
            )
        return self


def _require_factor_residues(result: GaloisFactorResult) -> None:
    from sympy import GF, Poly, Symbol

    prime = result.field_order
    if result.unit >= prime:
        raise ValueError("factorization unit must be a canonical nonzero residue")
    if any(not 0 <= coefficient < prime for coefficient in result.source_coefficients):
        raise ValueError("source coefficients must be canonical field residues")
    for factor in result.factors:
        if any(not 0 <= coefficient < prime for coefficient in factor.coefficients):
            raise ValueError("factor coefficients must be canonical field residues")
        if factor.coefficients[-1] != 1:
            raise ValueError("finite-field factors must be monic")
        polynomial = Poly(
            list(reversed(factor.coefficients)),
            Symbol("x"),
            domain=GF(prime),
        )
        if not polynomial.is_irreducible:
            raise ValueError("every finite-field factor must be irreducible")


def _reconstruct_factorization(result: GaloisFactorResult) -> tuple[int, ...]:
    reconstructed: tuple[int, ...] = (result.unit,)
    for factor in result.factors:
        for _ in range(factor.multiplicity):
            reconstructed = _multiply_mod(
                reconstructed, factor.coefficients, result.field_order
            )
    return reconstructed


def _factorization_is_irreducible(result: GaloisFactorResult) -> bool:
    return (
        len(result.factors) == 1
        and result.factors[0].multiplicity == 1
        and len(result.factors[0].coefficients) == len(result.source_coefficients)
    )


class FrobeniusCycleResult(StrictModel):
    cycle_type: tuple[PositiveFactorDegree, ...]
    degree: int = Field(ge=1, le=MAX_FACTOR_DEGREE, strict=True)
    is_irreducible: bool
    method: str = "FACTOR_DEGREE_SUMMARY"

    @model_validator(mode="after")
    def require_canonical_partition(self) -> Self:
        if sum(self.cycle_type) != self.degree:
            raise ValueError("cycle type must partition the polynomial degree")
        if self.cycle_type != tuple(sorted(self.cycle_type, reverse=True)):
            raise ValueError("cycle type must be sorted in descending order")
        if self.is_irreducible != (self.cycle_type == (self.degree,)):
            raise ValueError("irreducibility must agree with the cycle type")
        return self


class FinitePermutationGroup(StrictModel):
    """A composable permutation group on one explicit ordered root axis."""

    root_axis: tuple[RootIdentifier, ...] = Field(
        min_length=1,
        max_length=MAX_GALOIS_GROUP_DEGREE,
    )
    generators: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_permutations_on_axis(self) -> Self:
        if len(set(self.root_axis)) != len(self.root_axis):
            raise ValueError("root axis entries must be unique")
        expected = tuple(range(len(self.root_axis)))
        if any(
            len(generator) != len(expected) or tuple(sorted(generator)) != expected
            for generator in self.generators
        ):
            raise ValueError("every generator must permute the complete root axis")
        return self


def _permutation_group_properties(group: FinitePermutationGroup) -> tuple[int, bool]:
    from sympy.combinatorics import Permutation, PermutationGroup

    backend = PermutationGroup(
        *(Permutation(list(generator)) for generator in group.generators)
    )
    return int(backend.order()), bool(backend.is_solvable)


class GaloisGroupResult(StrictModel):
    group: FinitePermutationGroup
    group_name: str
    order: int = Field(ge=1)
    degree: int = Field(ge=1, le=MAX_GALOIS_GROUP_DEGREE)
    is_solvable: bool
    method: str = "SYMPY_GALOIS_GROUP"

    @model_validator(mode="after")
    def require_group_degree(self) -> Self:
        if self.degree != len(self.group.root_axis):
            raise ValueError("group root axis must match the polynomial degree")
        order, is_solvable = _permutation_group_properties(self.group)
        if self.order != order or self.is_solvable != is_solvable:
            raise ValueError(
                "reported group properties must agree with the permutation generators"
            )
        return self


class SolvableResult(StrictModel):
    solvable_by_radicals: bool
    group: FinitePermutationGroup
    method: str = "GALOIS_GROUP_SOLVABILITY"

    @model_validator(mode="after")
    def require_group_certificate(self) -> Self:
        _, is_solvable = _permutation_group_properties(self.group)
        if self.solvable_by_radicals != is_solvable:
            raise ValueError(
                "radical solvability must agree with the permutation generators"
            )
        return self


__all__ = [
    "FiniteFieldFactor",
    "FinitePermutationGroup",
    "FrobeniusCycleRequest",
    "FrobeniusCycleResult",
    "GaloisFactorRequest",
    "GaloisFactorResult",
    "GaloisGroupRequest",
    "GaloisGroupResult",
    "SolvableRequest",
    "SolvableResult",
]
