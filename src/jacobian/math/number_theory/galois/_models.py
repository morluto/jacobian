"""Typed wire contracts for bounded Galois-theory operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

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


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"galois_theory.{reason}", message)


def _require_prime(value: int) -> None:
    from sympy import isprime

    if not isprime(value):
        raise _validation_error("field_order_not_prime", "field_order must be prime")


def _supported_galois_polynomial(coefficients: tuple[int, ...]) -> None:
    from sympy import Poly, Symbol

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

    @model_validator(mode="after")
    def require_supported_polynomial(self) -> Self:
        _require_prime(self.field_order)
        if any(
            not 0 <= coefficient < self.field_order for coefficient in self.coefficients
        ):
            raise _validation_error(
                "coefficients_not_canonical",
                "coefficients must be canonical field residues",
            )
        if self.coefficients[-1] == 0:
            raise _validation_error(
                "polynomial_zero",
                "factorization requires a nonzero polynomial with canonical degree",
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
            raise _validation_error(
                "partition_degree_mismatch",
                "factorization degrees must sum to polynomial degree",
            )
        for degree, count in Counter(self.factorization_degrees).items():
            available = (
                sum(
                    int(mobius(divisor)) * self.field_order ** (degree // divisor)
                    for divisor in divisors(degree)
                )
                // degree
            )
            if count > available:
                raise _validation_error(
                    "partition_unrealizable",
                    "factorization pattern exceeds the available distinct "
                    f"degree-{degree} irreducible factors over the field",
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

    @classmethod
    def _from_kernel(
        cls,
        *,
        field_order: int,
        source_coefficients: tuple[int, ...],
        unit: int,
        factors: tuple[FiniteFieldFactor, ...],
        distinct_factor_count: int,
        factor_count: int,
        is_irreducible: bool,
    ) -> Self:
        """Construct output whose complete factorization came from the kernel."""

        return cls.model_construct(
            field_order=field_order,
            source_coefficients=source_coefficients,
            unit=unit,
            factors=factors,
            distinct_factor_count=distinct_factor_count,
            factor_count=factor_count,
            is_irreducible=is_irreducible,
        )

    @model_validator(mode="after")
    def require_structural_consistency(self) -> Self:
        _require_prime(self.field_order)
        _require_factor_residues(self)
        if self.distinct_factor_count != len(self.factors):
            raise _validation_error(
                "distinct_factor_count_mismatch",
                "distinct_factor_count must equal the number of factors",
            )
        total = sum(factor.multiplicity for factor in self.factors)
        if self.factor_count != total:
            raise _validation_error(
                "factor_count_mismatch",
                "factor_count must include factor multiplicities",
            )
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
    degree: int = Field(ge=1, le=MAX_FACTOR_DEGREE, strict=True)
    is_irreducible: bool
    method: str = "FACTOR_DEGREE_SUMMARY"

    @model_validator(mode="after")
    def require_canonical_partition(self) -> Self:
        if sum(self.cycle_type) != self.degree:
            raise _validation_error(
                "cycle_type_degree_mismatch",
                "cycle type must partition the polynomial degree",
            )
        if self.cycle_type != tuple(sorted(self.cycle_type, reverse=True)):
            raise _validation_error(
                "cycle_type_not_canonical",
                "cycle type must be sorted in descending order",
            )
        if self.is_irreducible != (self.cycle_type == (self.degree,)):
            raise _validation_error(
                "cycle_type_irreducibility_mismatch",
                "irreducibility must agree with the cycle type",
            )
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
            raise _validation_error(
                "root_axis_not_unique", "root axis entries must be unique"
            )
        expected = tuple(range(len(self.root_axis)))
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
    method: str = "SYMPY_GALOIS_GROUP"

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
        if self.degree != len(self.group.root_axis):
            raise _validation_error(
                "group_degree_mismatch",
                "group root axis must match the polynomial degree",
            )
        return self


class SolvableResult(StrictModel):
    solvable_by_radicals: bool
    group: FinitePermutationGroup
    method: str = "GALOIS_GROUP_SOLVABILITY"

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

    @model_validator(mode="after")
    def require_group_certificate(self) -> Self:
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
