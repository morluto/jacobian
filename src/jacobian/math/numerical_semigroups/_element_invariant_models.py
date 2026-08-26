"""Contracts owned by numerical-semigroup element-invariant kernels."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.numerical_semigroups._algorithms import (
    catenary_degree_from_factorizations,
    factorization_length_extrema,
    factorization_lengths,
    factorizations,
)
from jacobian.math.numerical_semigroups._models import (
    _GENERAL_ELEMENT_ENVELOPE,
    _GENERAL_GENERATOR_ENVELOPE,
    MAX_GENERATORS,
    MAX_GRAPH_FACTORIZATIONS,
    _require_bounded_value,
    _require_canonical_minimal_axis,
    _require_materializable_factorizations,
    _require_member,
    _require_minimal_generators,
    _validation_error,
)


class ElementDeltaSetRequest(StrictModel):
    """Delta set of one element in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; factorization "
            "lengths use its increasing minimal generator axis."
        ),
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_semigroup_element(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(generators, self.value)
        _require_member(generators, value)
        return self


class ElementDeltaSetResult(StrictModel):
    """Delta set of one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    factorization_lengths: tuple[int, ...]
    delta_set: tuple[int, ...]

    @model_validator(mode="after")
    def require_set_semantics(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        value = parse_canonical_integer(self.value)
        expected_lengths = factorization_lengths(generators, value)
        if self.factorization_lengths != expected_lengths:
            raise _validation_error("factorization_lengths do not match the element")
        expected_delta = tuple(
            sorted({right - left for left, right in pairwise(expected_lengths)})
        )
        if self.delta_set != tuple(sorted(set(self.delta_set))):
            raise _validation_error(
                "delta_set must be strictly increasing and duplicate-free"
            )
        if any(delta <= 0 for delta in self.delta_set):
            raise _validation_error("delta values must be positive")
        if self.delta_set != expected_delta:
            raise _validation_error("delta_set does not match the complete length set")
        return self


class ElementElasticityRequest(StrictModel):
    """Elasticity of one element in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; results use its "
            "increasing minimal generator axis."
        ),
    )
    value: CanonicalInteger = Field(
        description="Positive semigroup element. " + _GENERAL_ELEMENT_ENVELOPE
    )

    @model_validator(mode="after")
    def require_nonzero_semigroup_element(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(generators, self.value)
        if value <= 0:
            raise _validation_error(
                "elasticity is defined here only for positive elements"
            )
        _require_member(generators, value)
        return self


class ElementElasticityResult(StrictModel):
    """Elasticity of one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    minimum_length: int = Field(ge=1)
    maximum_length: int = Field(ge=1)
    elasticity: str

    @model_validator(mode="after")
    def require_length_ratio(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        expected_extrema = factorization_length_extrema(
            generators, parse_canonical_integer(self.value)
        )
        if (self.minimum_length, self.maximum_length) != expected_extrema:
            raise _validation_error("length extrema do not match the element")
        if Fraction(self.elasticity) != Fraction(
            self.maximum_length, self.minimum_length
        ):
            raise _validation_error("elasticity does not match the length ratio")
        return self


class ElementCatenaryDegreeRequest(StrictModel):
    """Catenary degree of one element in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; factorization "
            "coordinates use its increasing minimal generator axis."
        ),
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_exact_bounded_element(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(generators, self.value)
        _require_member(generators, value)
        _require_materializable_factorizations(
            generators, value, MAX_GRAPH_FACTORIZATIONS
        )
        return self


class ElementCatenaryDegreeResult(StrictModel):
    """Catenary degree of one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    factorization_count: int = Field(ge=1)
    catenary_degree: int = Field(ge=0)

    @model_validator(mode="after")
    def require_exact_degree(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        family = factorizations(generators, parse_canonical_integer(self.value))
        if self.factorization_count != len(family):
            raise _validation_error("factorization_count does not match the element")
        if self.catenary_degree != catenary_degree_from_factorizations(family):
            raise _validation_error(
                "catenary_degree does not match the factorization graph"
            )
        return self


__all__ = [
    "ElementCatenaryDegreeRequest",
    "ElementCatenaryDegreeResult",
    "ElementDeltaSetRequest",
    "ElementDeltaSetResult",
    "ElementElasticityRequest",
    "ElementElasticityResult",
]
