"""Contracts owned by numerical-semigroup element-invariant kernels."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.number_theory.numerical_semigroups._models import (
    _GENERAL_ELEMENT_ENVELOPE,
    _GENERAL_GENERATOR_ENVELOPE,
    MAX_GENERATORS,
    _require_bounded_value,
    _require_canonical_minimal_axis,
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


class ElementDeltaSetResult(StrictModel):
    """Delta set of one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    factorization_lengths: tuple[int, ...]
    delta_set: tuple[int, ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        value: CanonicalInteger,
        minimal_generators: tuple[CanonicalInteger, ...],
        factorization_lengths: tuple[int, ...],
        delta_set: tuple[int, ...],
    ) -> Self:
        """Construct output derived from one admitted length-set kernel call."""

        return cls.model_construct(
            value=value,
            minimal_generators=minimal_generators,
            factorization_lengths=factorization_lengths,
            delta_set=delta_set,
        )

    @model_validator(mode="after")
    def require_set_semantics(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        _require_bounded_value(generators, self.value)
        if self.factorization_lengths != tuple(
            sorted(set(self.factorization_lengths))
        ) or any(length < 0 for length in self.factorization_lengths):
            raise _validation_error(
                "factorization_lengths must be non-negative, increasing, and duplicate-free"
            )
        if self.delta_set != tuple(sorted(set(self.delta_set))):
            raise _validation_error(
                "delta_set must be strictly increasing and duplicate-free"
            )
        if any(delta <= 0 for delta in self.delta_set):
            raise _validation_error("delta values must be positive")
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


class ElementElasticityResult(StrictModel):
    """Elasticity of one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    minimum_length: int = Field(ge=1)
    maximum_length: int = Field(ge=1)
    elasticity: str

    @classmethod
    def _from_kernel(
        cls,
        *,
        value: CanonicalInteger,
        minimal_generators: tuple[CanonicalInteger, ...],
        minimum_length: int,
        maximum_length: int,
        elasticity: str,
    ) -> Self:
        """Construct output derived from one admitted extrema kernel call."""

        return cls.model_construct(
            value=value,
            minimal_generators=minimal_generators,
            minimum_length=minimum_length,
            maximum_length=maximum_length,
            elasticity=elasticity,
        )

    @model_validator(mode="after")
    def require_length_ratio(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        _require_bounded_value(generators, self.value)
        if self.minimum_length > self.maximum_length:
            raise _validation_error("minimum_length must not exceed maximum_length")
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


class ElementCatenaryDegreeResult(StrictModel):
    """Catenary degree of one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    factorization_count: int = Field(ge=1)
    catenary_degree: int = Field(ge=0)

    @classmethod
    def _from_kernel(
        cls,
        *,
        value: CanonicalInteger,
        minimal_generators: tuple[CanonicalInteger, ...],
        factorization_count: int,
        catenary_degree: int,
    ) -> Self:
        """Construct an output established from one complete factorization family."""

        return cls.model_construct(
            value=value,
            minimal_generators=minimal_generators,
            factorization_count=factorization_count,
            catenary_degree=catenary_degree,
        )

    @model_validator(mode="after")
    def require_structural_degree(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        _require_bounded_value(generators, self.value)
        return self


__all__ = [
    "ElementCatenaryDegreeRequest",
    "ElementCatenaryDegreeResult",
    "ElementDeltaSetRequest",
    "ElementDeltaSetResult",
    "ElementElasticityRequest",
    "ElementElasticityResult",
]
