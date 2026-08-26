"""Contracts owned by numerical-semigroup presentation kernels."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.numerical_semigroups._models import (
    _GENERAL_GENERATOR_ENVELOPE,
    MAX_GENERATORS,
    _require_canonical_minimal_axis,
    _require_global_betti_bound,
    _require_minimal_generators,
    _validation_error,
)


class MinimalPresentationRequest(StrictModel):
    """One minimal presentation of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; returned relations use its increasing minimal generator axis."
        ),
    )

    @model_validator(mode="after")
    def require_complete_candidate_range(self) -> Self:
        _require_global_betti_bound(_require_minimal_generators(self.generators))
        return self


class MinimalPresentationRelation(StrictModel):
    """One relation (pair of distinct factorizations) in a presentation."""

    first: tuple[int, ...]
    second: tuple[int, ...]

    @model_validator(mode="after")
    def require_distinct_nonnegative_factorizations(self) -> Self:
        if len(self.first) != len(self.second):
            raise _validation_error("relation factorizations must have equal arity")
        if any(value < 0 for value in (*self.first, *self.second)):
            raise _validation_error("relation factorizations must be non-negative")
        if self.first == self.second:
            raise _validation_error("relation factorizations must be distinct")
        return self


class MinimalPresentationResult(StrictModel):
    """One minimal presentation of the semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    betti_elements: tuple[CanonicalInteger, ...]
    relations: tuple[MinimalPresentationRelation, ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        minimal_generators: tuple[CanonicalInteger, ...],
        betti_elements: tuple[CanonicalInteger, ...],
        relations: tuple[MinimalPresentationRelation, ...],
    ) -> Self:
        """Construct a minimal presentation derived by the admitted kernel."""

        return cls.model_construct(
            minimal_generators=minimal_generators,
            betti_elements=betti_elements,
            relations=relations,
        )

    @model_validator(mode="after")
    def require_minimal_relation_counts(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        for relation in self.relations:
            if len(relation.first) != len(generators):
                raise _validation_error(
                    "relation arity does not match minimal generators"
                )
            first_degree = sum(
                coordinate * generator
                for coordinate, generator in zip(
                    relation.first, generators, strict=True
                )
            )
            second_degree = sum(
                coordinate * generator
                for coordinate, generator in zip(
                    relation.second, generators, strict=True
                )
            )
            if first_degree != second_degree:
                raise _validation_error("relation terms must have the same degree")
        return self


class PresentationBinomialsRequest(StrictModel):
    """Convert a minimal presentation to sparse binomial form."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; relation coordinates must use its increasing minimal generator axis."
        ),
    )
    relations: tuple[MinimalPresentationRelation, ...]

    @model_validator(mode="after")
    def require_kernel_relations(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        for relation in self.relations:
            if len(relation.first) != len(generators):
                raise _validation_error(
                    "relation coordinates must match the minimal generating system"
                )
            if sum(
                coefficient * generator
                for coefficient, generator in zip(
                    relation.first, generators, strict=True
                )
            ) != sum(
                coefficient * generator
                for coefficient, generator in zip(
                    relation.second, generators, strict=True
                )
            ):
                raise _validation_error(
                    "relation factorizations must have the same semigroup degree"
                )
        return self


class PresentationBinomial(StrictModel):
    """One sparse binomial (aX - bX) arising from a presentation relation."""

    left_coefficient: Literal["1"] = "1"
    left_exponents: tuple[int, ...]
    right_coefficient: Literal["-1"] = "-1"
    right_exponents: tuple[int, ...]


class PresentationBinomialsResult(StrictModel):
    """Presentation converted to sparse binomials."""

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    binomials: tuple[PresentationBinomial, ...]

    @model_validator(mode="after")
    def require_canonical_axis_and_homogeneous_binomials(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        for binomial in self.binomials:
            if len(binomial.left_exponents) != len(generators) or len(
                binomial.right_exponents
            ) != len(generators):
                raise _validation_error(
                    "binomial exponents must match the minimal generator axis"
                )
            if any(
                exponent < 0
                for exponent in (*binomial.left_exponents, *binomial.right_exponents)
            ):
                raise _validation_error("binomial exponents must be non-negative")
            if sum(
                exponent * generator
                for exponent, generator in zip(
                    binomial.left_exponents, generators, strict=True
                )
            ) != sum(
                exponent * generator
                for exponent, generator in zip(
                    binomial.right_exponents, generators, strict=True
                )
            ):
                raise _validation_error(
                    "binomial terms must have the same semigroup degree"
                )
        return self


__all__ = [
    "MinimalPresentationRelation",
    "MinimalPresentationRequest",
    "MinimalPresentationResult",
    "PresentationBinomial",
    "PresentationBinomialsRequest",
    "PresentationBinomialsResult",
]
