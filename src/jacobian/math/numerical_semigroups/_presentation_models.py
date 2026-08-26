"""Contracts owned by numerical-semigroup presentation kernels."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.numerical_semigroups._algorithms import betti_data
from jacobian.math.numerical_semigroups._models import (
    _GENERAL_GENERATOR_ENVELOPE,
    MAX_GENERATORS,
    _betti_component_index,
    _edges_span,
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

    @model_validator(mode="after")
    def require_minimal_relation_counts(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        _, _, disconnected = betti_data(generators)
        if tuple(map(parse_canonical_integer, self.betti_elements)) != tuple(
            disconnected
        ):
            raise _validation_error(
                "betti_elements do not match the minimal generators"
            )
        relation_components: dict[int, list[tuple[int, int]]] = {
            betti: [] for betti in disconnected
        }
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
            if first_degree != second_degree or first_degree not in relation_components:
                raise _validation_error("relation is not bound to a Betti element")
            components = disconnected[first_degree]
            left = _betti_component_index(relation.first, components)
            right = _betti_component_index(relation.second, components)
            if left == right:
                raise _validation_error(
                    "relation must connect distinct Betti components"
                )
            relation_components[first_degree].append((left, right))
        expected = {
            betti: len(components) - 1 for betti, components in disconnected.items()
        }
        if {
            betti: len(edges) for betti, edges in relation_components.items()
        } != expected:
            raise _validation_error(
                "relations do not have minimal per-Betti cardinality"
            )
        if any(
            not _edges_span(len(disconnected[betti]), edges)
            for betti, edges in relation_components.items()
        ):
            raise _validation_error("relations must span all Betti components")
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
