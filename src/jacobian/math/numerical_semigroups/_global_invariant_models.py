"""Contracts owned by numerical-semigroup global-invariant kernels."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.numerical_semigroups._algorithms import (
    betti_data,
    catenary_degree_from_factorizations,
    delta_periodicity_bound,
    factorizations,
)
from jacobian.math.numerical_semigroups._models import (
    _GENERAL_GENERATOR_ENVELOPE,
    MAX_GENERATORS,
    _require_canonical_minimal_axis,
    _require_global_betti_bound,
    _require_global_catenary_bound,
    _require_global_delta_bound,
    _require_minimal_generators,
    _validation_error,
)


class BettiElementsRequest(StrictModel):
    """Betti elements of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; derived data uses its increasing minimal generator axis."
        ),
    )

    @model_validator(mode="after")
    def require_complete_candidate_range(self) -> Self:
        _require_global_betti_bound(_require_minimal_generators(self.generators))
        return self


class BettiElementsResult(StrictModel):
    """Betti elements of a semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    apery_set: tuple[CanonicalInteger, ...]
    candidate_count: int = Field(ge=0)
    betti_elements: tuple[CanonicalInteger, ...]

    @model_validator(mode="after")
    def require_complete_betti_data(self) -> Self:
        apery, candidates, disconnected = betti_data(
            _require_canonical_minimal_axis(self.minimal_generators)
        )
        if tuple(map(parse_canonical_integer, self.apery_set)) != apery:
            raise _validation_error("apery_set does not match the minimal generators")
        if self.candidate_count != len(candidates):
            raise _validation_error("candidate_count does not match the complete range")
        if tuple(map(parse_canonical_integer, self.betti_elements)) != tuple(
            disconnected
        ):
            raise _validation_error(
                "betti_elements do not match disconnected candidates"
            )
        return self


class DeltaSetRequest(StrictModel):
    """Global delta set of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; the complete delta set uses its increasing minimal generator axis."
        ),
    )

    @model_validator(mode="after")
    def require_complete_periodicity_range(self) -> Self:
        _require_global_delta_bound(_require_minimal_generators(self.generators))
        return self


class DeltaSetResult(StrictModel):
    """Global delta set of the semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    delta_set: tuple[int, ...]
    periodicity_bound: int = Field(ge=0)
    checked_through: int = Field(ge=0)
    completeness_basis: Literal["EVENTUAL_PERIODICITY_BOUND"] = (
        "EVENTUAL_PERIODICITY_BOUND"
    )

    @model_validator(mode="after")
    def require_set_semantics(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        if self.delta_set != tuple(sorted(set(self.delta_set))):
            raise _validation_error(
                "delta_set must be strictly increasing and duplicate-free"
            )
        if any(delta <= 0 for delta in self.delta_set):
            raise _validation_error("delta values must be positive")
        expected_bound = delta_periodicity_bound(generators)
        if self.periodicity_bound != expected_bound:
            raise _validation_error("periodicity_bound does not match the generators")
        if self.checked_through != expected_bound + generators[-1] - 1:
            raise _validation_error(
                "checked_through does not match the completeness theorem"
            )
        return self


class ElasticityRequest(StrictModel):
    """Global elasticity of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; the ratio uses its increasing minimal generator axis."
        ),
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_minimal_generators(self.generators)
        return self


class ElasticityResult(StrictModel):
    """Global elasticity of the semigroup."""

    elasticity: str
    smallest_generator: CanonicalInteger
    largest_generator: CanonicalInteger

    @model_validator(mode="after")
    def require_generator_ratio(self) -> Self:
        smallest = parse_canonical_integer(self.smallest_generator)
        largest = parse_canonical_integer(self.largest_generator)
        if smallest > largest:
            raise _validation_error("generator extrema are reversed")
        if Fraction(self.elasticity) != Fraction(largest, smallest):
            raise _validation_error("elasticity must equal largest/smallest generator")
        return self


class CatenaryDegreeRequest(StrictModel):
    """Global catenary degree of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; factorization coordinates use its increasing minimal generator axis."
        ),
    )

    @model_validator(mode="after")
    def require_complete_betti_graphs(self) -> Self:
        _require_global_catenary_bound(_require_minimal_generators(self.generators))
        return self


class BettiCatenaryDegree(StrictModel):
    """Catenary degree witnessed at one Betti element."""

    betti_element: CanonicalInteger
    catenary_degree: int = Field(ge=0)


class CatenaryDegreeResult(StrictModel):
    """Global catenary degree of the semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    catenary_degree: int = Field(ge=0)
    betti_degrees: tuple[BettiCatenaryDegree, ...]
    witness_betti_elements: tuple[CanonicalInteger, ...]

    @model_validator(mode="after")
    def require_maximizing_witnesses(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        _, _, disconnected = betti_data(generators)
        expected_records = tuple(
            BettiCatenaryDegree(
                betti_element=str(value),
                catenary_degree=catenary_degree_from_factorizations(
                    factorizations(generators, value)
                ),
            )
            for value in disconnected
        )
        if self.betti_degrees != expected_records:
            raise _validation_error("betti_degrees do not match the complete Betti set")
        maximum = max(
            (record.catenary_degree for record in self.betti_degrees), default=0
        )
        if self.catenary_degree != maximum:
            raise _validation_error("global catenary degree must be the Betti maximum")
        expected = tuple(
            record.betti_element
            for record in self.betti_degrees
            if maximum > 0 and record.catenary_degree == maximum
        )
        if self.witness_betti_elements != expected:
            raise _validation_error(
                "witnesses must be exactly the maximizing Betti elements"
            )
        return self


__all__ = [
    name
    for name in globals()
    if name
    in {
        "BettiCatenaryDegree",
        "BettiElementsRequest",
        "BettiElementsResult",
        "CatenaryDegreeRequest",
        "CatenaryDegreeResult",
        "DeltaSetRequest",
        "DeltaSetResult",
        "ElasticityRequest",
        "ElasticityResult",
    }
]
