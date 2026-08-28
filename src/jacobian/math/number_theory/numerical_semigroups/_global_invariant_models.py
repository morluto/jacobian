"""Contracts owned by numerical-semigroup global-invariant kernels."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory.numerical_semigroups._models import (
    _GENERAL_GENERATOR_ENVELOPE,
    MAX_GENERATORS,
    _require_canonical_minimal_axis,
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


class BettiElementsResult(StrictModel):
    """Betti elements of a semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    apery_set: tuple[CanonicalInteger, ...]
    candidate_count: int = Field(ge=0)
    betti_elements: tuple[CanonicalInteger, ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        minimal_generators: tuple[CanonicalInteger, ...],
        apery_set: tuple[CanonicalInteger, ...],
        candidate_count: int,
        betti_elements: tuple[CanonicalInteger, ...],
    ) -> Self:
        """Construct output established by one admitted Betti-data kernel call."""

        return cls.model_construct(
            minimal_generators=minimal_generators,
            apery_set=apery_set,
            candidate_count=candidate_count,
            betti_elements=betti_elements,
        )

    @model_validator(mode="after")
    def require_complete_betti_data(self) -> Self:
        _require_canonical_minimal_axis(self.minimal_generators)
        if self.betti_elements != tuple(
            sorted(set(self.betti_elements), key=parse_canonical_integer)
        ):
            raise _validation_error(
                "betti_elements must be increasing and duplicate-free"
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

    @classmethod
    def _from_kernel(
        cls,
        *,
        minimal_generators: tuple[CanonicalInteger, ...],
        delta_set: tuple[int, ...],
        periodicity_bound: int,
        checked_through: int,
    ) -> Self:
        """Construct output derived from one admitted periodicity computation."""

        return cls.model_construct(
            minimal_generators=minimal_generators,
            delta_set=delta_set,
            periodicity_bound=periodicity_bound,
            checked_through=checked_through,
        )

    @model_validator(mode="after")
    def require_set_semantics(self) -> Self:
        _require_canonical_minimal_axis(self.minimal_generators)
        if self.delta_set != tuple(sorted(set(self.delta_set))):
            raise _validation_error(
                "delta_set must be strictly increasing and duplicate-free"
            )
        if any(delta <= 0 for delta in self.delta_set):
            raise _validation_error("delta values must be positive")
        if self.checked_through < self.periodicity_bound:
            raise _validation_error(
                "checked_through must include the periodicity bound"
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

    @classmethod
    def _from_kernel(
        cls,
        *,
        minimal_generators: tuple[CanonicalInteger, ...],
        catenary_degree: int,
        betti_degrees: tuple[BettiCatenaryDegree, ...],
        witness_betti_elements: tuple[CanonicalInteger, ...],
    ) -> Self:
        """Construct output established from one admitted Betti pass."""

        return cls.model_construct(
            minimal_generators=minimal_generators,
            catenary_degree=catenary_degree,
            betti_degrees=betti_degrees,
            witness_betti_elements=witness_betti_elements,
        )

    @model_validator(mode="after")
    def require_maximizing_witnesses(self) -> Self:
        _require_canonical_minimal_axis(self.minimal_generators)
        if tuple(record.betti_element for record in self.betti_degrees) != tuple(
            sorted(
                (record.betti_element for record in self.betti_degrees),
                key=parse_canonical_integer,
            )
        ):
            raise _validation_error("betti_degrees must be increasing by Betti element")
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
