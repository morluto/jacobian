"""Bounded finite-field and integral simplicial-homology contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._digest import Sha256Digest
from jacobian._models import StrictModel
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_CHAIN_GROUP,
    MAX_TOPOLOGY_DIMENSION,
    MAX_TOPOLOGY_PRIME,
    FiniteSimplicialComplex,
    HomologyConvention,
    _validation_error,
    is_bounded_prime,
)
from jacobian.math.topology.chain_complexes.values import (
    CoefficientRing,
    HomologyResult,
)

MAX_INLINE_HOMOLOGY_CHAIN_GROUP = 64


class SimplicialHomologyRequest(StrictModel):
    complex: FiniteSimplicialComplex
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention = HomologyConvention.UNREDUCED


class ModularVector(StrictModel):
    coefficients: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )


class HomologyGroupResult(StrictModel):
    dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    chain_dimension: StrictInt = Field(ge=1, le=MAX_TOPOLOGY_CHAIN_GROUP)
    outgoing_boundary_rank: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    cycle_dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    incoming_boundary_rank: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    betti_number: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)
    cycle_basis: tuple[ModularVector, ...] = Field(
        default=(),
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )
    boundary_basis: tuple[ModularVector, ...] = Field(
        default=(),
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )
    homology_basis: tuple[ModularVector, ...] = Field(
        default=(),
        max_length=MAX_TOPOLOGY_CHAIN_GROUP,
    )
    quotient_span_rank: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_CHAIN_GROUP)

    @model_validator(mode="after")
    def require_dimension_ledger(self) -> Self:
        if self.cycle_dimension != (self.chain_dimension - self.outgoing_boundary_rank):
            raise _validation_error(
                "topology.require_dimension_ledger_1",
                "cycle dimension does not equal nullity",
            )
        if self.betti_number != (self.cycle_dimension - self.incoming_boundary_rank):
            raise _validation_error(
                "topology.require_dimension_ledger_2",
                "Betti number does not equal dim cycles minus boundaries",
            )
        if (
            len(self.cycle_basis) != self.cycle_dimension
            or len(self.boundary_basis) != self.incoming_boundary_rank
            or len(self.homology_basis) != self.betti_number
            or self.quotient_span_rank != self.cycle_dimension
        ):
            raise _validation_error(
                "topology.require_dimension_ledger_3",
                "homology bases do not match the dimension ledger",
            )
        vectors = (
            *self.cycle_basis,
            *self.boundary_basis,
            *self.homology_basis,
        )
        if any(len(vector.coefficients) != self.chain_dimension for vector in vectors):
            raise _validation_error(
                "topology.require_dimension_ledger_4",
                "homology vector does not use the declared chain basis",
            )
        return self


class SimplicialHomologyResult(StrictModel):
    complex: FiniteSimplicialComplex
    coefficient_field: Literal["PRIME_FIELD"] = "PRIME_FIELD"
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention
    orientation_convention: Literal["LEXICOGRAPHIC_VERTEX_ORDER"] = (
        "LEXICOGRAPHIC_VERTEX_ORDER"
    )
    dimension_range: tuple[StrictInt, StrictInt]
    groups: tuple[HomologyGroupResult, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )

    @model_validator(mode="after")
    def require_complete_dimension_range(self) -> Self:
        if not is_bounded_prime(self.prime):
            raise _validation_error(
                "topology.require_complete_dimension_range_1",
                "homology result requires a bounded prime",
            )
        dimensions = tuple(group.dimension for group in self.groups)
        if dimensions != tuple(range(len(self.groups))):
            raise _validation_error(
                "topology.require_complete_dimension_range_2",
                "homology groups must cover contiguous dimensions",
            )
        if self.dimension_range != (0, len(self.groups) - 1):
            raise _validation_error(
                "topology.require_complete_dimension_range_3",
                "dimension_range does not cover every returned group",
            )
        if any(
            coefficient < 0 or coefficient >= self.prime
            for group in self.groups
            for vector in (
                *group.cycle_basis,
                *group.boundary_basis,
                *group.homology_basis,
            )
            for coefficient in vector.coefficients
        ):
            raise _validation_error(
                "topology.require_complete_dimension_range_4",
                "homology vector coefficient is outside the prime field",
            )
        return self

    @property
    def complex_digest(self) -> Sha256Digest:
        """Compatibility projection of the retained source complex digest."""

        return self.complex.complex_digest


class IntegralSimplicialHomologyRequest(StrictModel):
    complex: FiniteSimplicialComplex
    convention: HomologyConvention = HomologyConvention.UNREDUCED


class IntegralSimplicialHomologyResult(StrictModel):
    complex: FiniteSimplicialComplex
    convention: HomologyConvention
    orientation_convention: Literal["LEXICOGRAPHIC_VERTEX_ORDER"] = (
        "LEXICOGRAPHIC_VERTEX_ORDER"
    )
    homology: HomologyResult

    @model_validator(mode="after")
    def require_integral_chain_owned_result(self) -> Self:
        expected_min = -1 if self.convention is HomologyConvention.REDUCED else 0
        if (
            self.homology.coefficient_ring is not CoefficientRing.INTEGER
            or self.homology.degree_min != expected_min
        ):
            raise _validation_error(
                "topology.integral_homology_context_mismatch",
                "simplicial integral homology must retain the chain-owned ZZ result under the selected convention",
            )
        return self

    @property
    def complex_digest(self) -> Sha256Digest:
        """Compatibility projection of the retained source complex digest."""

        return self.complex.complex_digest


__all__ = [
    "MAX_INLINE_HOMOLOGY_CHAIN_GROUP",
    "HomologyGroupResult",
    "IntegralSimplicialHomologyRequest",
    "IntegralSimplicialHomologyResult",
    "ModularVector",
    "SimplicialHomologyRequest",
    "SimplicialHomologyResult",
]
