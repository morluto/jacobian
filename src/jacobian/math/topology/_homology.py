"""Bounded finite-field and integral simplicial-homology contracts."""

from __future__ import annotations

from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._digest import Sha256Digest
from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.matrices.certified_snf.values import (
    MAX_CERTIFIED_SNF_DIMENSION,
    CertifiedIntegerMatrix,
    SmithNormalFormCertificate,
)
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_CHAIN_GROUP,
    MAX_TOPOLOGY_DIMENSION,
    MAX_TOPOLOGY_PRIME,
    FiniteSimplicialComplex,
    HomologyConvention,
    TopologyExactResult,
    _validation_error,
    is_bounded_prime,
    require_linear_algebra_bounds,
)

# Integral homology embeds every boundary/augmentation matrix, Smith
# transformation, and derived coordinate matrix in ``CertifiedIntegerMatrix``,
# whose dimension contract caps at ``MAX_CERTIFIED_SNF_DIMENSION``. The
# admitted chain groups therefore derive from that certificate bound; raising
# them requires expanding the certificate contract first, not this constant
# alone. The total-rank and cell bounds are independent work nets over the
# whole request (SNF cost scales with total chain size), not output-shape
# bounds.
MAX_INLINE_HOMOLOGY_CHAIN_GROUP = 64
MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP = MAX_CERTIFIED_SNF_DIMENSION
MAX_INTEGRAL_HOMOLOGY_TOTAL_CHAIN_RANK = 100
MAX_INTEGRAL_HOMOLOGY_MATRIX_CELLS = 2500
MAX_INTEGRAL_HOMOLOGY_OUTPUT_DIGITS = 256


class SimplicialHomologyRequest(StrictModel):
    complex: FiniteSimplicialComplex
    prime: StrictInt = Field(ge=2, le=MAX_TOPOLOGY_PRIME)
    convention: HomologyConvention = HomologyConvention.UNREDUCED

    @model_validator(mode="after")
    def require_prime_and_bounds(self) -> Self:
        if not is_bounded_prime(self.prime):
            raise _validation_error(
                "topology.require_prime_and_bounds_1",
                "homology coefficients require a bounded prime",
            )
        require_linear_algebra_bounds(self.complex)
        if any(
            size > MAX_INLINE_HOMOLOGY_CHAIN_GROUP for size in self.complex.f_vector
        ):
            raise _validation_error(
                "topology.require_prime_and_bounds_2",
                "inline homology bases require at most "
                f"{MAX_INLINE_HOMOLOGY_CHAIN_GROUP} simplices in each chain group",
            )
        return self


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


class SimplicialHomologyResult(TopologyExactResult):
    complex_digest: Sha256Digest
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


class IntegralSimplicialHomologyRequest(StrictModel):
    complex: FiniteSimplicialComplex
    convention: HomologyConvention = HomologyConvention.UNREDUCED

    @model_validator(mode="after")
    def require_integral_certificate_bounds(self) -> Self:
        require_linear_algebra_bounds(self.complex)
        if any(
            size > MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP for size in self.complex.f_vector
        ):
            raise _validation_error(
                "topology.require_integral_certificate_bounds_1",
                "integral homology requires at most "
                f"{MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP} simplices in each chain group",
            )
        if sum(self.complex.f_vector) > MAX_INTEGRAL_HOMOLOGY_TOTAL_CHAIN_RANK:
            raise _validation_error(
                "topology.require_integral_certificate_bounds_2",
                "integral homology requires total chain rank at most "
                f"{MAX_INTEGRAL_HOMOLOGY_TOTAL_CHAIN_RANK}",
            )
        padded = (0, *self.complex.f_vector)
        if any(
            rows * columns > MAX_INTEGRAL_HOMOLOGY_MATRIX_CELLS
            for rows, columns in pairwise(padded)
        ):
            raise _validation_error(
                "topology.require_integral_certificate_bounds_3",
                "integral homology boundary exceeds the "
                f"{MAX_INTEGRAL_HOMOLOGY_MATRIX_CELLS}-cell bound",
            )
        return self


class IntegralVector(StrictModel):
    coefficients: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )

    @model_validator(mode="after")
    def require_output_digit_budget(self) -> Self:
        if any(
            len(value.lstrip("-")) > MAX_INTEGRAL_HOMOLOGY_OUTPUT_DIGITS
            for value in self.coefficients
        ):
            raise _validation_error(
                "topology.require_output_digit_budget_1",
                "integral homology vector exceeds the output digit bound",
            )
        return self


class IntegralFreeGenerator(StrictModel):
    cycle: IntegralVector
    cycle_coordinates: IntegralVector


class IntegralTorsionGenerator(StrictModel):
    order: CanonicalInteger
    cycle: IntegralVector
    cycle_coordinates: IntegralVector
    bounding_chain: IntegralVector

    @model_validator(mode="after")
    def require_nontrivial_bounded_order(self) -> Self:
        if (
            int(self.order) <= 1
            or len(self.order.lstrip("-")) > MAX_INTEGRAL_HOMOLOGY_OUTPUT_DIGITS
        ):
            raise _validation_error(
                "topology.require_nontrivial_bounded_order_1",
                "torsion generator order must be a bounded integer > 1",
            )
        return self


class IntegralHomologyGroupResult(StrictModel):
    dimension: StrictInt = Field(ge=0, le=MAX_TOPOLOGY_DIMENSION)
    chain_dimension: StrictInt = Field(ge=1, le=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP)
    incoming_chain_dimension: StrictInt = Field(
        ge=0, le=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )
    outgoing_boundary_rank: StrictInt = Field(
        ge=0, le=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )
    cycle_rank: StrictInt = Field(ge=0, le=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP)
    incoming_boundary_rank: StrictInt = Field(
        ge=0, le=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )
    betti_number: StrictInt = Field(ge=0, le=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP)
    torsion_coefficients: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )
    free_generators: tuple[IntegralFreeGenerator, ...] = Field(
        max_length=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )
    torsion_generators: tuple[IntegralTorsionGenerator, ...] = Field(
        max_length=MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP
    )
    outgoing_smith_certificate: SmithNormalFormCertificate
    boundary_in_cycle_coordinates: CertifiedIntegerMatrix
    incoming_smith_certificate: SmithNormalFormCertificate
    generator_basis: Literal[
        "CANONICAL_SIMPLEX_BASIS_VIA_CERTIFIED_SMITH_TRANSFORMATIONS"
    ] = "CANONICAL_SIMPLEX_BASIS_VIA_CERTIFIED_SMITH_TRANSFORMATIONS"

    @model_validator(mode="after")
    def require_complete_integral_group_ledger(self) -> Self:
        outgoing = self.outgoing_smith_certificate
        incoming = self.incoming_smith_certificate
        if (
            outgoing.source.column_count != self.chain_dimension
            or outgoing.rank != self.outgoing_boundary_rank
            or self.cycle_rank != self.chain_dimension - self.outgoing_boundary_rank
            or (
                self.boundary_in_cycle_coordinates.row_count,
                self.boundary_in_cycle_coordinates.column_count,
            )
            != (self.cycle_rank, self.incoming_chain_dimension)
            or incoming.source != self.boundary_in_cycle_coordinates
            or incoming.rank != self.incoming_boundary_rank
            or self.betti_number != self.cycle_rank - self.incoming_boundary_rank
            or len(self.free_generators) != self.betti_number
        ):
            raise _validation_error(
                "topology.require_complete_integral_group_ledger_1",
                "integral homology rank and certificate ledger is invalid",
            )
        torsion = tuple(
            factor for factor in incoming.invariant_factors if int(factor) > 1
        )
        if (
            self.torsion_coefficients != torsion
            or tuple(item.order for item in self.torsion_generators) != torsion
        ):
            raise _validation_error(
                "topology.require_complete_integral_group_ledger_2",
                "integral homology torsion generators must match Smith factors",
            )
        if any(
            len(item.cycle.coefficients) != self.chain_dimension
            or len(item.cycle_coordinates.coefficients) != self.cycle_rank
            for item in self.free_generators
        ) or any(
            len(item.cycle.coefficients) != self.chain_dimension
            or len(item.cycle_coordinates.coefficients) != self.cycle_rank
            or len(item.bounding_chain.coefficients) != self.incoming_chain_dimension
            for item in self.torsion_generators
        ):
            raise _validation_error(
                "topology.require_complete_integral_group_ledger_3",
                "integral homology generators must use the declared simplex bases",
            )
        matrices = (
            outgoing.source,
            outgoing.diagonal,
            outgoing.left_transformation,
            outgoing.right_transformation,
            self.boundary_in_cycle_coordinates,
            incoming.source,
            incoming.diagonal,
            incoming.left_transformation,
            incoming.right_transformation,
        )
        scalar_values = (
            value for matrix in matrices for row in matrix.entries for value in row
        )
        if any(
            len(value.lstrip("-")) > MAX_INTEGRAL_HOMOLOGY_OUTPUT_DIGITS
            for value in scalar_values
        ):
            raise _validation_error(
                "topology.require_complete_integral_group_ledger_4",
                "integral homology certificate exceeds the output digit bound",
            )
        return self


class IntegralSimplicialHomologyResult(TopologyExactResult):
    complex_digest: Sha256Digest
    coefficient_ring: Literal["ZZ"] = "ZZ"
    convention: HomologyConvention
    orientation_convention: Literal["LEXICOGRAPHIC_VERTEX_ORDER"] = (
        "LEXICOGRAPHIC_VERTEX_ORDER"
    )
    dimension_range: tuple[StrictInt, StrictInt]
    groups: tuple[IntegralHomologyGroupResult, ...] = Field(
        min_length=1,
        max_length=MAX_TOPOLOGY_DIMENSION + 1,
    )
    completeness: Literal["FREE_TORSION_AND_BOUND_GENERATORS"] = (
        "FREE_TORSION_AND_BOUND_GENERATORS"
    )
    decomposition: Literal["DIRECT_SUM_Z_AND_FINITE_CYCLIC_FACTORS"] = (
        "DIRECT_SUM_Z_AND_FINITE_CYCLIC_FACTORS"
    )

    @model_validator(mode="after")
    def require_complete_integral_dimension_range(self) -> Self:
        dimensions = tuple(group.dimension for group in self.groups)
        if dimensions != tuple(range(len(self.groups))):
            raise _validation_error(
                "topology.require_complete_integral_dimension_range_1",
                "integral homology groups must cover contiguous dimensions",
            )
        if self.dimension_range != (0, len(self.groups) - 1):
            raise _validation_error(
                "topology.require_complete_integral_dimension_range_2",
                "integral homology dimension_range must cover every group",
            )
        return self


__all__ = [
    "MAX_INLINE_HOMOLOGY_CHAIN_GROUP",
    "MAX_INTEGRAL_HOMOLOGY_CHAIN_GROUP",
    "MAX_INTEGRAL_HOMOLOGY_MATRIX_CELLS",
    "MAX_INTEGRAL_HOMOLOGY_OUTPUT_DIGITS",
    "MAX_INTEGRAL_HOMOLOGY_TOTAL_CHAIN_RANK",
    "HomologyGroupResult",
    "IntegralFreeGenerator",
    "IntegralHomologyGroupResult",
    "IntegralSimplicialHomologyRequest",
    "IntegralSimplicialHomologyResult",
    "IntegralTorsionGenerator",
    "IntegralVector",
    "ModularVector",
    "SimplicialHomologyRequest",
    "SimplicialHomologyResult",
]
