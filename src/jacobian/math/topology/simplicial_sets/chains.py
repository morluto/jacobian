"""Exact unnormalized chains of a finite table-based simplicial set."""

from __future__ import annotations

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes.values import (
    MAX_OPERATION_MATRIX_CELLS,
    ChainCoefficient,
    ChainComplexValue,
    CoefficientRing,
    require_prime_field_admission,
)
from jacobian.math.topology.simplicial_sets._models import (
    MAX_TOTAL_SIMPLICES,
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables

MAX_UNNORMALIZED_CHAIN_OUTPUT_CELLS = 256_000
# Retained as the byte-named private admission lane used by composed operations.
MAX_UNNORMALIZED_CHAIN_OUTPUT_BYTES = MAX_UNNORMALIZED_CHAIN_OUTPUT_CELLS
_CHAIN_RESULT_STRUCTURAL_CELLS = 4_096


class UnnormalizedChainsRequest(StrictModel):
    """A checked finite simplicial-set prefix through degree N."""

    simplicial_set: FiniteTruncatedSimplicialSet
    coefficient_ring: CoefficientRing = CoefficientRing.INTEGER
    prime: StrictInt | None = Field(default=None, ge=2, le=1_000_003)

    @model_validator(mode="after")
    def require_ring_and_prime_coupling(self) -> UnnormalizedChainsRequest:
        if self.coefficient_ring is CoefficientRing.PRIME_FIELD:
            if self.prime is None:
                raise ValueError("GF_p coefficients require a prime modulus")
        elif self.prime is not None:
            raise ValueError("ZZ and QQ coefficients do not take a prime modulus")
        return self


class UnnormalizedChainsResult(StrictModel):
    """An exact chain complex with its degreewise simplex basis transport."""

    simplicial_set: FiniteTruncatedSimplicialSet
    simplex_bases: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=5)
    chain_complex: ChainComplexValue

    @model_validator(mode="after")
    def require_source_bound_chain_axes(self) -> UnnormalizedChainsResult:
        if self.simplex_bases != self.simplicial_set.sets:
            raise ValueError("simplex bases must retain the source degree axes")
        value = self.chain_complex
        if (
            value.degree_min != 0
            or value.degree_max != self.simplicial_set.max_degree
            or value.basis_sizes
            != tuple(len(level) for level in self.simplicial_set.sets)
        ):
            raise ValueError("chain complex must use the source unnormalized bases")
        return self


def _estimate_output_cells(
    source: FiniteTruncatedSimplicialSet, sizes: tuple[int, ...]
) -> int:
    """Bound retained labels, matrix entries, and structural output cells."""
    cells = sum(sizes[n - 1] * sizes[n] for n in range(1, len(sizes)))
    label_chars = sum(len(label) for level in source.sets for label in level)
    return _CHAIN_RESULT_STRUCTURAL_CELLS + label_chars * 2 + 12 * cells


def _estimate_output_bytes(
    source: FiniteTruncatedSimplicialSet, sizes: tuple[int, ...]
) -> int:
    return _estimate_output_cells(source, sizes)


def _preflight(
    source: FiniteTruncatedSimplicialSet,
    *,
    additional_matrix_cells: int = 0,
    additional_output_bytes: int = 0,
    output_name: str = "unnormalized chain",
    output_error_code: str = "simplicial_set.unnormalized_chain_output_budget_exceeded",
) -> int:
    sizes = tuple(len(level) for level in source.sets)
    cells = sum(sizes[n - 1] * sizes[n] for n in range(1, len(sizes)))
    total_cells = cells + additional_matrix_cells
    if total_cells > MAX_OPERATION_MATRIX_CELLS:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.unnormalized_chain_matrix_budget_exceeded",
            message=(
                f"{output_name} matrices require {total_cells} cells, exceeding "
                f"the {MAX_OPERATION_MATRIX_CELLS}-cell construction bound"
            ),
        )
    estimate = _estimate_output_bytes(source, sizes) + additional_output_bytes
    output_limit = min(
        MAX_UNNORMALIZED_CHAIN_OUTPUT_BYTES, MAX_UNNORMALIZED_CHAIN_OUTPUT_CELLS
    )
    if estimate > output_limit:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code=output_error_code,
            message=(
                f"estimated chain result size {estimate} cells exceeds the "
                f"{output_limit}-cell output bound"
            ),
        )
    if source.total_simplices > MAX_TOTAL_SIMPLICES:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.unnormalized_chain_simplex_budget_exceeded",
            message="source simplex count exceeds the chain-construction bound",
        )
    return cells


def _checked_simplicial_set(
    source: FiniteTruncatedSimplicialSet,
) -> FiniteTruncatedSimplicialSet:
    checked = from_tables(
        source.max_degree, source.sets, source.face_maps, source.degeneracy_maps
    )
    if checked.status != "SIMPLICIAL_SET" or checked.simplicial_set is None:
        obstruction = checked.obstruction
        raise OperationDomainValidationError(
            location=("simplicial_set",),
            code="simplicial_set.unnormalized_source_invalid",
            message=(
                "source tables fail a visible simplicial identity"
                if obstruction is None
                else (
                    f"source fails {obstruction.left_description} = {obstruction.right_description} "
                    f"at degree {obstruction.degree}, simplex {obstruction.row}"
                )
            ),
        )
    return checked.simplicial_set


def _unnormalized_chains_from_checked_source(
    request: UnnormalizedChainsRequest, source: FiniteTruncatedSimplicialSet
) -> UnnormalizedChainsResult:
    sizes = tuple(len(level) for level in source.sets)
    matrices: list[tuple[tuple[ChainCoefficient, ...], ...]] = []
    for degree in range(1, source.max_degree + 1):
        matrix = [[0] * sizes[degree] for _ in range(sizes[degree - 1])]
        for face_index, face in enumerate(source.face_maps[degree - 1]):
            sign = 1 if face_index % 2 == 0 else -1
            for column, row in enumerate(face):
                matrix[row][column] += sign
        if request.coefficient_ring is CoefficientRing.PRIME_FIELD:
            prime = request.prime
            if prime is None:
                raise RuntimeError("GF_p coefficients require a prime modulus")
            matrix = [[entry % prime for entry in row] for row in matrix]
        matrices.append(tuple(tuple(entry for entry in row) for row in matrix))
    value = ChainComplexValue(
        coefficient_ring=request.coefficient_ring,
        prime=request.prime,
        degree_min=0,
        degree_max=source.max_degree,
        basis_sizes=sizes,
        differential_matrices=tuple(matrices),
    )
    return UnnormalizedChainsResult(
        simplicial_set=source, simplex_bases=source.sets, chain_complex=value
    )


def unnormalized_chains(
    request: UnnormalizedChainsRequest,
) -> UnnormalizedChainsResult:
    require_prime_field_admission(request.coefficient_ring, request.prime)
    source = request.simplicial_set
    _preflight(source)
    # Serialized source values do not carry trusted producer provenance. Since
    # d^2=0 depends on simplicial identities, re-establish those caller claims
    # once before constructing the chain value.
    checked = _checked_simplicial_set(source)
    return _unnormalized_chains_from_checked_source(request, checked)


__all__ = [
    "UnnormalizedChainsRequest",
    "UnnormalizedChainsResult",
    "unnormalized_chains",
]
