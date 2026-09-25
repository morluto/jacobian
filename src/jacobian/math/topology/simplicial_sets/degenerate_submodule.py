"""Exact degenerate subcomplex of a finite simplicial-set chain prefix."""

from __future__ import annotations

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.chain_complexes.values import (
    ChainCoefficient,
    ChainComplexValue,
    require_prime_field_admission,
)
from jacobian.math.topology.simplicial_sets import chains as chains_module
from jacobian.math.topology.simplicial_sets._models import (
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets.chains import (
    UnnormalizedChainsRequest,
    UnnormalizedChainsResult,
)

MAX_DEGENERATE_SUBMODULE_WORK_UNITS = 20_000
_DEGENERATE_SUBMODULE_OUTPUT_OVERHEAD = 8_192


class DegenerateSubmoduleRequest(UnnormalizedChainsRequest):
    """A finite simplicial-set prefix and exact coefficient ring."""


class DegenerateSubmoduleResult(StrictModel):
    """The degenerate-chain subcomplex and its source-basis inclusions.

    Each column of ``inclusion_matrices[n]`` is the corresponding standard
    basis vector of the unnormalized group on ``simplex_bases[n]``. Its column
    order is the order of ``degenerate_basis_indices[n]``.
    """

    unnormalized_chains: UnnormalizedChainsResult
    degenerate_basis_indices: tuple[tuple[StrictInt, ...], ...] = Field(
        min_length=1, max_length=5
    )
    inclusion_matrices: tuple[tuple[tuple[ChainCoefficient, ...], ...], ...] = Field(
        min_length=1, max_length=5
    )
    degenerate_complex: ChainComplexValue

    @model_validator(mode="after")
    def require_source_axes(self) -> DegenerateSubmoduleResult:
        source = self.unnormalized_chains
        ambient = source.chain_complex
        degenerate = self.degenerate_complex
        if len(self.degenerate_basis_indices) != len(source.simplex_bases) or len(
            self.inclusion_matrices
        ) != len(source.simplex_bases):
            raise ValueError("degenerate bases must cover every source degree")
        if (
            degenerate.degree_min != ambient.degree_min
            or degenerate.degree_max != ambient.degree_max
            or degenerate.coefficient_ring is not ambient.coefficient_ring
            or degenerate.prime != ambient.prime
            or degenerate.basis_sizes
            != tuple(len(indices) for indices in self.degenerate_basis_indices)
        ):
            raise ValueError("degenerate complex must retain the ambient context")
        for degree, (indices, matrix) in enumerate(
            zip(
                self.degenerate_basis_indices,
                self.inclusion_matrices,
                strict=True,
            )
        ):
            ambient_size = ambient.basis_sizes[degree]
            if tuple(sorted(set(indices))) != indices or any(
                not 0 <= index < ambient_size for index in indices
            ):
                raise ValueError("degenerate indices must be increasing source indices")
            if len(matrix) != ambient_size or any(
                len(row) != len(indices) for row in matrix
            ):
                raise ValueError(
                    "inclusion matrix axes must match source and submodule"
                )
        return self


def _admit(
    request: DegenerateSubmoduleRequest,
) -> tuple[FiniteTruncatedSimplicialSet, tuple[tuple[int, ...], ...]]:
    """Check scalars/source and admit the combined result before matrices."""
    require_prime_field_admission(request.coefficient_ring, request.prime)
    source = chains_module._checked_simplicial_set(request.simplicial_set)
    basis_indices = _degenerate_indices(source)
    sizes = tuple(len(level) for level in source.sets)
    ambient_cells = sum(
        sizes[degree - 1] * sizes[degree] for degree in range(1, len(sizes))
    )
    ranks = tuple(len(indices) for indices in basis_indices)
    inclusion_cells = sum(size * rank for size, rank in zip(sizes, ranks, strict=True))
    degenerate_boundary_cells = sum(
        ranks[degree - 1] * ranks[degree] for degree in range(1, len(ranks))
    )
    degeneracy_rows = sum(
        (degree + 1) * sizes[degree] for degree in range(max(0, source.max_degree))
    )
    face_rows = sum((degree + 1) * sizes[degree] for degree in range(1, len(sizes)))
    restricted_boundary_rows = sum(
        sizes[degree - 1] * ranks[degree] for degree in range(1, len(sizes))
    )
    work = (
        degeneracy_rows
        + face_rows
        + ambient_cells
        + inclusion_cells
        + restricted_boundary_rows
        + degenerate_boundary_cells
    )
    if work > MAX_DEGENERATE_SUBMODULE_WORK_UNITS:
        _admission_error(
            "degenerate_submodule_work_budget_exceeded",
            f"degenerate-submodule construction requires {work} work units, "
            f"exceeding {MAX_DEGENERATE_SUBMODULE_WORK_UNITS}",
        )

    # The shared estimate includes the source tables, the repeated serialized
    # simplex_bases labels (ASCII-escaped at their 32-character bound), and the
    # ambient differential. Add exact-rank inclusion/restricted matrices and
    # the standalone result envelope before permitting any chain matrices.
    additional_output_bytes = _DEGENERATE_SUBMODULE_OUTPUT_OVERHEAD // 2 + 12 * (
        sum(ranks) + inclusion_cells + degenerate_boundary_cells
    )
    chains_module._preflight(
        source,
        additional_matrix_cells=inclusion_cells + degenerate_boundary_cells,
        additional_output_bytes=additional_output_bytes,
        output_name="degenerate submodule",
        output_error_code="simplicial_set.degenerate_submodule_output_budget_exceeded",
    )
    return source, basis_indices


def _admission_error(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("simplicial_set",),
        code=f"simplicial_set.{code}",
        message=message,
    )


def _degenerate_indices(
    source: FiniteTruncatedSimplicialSet,
) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = [()]
    for degree in range(1, source.max_degree + 1):
        images = {
            target
            for degeneracy in source.degeneracy_maps[degree - 1]
            for target in degeneracy
        }
        result.append(tuple(sorted(images)))
    return tuple(result)


def degenerate_submodule(
    request: DegenerateSubmoduleRequest,
) -> DegenerateSubmoduleResult:
    """Return the span of degenerate simplices as a based chain subcomplex."""
    source, basis_indices = _admit(request)
    ambient = chains_module._unnormalized_chains_from_checked_source(request, source)
    sizes = tuple(len(level) for level in ambient.simplex_bases)
    ranks = tuple(len(indices) for indices in basis_indices)

    inclusion_matrices: list[tuple[tuple[ChainCoefficient, ...], ...]] = []
    for ambient_rank, indices in zip(sizes, basis_indices, strict=True):
        column_for_index = {index: column for column, index in enumerate(indices)}
        inclusion_matrices.append(
            tuple(
                tuple(
                    1 if column_for_index.get(row_index) == column else 0
                    for column in range(len(indices))
                )
                for row_index in range(ambient_rank)
            )
        )

    differentials: list[tuple[tuple[ChainCoefficient, ...], ...]] = []
    for degree in range(1, len(sizes)):
        ambient_matrix = ambient.chain_complex.differential_matrices[degree - 1]
        source_indices = basis_indices[degree]
        target_positions = {
            index: position for position, index in enumerate(basis_indices[degree - 1])
        }
        matrix: list[list[ChainCoefficient]] = [
            [0] * len(source_indices) for _ in basis_indices[degree - 1]
        ]
        for column, source_index in enumerate(source_indices):
            for ambient_row, row in enumerate(ambient_matrix):
                coefficient = row[source_index]
                target_row = target_positions.get(ambient_row)
                if target_row is None:
                    if coefficient != 0:
                        raise ArithmeticError(
                            "alternating boundary of a degenerate simplex has a "
                            "nondegenerate component"
                        )
                    continue
                matrix[target_row][column] = coefficient
        differentials.append(tuple(tuple(row) for row in matrix))

    degenerate_complex = ChainComplexValue(
        coefficient_ring=request.coefficient_ring,
        prime=request.prime,
        degree_min=0,
        degree_max=ambient.simplicial_set.max_degree,
        basis_sizes=ranks,
        differential_matrices=tuple(differentials),
    )
    return DegenerateSubmoduleResult(
        unnormalized_chains=ambient,
        degenerate_basis_indices=basis_indices,
        inclusion_matrices=tuple(inclusion_matrices),
        degenerate_complex=degenerate_complex,
    )


__all__ = [
    "DegenerateSubmoduleRequest",
    "DegenerateSubmoduleResult",
    "degenerate_submodule",
]
