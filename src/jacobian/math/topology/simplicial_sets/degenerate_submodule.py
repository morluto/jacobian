"""Exact degenerate subcomplex of a finite simplicial-set chain prefix."""

from __future__ import annotations

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.chain_complexes.values import (
    MAX_MATRIX_CELLS,
    MAX_OPERATION_MATRIX_CELLS,
    ChainCoefficient,
    ChainComplexValue,
)
from jacobian.math.topology.simplicial_sets._models import (
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets.chains import (
    UnnormalizedChainsRequest,
    UnnormalizedChainsResult,
    unnormalized_chains,
)

MAX_DEGENERATE_SUBMODULE_OUTPUT_BYTES = 256_000
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


def _admit(source: FiniteTruncatedSimplicialSet) -> None:
    """Bound work and output from source ranks before any chain matrices exist.

    In each degree, the degenerate rank is at most the ambient rank. These
    source-derived upper bounds cover the exact matrices later built, including
    empty and fully degenerate degrees.
    """
    sizes = tuple(len(level) for level in source.sets)
    ambient_cells = sum(
        sizes[degree - 1] * sizes[degree] for degree in range(1, len(sizes))
    )
    # A degreewise inclusion has at most source-rank squared cells; restricting
    # the differential has at most the ambient boundary shape.
    inclusion_cells = sum(size * size for size in sizes)
    degenerate_boundary_cells = ambient_cells
    degeneracy_rows = sum(
        (degree + 1) * sizes[degree] for degree in range(max(0, source.max_degree))
    )
    face_rows = sum((degree + 1) * sizes[degree] for degree in range(1, len(sizes)))
    work = (
        degeneracy_rows
        + face_rows
        + ambient_cells
        + inclusion_cells
        + degenerate_boundary_cells
    )
    if ambient_cells > MAX_OPERATION_MATRIX_CELLS:
        _admission_error(
            "degenerate_submodule_ambient_matrix_budget_exceeded",
            f"ambient boundaries require {ambient_cells} cells, exceeding "
            f"the {MAX_OPERATION_MATRIX_CELLS}-cell construction bound",
        )
    if inclusion_cells + ambient_cells + degenerate_boundary_cells > MAX_MATRIX_CELLS:
        _admission_error(
            "degenerate_submodule_matrix_budget_exceeded",
            "ambient, inclusion, and subcomplex matrices exceed the aggregate "
            f"{MAX_MATRIX_CELLS}-cell bound",
        )
    if work > MAX_DEGENERATE_SUBMODULE_WORK_UNITS:
        _admission_error(
            "degenerate_submodule_work_budget_exceeded",
            f"degenerate-submodule construction requires {work} work units, "
            f"exceeding {MAX_DEGENERATE_SUBMODULE_WORK_UNITS}",
        )

    source_bytes = len(source.model_dump_json().encode("utf-8"))
    # Every matrix scalar is at most seven digits over the admitted prime
    # fields, and has absolute value at most five over ZZ or QQ. Twelve bytes
    # per scalar also covers separators; the overhead covers all JSON shape.
    estimated_bytes = (
        _DEGENERATE_SUBMODULE_OUTPUT_OVERHEAD
        + source_bytes
        + 12 * (ambient_cells + inclusion_cells + degenerate_boundary_cells)
        + 12 * source.total_simplices
    )
    if estimated_bytes > MAX_DEGENERATE_SUBMODULE_OUTPUT_BYTES:
        _admission_error(
            "degenerate_submodule_output_budget_exceeded",
            f"estimated result size {estimated_bytes} bytes exceeds "
            f"{MAX_DEGENERATE_SUBMODULE_OUTPUT_BYTES}",
        )


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
    source = request.simplicial_set
    _admit(source)
    ambient = unnormalized_chains(request)
    basis_indices = _degenerate_indices(ambient.simplicial_set)
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
        matrix = [[0] * len(source_indices) for _ in basis_indices[degree - 1]]
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
