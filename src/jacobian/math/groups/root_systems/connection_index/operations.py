"""Exact quotient of root and weight lattices for finite Cartan data."""

from math import prod

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.root_systems._models import (
    MAX_RANK,
    CartanMatrix,
)
from jacobian.math.groups.root_systems.connection_index._models import (
    RootSystemConnectionIndexResult,
)
from jacobian.math.groups.root_systems.operations import _admit_cartan_finite_type
from jacobian.math.matrices.operations import smith_normal_form_result
from jacobian.math.matrices.values import IntegerMatrix

MAX_CONNECTION_INDEX_OUTPUT_CELLS = 8_192


def _admit_connection_cartan(value: CartanMatrix) -> CartanMatrix:
    """Bound raw Cartan carrier fields before copying them for validation."""

    if not isinstance(value, CartanMatrix):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.connection_index.invalid_cartan_carrier",
            message="connection index requires a canonical Cartan matrix",
        )
    matrix = getattr(value, "matrix", None)
    entries = getattr(matrix, "entries", None)
    axis = getattr(value, "simple_root_axis", None)
    if not isinstance(entries, tuple):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.connection_index.invalid_cartan_carrier",
            message="Cartan entries must be a bounded tuple of integer rows",
        )
    rank = len(entries)
    if not 1 <= rank <= MAX_RANK:
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="root_system.connection_index.rank_exceeded",
            message=f"connection-index Cartan rank is limited to {MAX_RANK}",
        )
    if (
        not isinstance(matrix, IntegerMatrix)
        or getattr(matrix, "domain", None) != "ZZ"
        or not isinstance(axis, tuple)
        or axis != tuple(range(rank))
        or type(matrix.row_count) is not int
        or type(matrix.column_count) is not int
        or matrix.row_count != rank
        or matrix.column_count != rank
        or any(
            not isinstance(row, tuple)
            or len(row) != rank
            or any(
                type(entry) is not int or abs(entry).bit_length() > 2 for entry in row
            )
            for row in entries
        )
    ):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.connection_index.invalid_cartan_carrier",
            message="Cartan matrix must be a bounded square integer matrix on its canonical axis",
        )
    return CartanMatrix.model_validate(
        {
            "matrix": {
                "domain": "ZZ",
                "row_count": rank,
                "column_count": rank,
                "entries": entries,
            },
            "simple_root_axis": axis,
        },
        strict=True,
    )


def root_system_connection_index(
    matrix: CartanMatrix,
) -> RootSystemConnectionIndexResult:
    """Compute ``P/Q`` using the Smith form of the root-to-weight inclusion.

    The Cartan convention is ``A[i,j] = <alpha_i^vee, alpha_j>``. Columns of
    ``A`` give the simple roots in the fundamental-weight basis, so
    ``P/Q`` is the cokernel of ``A`` over ``ZZ``.
    """

    cartan = _admit_connection_cartan(matrix)
    rows = cartan.entries
    _admit_cartan_finite_type(rows)

    rank = len(rows)
    output_cells = 4 * rank * rank + 4 * rank + 8
    if output_cells > MAX_CONNECTION_INDEX_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="root_system.connection_index.result_size_exceeded",
            message="root-system connection-index result exceeds its cell bound",
        )

    root_to_weight = IntegerMatrix(
        row_count=rank,
        column_count=rank,
        entries=rows,
    )
    smith = smith_normal_form_result(root_to_weight)
    if smith.rank != rank:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="root_system.connection_index.singular_cartan_matrix",
            message="finite Cartan matrices must have full rank",
        )
    invariant_factors = tuple(
        factor for factor in smith.invariant_factors if factor > 1
    )
    return RootSystemConnectionIndexResult._from_kernel(
        cartan_matrix=cartan,
        root_to_weight=root_to_weight,
        invariant_factors=invariant_factors,
        connection_index=prod(invariant_factors, start=1),
    )


__all__ = ["MAX_CONNECTION_INDEX_OUTPUT_CELLS", "root_system_connection_index"]
