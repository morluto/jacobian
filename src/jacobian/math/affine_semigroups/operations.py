"""Native exact integer relation-lattice operation."""

from __future__ import annotations

from itertools import combinations
from math import comb, gcd

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups._kernel import (
    compute_relation_lattice_data,
)
from jacobian.math.affine_semigroups._models import (
    MAX_CIRCUIT_CONTEXT_BYTES,
    MAX_CIRCUIT_COORDINATE_DIGITS,
    MAX_CIRCUIT_KERNEL_WORK,
    MAX_CIRCUIT_OUTPUT_BYTES,
    MAX_CIRCUIT_RANK_WORK,
    MAX_RELATION_LATTICE_DIMENSION,
    MAX_RELATION_LATTICE_INPUT_DIGITS,
    IntegerConfigurationCircuitsResult,
    RelationLatticeResult,
)
from jacobian.math.matrices.values import IntegerMatrix


def _admit_relation_lattice(configuration: IntegerMatrix) -> None:
    """Enforce the published envelope on the native admission boundary.

    Catalog requests are bounded by the request model; native callers bypass
    wire validation, so the same axes and scalar-digit limits are enforced
    here before any Smith or Hermite work starts.
    """

    rows = configuration.row_count
    columns = configuration.column_count
    if not (
        1 <= rows <= MAX_RELATION_LATTICE_DIMENSION
        and 1 <= columns <= MAX_RELATION_LATTICE_DIMENSION
    ):
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.relation_lattice.budget_exceeded",
            message=(
                "relation-lattice configuration axes are limited to "
                f"{MAX_RELATION_LATTICE_DIMENSION} rows and columns"
            ),
        )
    limit = 10**MAX_RELATION_LATTICE_INPUT_DIGITS
    if any(abs(int(value)) >= limit for row in configuration.entries for value in row):
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.relation_lattice.budget_exceeded",
            message=(
                "relation-lattice configuration scalars are limited to "
                f"{MAX_RELATION_LATTICE_INPUT_DIGITS} decimal digits"
            ),
        )


def relation_lattice(configuration: IntegerMatrix) -> RelationLatticeResult:
    """Return the canonical integer kernel lattice ``ker_Z(A)``.

    Admission bounds the configuration axes and scalar digits at both the
    request model and this native boundary; the maintained Smith and Hermite
    kernels bound their own intermediate growth and exact output.
    """

    _admit_relation_lattice(configuration)
    data = compute_relation_lattice_data(configuration)
    return RelationLatticeResult._from_kernel(
        configuration=configuration,
        relation_lattice=data.relation_lattice,
        relation_basis=data.relation_basis,
        hnf_transformation=data.hnf_transformation,
        rank=data.rank,
        nullity=data.nullity,
        smith_invariant_factors=data.smith_invariant_factors,
        smith_rank=data.smith_rank,
        saturated_basis=data.saturated_basis,
        saturation_inclusion_transform=data.saturation_inclusion_transform,
        saturation_index=data.saturation_index,
    )


def _admit_circuit_configuration(configuration: object) -> IntegerMatrix:
    if type(configuration) is not IntegerMatrix:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="affine_semigroup.circuit_configuration",
            message="configuration must be a canonical integer matrix",
        )
    rows = getattr(configuration, "row_count", None)
    columns = getattr(configuration, "column_count", None)
    entries = getattr(configuration, "entries", None)
    if (
        type(rows) is not int
        or type(columns) is not int
        or getattr(configuration, "domain", None) != "ZZ"
        or type(entries) is not tuple
        or len(entries) != rows
        or any(type(row) is not tuple or len(row) != columns for row in entries)
    ):
        raise OperationDomainValidationError(
            location=("configuration",),
            code="affine_semigroup.circuit_configuration",
            message="configuration is malformed",
        )
    if not (
        1 <= rows <= MAX_RELATION_LATTICE_DIMENSION
        and 1 <= columns <= MAX_RELATION_LATTICE_DIMENSION
    ):
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.circuit_budget_exceeded",
            message="circuit configuration axes are limited to 12 rows and columns",
        )
    limit = 10**MAX_RELATION_LATTICE_INPUT_DIGITS
    for row in entries:
        for value in row:
            if type(value) is not int:
                raise OperationDomainValidationError(
                    location=("configuration",),
                    code="affine_semigroup.circuit_configuration",
                    message="configuration entries must be exact integers",
                )
            if abs(value) >= limit:
                raise OperationResourceAdmissionError(
                    location=("configuration",),
                    code="affine_semigroup.circuit_budget_exceeded",
                    message="circuit configuration entries are limited to 8 decimal digits",
                )
    try:
        admitted = IntegerMatrix.model_validate(configuration.model_dump(mode="python"))
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="affine_semigroup.circuit_configuration",
            message="configuration is malformed",
        ) from exc
    rows, columns = admitted.row_count, admitted.column_count
    support_limit = min(columns, rows + 1)
    support_count = sum(comb(columns, size) for size in range(1, support_limit + 1))
    rank_work = sum(
        comb(columns, size) * rows * size * min(rows, size)
        for size in range(1, support_limit + 1)
    )
    kernel_work = support_count * rows * support_limit * min(rows, support_limit)
    if rank_work > MAX_CIRCUIT_RANK_WORK or kernel_work > MAX_CIRCUIT_KERNEL_WORK:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.circuit_work",
            message="configuration exceeds the bounded circuit enumeration work envelope",
        )
    # A support-minimal relation on at most 12 columns is determined by
    # cofactors of an at-most 11 by 11 minor. Hadamard's bound gives at most
    # 94 decimal digits per coordinate for the admitted 8-digit inputs.
    output_bound = MAX_CIRCUIT_CONTEXT_BYTES + support_count * (
        columns * (MAX_CIRCUIT_COORDINATE_DIGITS + 2) + columns + 2
    )
    if output_bound > MAX_CIRCUIT_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.circuit_output",
            message="worst-case circuit result exceeds its 8 MiB output envelope",
        )
    return admitted


def integer_configuration_circuits(
    configuration: IntegerMatrix,
) -> IntegerConfigurationCircuitsResult:
    """Return all primitive support-minimal integer kernel vectors of A.

    Every candidate generator subset of size at most ``rows + 1`` is ranked
    once; larger subsets cannot be support-minimal. A subset is a circuit
    support exactly when its rank drops by one and every one-generator deletion
    has full column rank. The unique rational kernel line is converted to its
    primitive integral vector and normalized by its first nonzero coordinate.
    """
    configuration = _admit_circuit_configuration(configuration)
    from flint import fmpz_mat

    rows = tuple(tuple(int(value) for value in row) for row in configuration.entries)
    column_count = configuration.column_count
    support_limit = min(column_count, configuration.row_count + 1)
    ranks: dict[tuple[int, ...], int] = {}
    circuits: list[tuple[int, ...]] = []
    ranks[()] = 0
    for size in range(1, support_limit + 1):
        for support in combinations(range(column_count), size):
            submatrix = tuple(tuple(row[index] for index in support) for row in rows)
            backend_matrix = fmpz_mat(submatrix)
            rank = int(backend_matrix.rank())
            ranks[support] = rank
            if rank != size - 1:
                continue
            if any(
                ranks[support[:index] + support[index + 1 :]] != size - 1
                for index in range(size)
            ):
                continue
            nullspace, nullity = backend_matrix.nullspace()
            if nullity != 1:
                raise RuntimeError("rank-minimal circuit support had nonunit nullity")
            vector = [int(nullspace[index, 0]) for index in range(size)]
            divisor = 0
            for value in vector:
                divisor = gcd(divisor, abs(value))
            if divisor == 0:
                raise RuntimeError("rank-minimal circuit support had zero nullspace")
            vector = [value // divisor for value in vector]
            first = next(value for value in vector if value)
            if first < 0:
                vector = [-value for value in vector]
            full = [0] * column_count
            for index, value in zip(support, vector, strict=True):
                full[index] = value
            circuits.append(tuple(full))
    result = tuple(sorted(circuits))
    return IntegerConfigurationCircuitsResult._from_kernel(
        configuration=configuration,
        circuits=result,
    )


__all__ = ["integer_configuration_circuits", "relation_lattice"]
