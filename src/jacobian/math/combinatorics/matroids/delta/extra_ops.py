from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta.extra import (
    MAX_BINARY_GROUND,
    MAX_BINARY_LABEL_BYTES,
    MAX_BINARY_PRINCIPAL_MINOR_WORK,
    MAX_BINARY_TWIST_OUTPUT_CELLS,
    MAX_BINARY_TWIST_OUTPUT_MEMBERSHIPS,
    MAX_BINARY_TWIST_STATES,
    MAX_BINARY_TWIST_TRANSPORT_WORK,
    BinaryMatrixResult,
    BinarySymmetricMatrix,
)
from jacobian.math.combinatorics.matroids.delta.values import (
    DeltaMatroidAdmissionError,
    FiniteDeltaMatroid,
    first_symmetric_exchange_obstruction,
    require_delta_matroid_admission,
)


def _admit_delta(value: object) -> FiniteDeltaMatroid:
    if type(value) is not FiniteDeltaMatroid:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.carrier",
            message="value must be a canonical finite delta-matroid",
        )
    try:
        return FiniteDeltaMatroid.model_validate(value.model_dump(mode="python"))
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.carrier",
            message="delta-matroid carrier is malformed",
        ) from exc


def _check(d: FiniteDeltaMatroid) -> FiniteFeasibleSetSystem:
    try:
        s = FiniteFeasibleSetSystem(ground=d.ground, feasible=d.feasible)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
            message="source feasible family is malformed",
        ) from exc
    try:
        require_delta_matroid_admission(s)
    except DeltaMatroidAdmissionError as exc:
        if exc.reason in {
            "memberships_exceeded",
            "label_bytes_exceeded",
            "candidate_work_exceeded",
        }:
            raise OperationResourceAdmissionError(
                location=("delta_matroid",),
                code=f"delta_matroid.{exc.reason}",
                message=str(exc),
            ) from exc
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code=f"delta_matroid.{exc.reason}",
            message=str(exc),
        ) from exc
    if first_symmetric_exchange_obstruction(s) is not None:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_delta",
            message="source is not a delta-matroid",
        )
    return s


def _validate_minor_axes(
    d: FiniteDeltaMatroid, delete: tuple[int, ...], contract: tuple[int, ...]
) -> None:
    n = len(d.ground)
    if (
        type(delete) is not tuple
        or type(contract) is not tuple
        or any(type(i) is not int for i in (*delete, *contract))
    ):
        raise OperationDomainValidationError(
            location=("minor",),
            code="delta_matroid.minor_axis",
            message="minor axes must be tuples of exact indices",
        )
    if (
        delete != tuple(sorted(set(delete)))
        or contract != tuple(sorted(set(contract)))
        or set(delete) & set(contract)
        or any(i < 0 or i >= n for i in (*delete, *contract))
    ):
        raise OperationDomainValidationError(
            location=("minor",),
            code="delta_matroid.minor_axis",
            message="minor indices must be sorted, disjoint, and in range",
        )


def dual(d: FiniteDeltaMatroid) -> FiniteDeltaMatroid:
    d = _admit_delta(d)
    _check(d)
    n = len(d.ground)
    rows = tuple(sorted(tuple(sorted(set(range(n)) - set(row))) for row in d.feasible))
    return FiniteDeltaMatroid(ground=d.ground, feasible=rows)


def _remove_axis(
    rows: tuple[tuple[int, ...], ...], element: int, *, contract: bool
) -> tuple[tuple[int, ...], ...]:
    sets = [set(row) for row in rows]
    present = any(element in row for row in sets)
    # A loop contracts as deletion; a coloop deletes as contraction.
    use_contract = contract and present
    if not contract and all(element in row for row in sets):
        use_contract = True
    if use_contract:
        selected = [row for row in sets if element in row]
    else:
        selected = [row for row in sets if element not in row]
    remapped = []
    for row in selected:
        reduced = row - {element}
        remapped.append(tuple(sorted(i if i < element else i - 1 for i in reduced)))
    return tuple(sorted(set(remapped)))


def minor(
    d: FiniteDeltaMatroid, delete: tuple[int, ...] = (), contract: tuple[int, ...] = ()
) -> FiniteDeltaMatroid:
    d = _admit_delta(d)
    _check(d)
    _validate_minor_axes(d, delete, contract)
    rows = tuple(d.feasible)
    labels = list(d.ground)
    # Descending original indices keep the remaining indices stable while the
    # requested deletion/contraction axes are compacted.
    for element in sorted(delete, reverse=True):
        rows = _remove_axis(rows, element, contract=False)
        labels.pop(element)
    for element in sorted(contract, reverse=True):
        # Contract indices are original indices; account for deletions above.
        current = element - sum(1 for deleted in delete if deleted < element)
        rows = _remove_axis(rows, current, contract=True)
        labels.pop(current)
    if not rows:
        raise DeltaMatroidAdmissionError("empty_minor", "minor has no feasible set")
    return FiniteDeltaMatroid(ground=tuple(labels), feasible=rows)


def _det2(a: list[list[int]]) -> int:
    a = [list(row) for row in a]
    n = len(a)
    for i in range(n):
        p = next((j for j in range(i, n) if a[j][i]), None)
        if p is None:
            return 0
        a[i], a[p] = a[p], a[i]
        for j in range(i + 1, n):
            if a[j][i]:
                for k in range(i, n):
                    a[j][k] ^= a[i][k]
    return 1


def _canonical_binary_matrix(matrix: BinarySymmetricMatrix) -> BinarySymmetricMatrix:
    if type(matrix) is not BinarySymmetricMatrix:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="delta_matroid.binary_carrier",
            message="matrix must be a canonical symmetric binary matrix",
        )
    try:
        matrix = BinarySymmetricMatrix.model_validate(
            {"ground": matrix.ground, "entries": matrix.entries}
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="delta_matroid.binary_carrier",
            message="matrix is not a canonical symmetric binary matrix",
        ) from exc
    return matrix


def _admit_binary_matrix(matrix: BinarySymmetricMatrix) -> int:
    n = len(matrix.ground)
    if n > MAX_BINARY_GROUND:
        raise OperationResourceAdmissionError(
            location=("matrix", "ground"),
            code="delta_matroid.binary_work",
            message="binary principal-minor work exceeds its envelope",
        )
    subsets = 1 << n
    elimination_work = subsets * max(1, n) ** 3
    if elimination_work > MAX_BINARY_PRINCIPAL_MINOR_WORK:
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="delta_matroid.binary_work",
            message="binary principal-minor work exceeds its envelope",
        )
    return elimination_work


def _binary_family(matrix: BinarySymmetricMatrix) -> BinaryMatrixResult:
    """Construct after the matrix and principal-minor work were admitted."""

    n = len(matrix.ground)
    rows = []
    for mask in range(1 << n):
        idx = [i for i in range(n) if mask >> i & 1]
        if _det2([[matrix.entries[i][j] for j in idx] for i in idx]):
            rows.append(tuple(idx))
    # Numeric masks enumerate by binary value, not by the canonical tuple
    # order required by the finite feasible-set carrier ((), (0), (0, 1),
    # (1), ...).  Sorting is part of result construction, not a cosmetic
    # presentation step: the carrier rejects non-canonical rows.
    canonical_rows = tuple(sorted(rows)) or ((),)
    return BinaryMatrixResult(
        matrix=matrix,
        delta_matroid=FiniteDeltaMatroid(ground=matrix.ground, feasible=canonical_rows),
    )


def binary(matrix: BinarySymmetricMatrix) -> BinaryMatrixResult:
    """Return D(A), whose feasible sets are its nonsingular principal axes."""

    matrix = _canonical_binary_matrix(matrix)
    _admit_binary_matrix(matrix)
    return _binary_family(matrix)


def binary_matrix_twist(
    matrix: BinarySymmetricMatrix, subset: tuple[int, ...] = ()
) -> BinaryMatrixResult:
    """Return D(A)*T with the matrix presentation and twist retained."""

    if type(subset) is not tuple:
        raise OperationDomainValidationError(
            location=("subset",),
            code="delta_matroid.binary_twist_subset",
            message="twist indices must be a tuple of exact integers",
        )
    if len(subset) > MAX_BINARY_GROUND:
        raise OperationDomainValidationError(
            location=("subset",),
            code="delta_matroid.binary_twist_subset",
            message="twist subset exceeds the maximum ground size",
        )
    if any(type(index) is not int for index in subset):
        raise OperationDomainValidationError(
            location=("subset",),
            code="delta_matroid.binary_twist_subset",
            message="twist indices must be a tuple of exact integers",
        )
    matrix = _canonical_binary_matrix(matrix)
    n = len(matrix.ground)
    if len(subset) > n or subset != tuple(sorted(set(subset))) or any(
        index < 0 or index >= n for index in subset
    ):
        raise OperationDomainValidationError(
            location=("subset",),
            code="delta_matroid.binary_twist_subset",
            message="twist indices must be sorted, distinct, and in range",
        )
    principal_work = _admit_binary_matrix(matrix)
    label_bytes = sum(len(label.encode("utf-8")) for label in matrix.ground)
    states = 1 << n
    max_output_memberships = (n * states) // 2
    transport_work = states * (1 + 2 * n + n**2)
    output_cells = n**2 + states + max_output_memberships + n
    output_label_bytes = 2 * label_bytes
    if (
        states > MAX_BINARY_TWIST_STATES
        or max_output_memberships > MAX_BINARY_TWIST_OUTPUT_MEMBERSHIPS
        or transport_work > MAX_BINARY_TWIST_TRANSPORT_WORK
        or output_cells > MAX_BINARY_TWIST_OUTPUT_CELLS
        or output_label_bytes > 2 * MAX_BINARY_LABEL_BYTES
        or principal_work > MAX_BINARY_PRINCIPAL_MINOR_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="delta_matroid.binary_twist_work",
            message="binary matrix twist exceeds its admitted work or output envelope",
        )

    source = _binary_family(matrix)
    twist_mask = sum(1 << index for index in subset)
    source_masks = tuple(
        sum(1 << index for index in row) for row in source.delta_matroid.feasible
    )
    target_masks = tuple(mask ^ twist_mask for mask in source_masks)
    target_memberships = sum(mask.bit_count() for mask in target_masks)
    if target_memberships > MAX_BINARY_TWIST_OUTPUT_MEMBERSHIPS:
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="delta_matroid.binary_twist_output",
            message="twisted feasible-family memberships exceed the admitted output bound",
        )
    target_rows = tuple(
        sorted(
            tuple(index for index in range(n) if mask >> index & 1)
            for mask in target_masks
        )
    )
    return BinaryMatrixResult(
        matrix=matrix,
        twist=subset,
        delta_matroid=FiniteDeltaMatroid._from_kernel(
            FiniteFeasibleSetSystem(ground=matrix.ground, feasible=target_rows)
        ),
    )


__all__ = ["binary", "binary_matrix_twist", "dual", "minor"]
