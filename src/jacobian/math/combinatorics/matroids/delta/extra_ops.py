from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta.extra import (
    MAX_BINARY_GROUND,
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


def binary(matrix: BinarySymmetricMatrix) -> BinaryMatrixResult:
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
    n = len(matrix.ground)
    if n > MAX_BINARY_GROUND:
        raise OperationResourceAdmissionError(
            location=("matrix", "ground"),
            code="delta_matroid.binary_work",
            message="binary principal-minor work exceeds its envelope",
        )
    subsets = 1 << n
    if subsets * max(1, n) ** 3 > 250_000:
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="delta_matroid.binary_work",
            message="binary principal-minor work exceeds its envelope",
        )
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


__all__ = ["binary", "dual", "minor"]
