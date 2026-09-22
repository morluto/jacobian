from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta.extra import (
    BinaryMatrixResult,
    BinarySymmetricMatrix,
)
from jacobian.math.combinatorics.matroids.delta.values import (
    DeltaMatroidAdmissionError,
    FiniteDeltaMatroid,
    first_symmetric_exchange_obstruction,
    require_delta_matroid_admission,
)


def _check(d: FiniteDeltaMatroid) -> FiniteFeasibleSetSystem:
    try:
        s = FiniteFeasibleSetSystem(ground=d.ground, feasible=d.feasible)
        require_delta_matroid_admission(s)
    except DeltaMatroidAdmissionError as exc:
        if exc.reason in {
            "memberships_exceeded",
            "label_bytes_exceeded",
            "candidate_work_exceeded",
        }:
            raise OperationResourceAdmissionError(
                location=("delta_matroid",),
                code="delta_matroid.source_work_bound",
                message=str(exc),
            ) from exc
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
            message=str(exc),
        ) from exc
    except OperationResourceAdmissionError:
        raise
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
            message=str(exc),
        ) from exc
    if first_symmetric_exchange_obstruction(s) is not None:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_delta",
            message="source is not a delta-matroid",
        )
    return s


def dual(d: FiniteDeltaMatroid) -> FiniteDeltaMatroid:
    _check(d)
    n = len(d.ground)
    rows = tuple(sorted(tuple(sorted(set(range(n)) - set(row))) for row in d.feasible))
    return FiniteDeltaMatroid(ground=d.ground, feasible=rows)


def _single_minor(
    ground: tuple[str, ...], rows: tuple[tuple[int, ...], ...], index: int, mode: str
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    present = [index in row for row in rows]
    is_loop = not any(present)
    is_coloop = all(present)
    # At a loop/coloop the standard delta-matroid convention exchanges the
    # requested operation so the minor remains nonempty.
    retain = (mode == "contract" and not is_loop) or (mode == "delete" and is_coloop)
    selected = tuple(
        row for row in rows if ((index in row) if retain else (index not in row))
    )
    remaining = tuple(i for i in range(len(ground)) if i != index)
    remap = {old: new for new, old in enumerate(remaining)}
    projected = tuple(
        tuple(sorted(remap[i] for i in row if i != index)) for row in selected
    )
    return (
        tuple(label for i, label in enumerate(ground) if i != index),
        tuple(sorted(set(projected))),
    )


def _validate_minor_axes(
    d: FiniteDeltaMatroid, delete: tuple[int, ...], contract: tuple[int, ...]
) -> None:
    n = len(d.ground)
    axes = (*delete, *contract)
    if (
        type(delete) is not tuple
        or type(contract) is not tuple
        or any(type(i) is not int for i in axes)
    ):
        raise OperationDomainValidationError(
            location=("minor",),
            code="delta_matroid.minor_axis",
            message="minor indices must be exact integers",
        )
    if delete != tuple(sorted(set(delete))) or contract != tuple(sorted(set(contract))):
        raise OperationDomainValidationError(
            location=("minor",),
            code="delta_matroid.minor_axis",
            message="minor indices must be sorted and distinct",
        )
    if any(i < 0 or i >= n for i in axes):
        raise OperationDomainValidationError(
            location=("minor",),
            code="delta_matroid.minor_axis",
            message="minor indices must be in range",
        )
    if set(delete) & set(contract):
        raise OperationDomainValidationError(
            location=("minor",),
            code="delta_matroid.minor_axis",
            message="minor deletion and contraction axes must be disjoint",
        )


def minor(
    d: FiniteDeltaMatroid, delete: tuple[int, ...] = (), contract: tuple[int, ...] = ()
) -> FiniteDeltaMatroid:
    _validate_minor_axes(d, delete, contract)
    _check(d)
    ground = d.ground
    rows = d.feasible
    # Source labels keep the requested axes stable while each step compacts
    # indices for the resulting carrier.
    for original in sorted(delete):
        label = d.ground[original]
        if label in ground:
            ground, rows = _single_minor(ground, rows, ground.index(label), "delete")
    for original in sorted(contract):
        label = d.ground[original]
        if label in ground:
            ground, rows = _single_minor(ground, rows, ground.index(label), "contract")
    return FiniteDeltaMatroid(ground=ground, feasible=rows)


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
    try:
        # Re-parse model instances as well: native callers can supply a
        # model_construct() value that bypassed the carrier validator.
        matrix = BinarySymmetricMatrix.model_validate(matrix.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="delta_matroid.binary_invalid",
            message=str(exc),
        ) from exc
    n = len(matrix.ground)
    principal_states = 1 << n
    if principal_states > 1 << 12:
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="delta_matroid.binary_work_bound",
            message="principal-minor enumeration exceeds the admitted work bound",
        )
    rows = []
    for mask in range(1 << n):
        idx = [i for i in range(n) if mask >> i & 1]
        if _det2([[matrix.entries[i][j] for j in idx] for i in idx]):
            rows.append(tuple(idx))
    return BinaryMatrixResult(
        matrix=matrix,
        delta_matroid=FiniteDeltaMatroid(
            ground=matrix.ground, feasible=tuple(rows or [()])
        ),
    )


__all__ = ["binary", "dual", "minor"]
