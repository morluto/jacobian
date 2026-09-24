from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta.extra import (
    MAX_BINARY_GROUND,
    MAX_FEASIBLE_SIZE_PROFILE_ENTRIES,
    MAX_FEASIBLE_SIZE_PROFILE_OUTPUT_BYTES,
    MAX_TWIST_WIDTH_STATES,
    MAX_TWIST_WIDTH_WORK,
    BinaryLoopComplementRequest,
    BinaryLoopComplementResult,
    BinaryMatrixResult,
    BinarySymmetricMatrix,
    DeltaMatroidFeasibleSizeProfile,
    DeltaMatroidTwistWidthProfile,
    DeltaMatroidTwistWidthProfileResult,
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


def loop_complement(
    matrix: BinarySymmetricMatrix, subset: tuple[int, ...] = ()
) -> BinaryLoopComplementResult:
    """Apply loop complementation by toggling diagonal entries over GF(2).

    For a symmetric binary presentation A and element e, the feasible family
    of A with A[e,e] toggled is D(A)+e: for every feasible X not containing e,
    membership of X union {e} is toggled. Distinct elements commute.
    """
    if type(matrix) is not BinarySymmetricMatrix:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="delta_matroid.binary_carrier",
            message="matrix must be a canonical symmetric binary matrix",
        )
    try:
        request = BinaryLoopComplementRequest(matrix=matrix, subset=subset)
        matrix = BinarySymmetricMatrix.model_validate(
            {"ground": matrix.ground, "entries": matrix.entries}
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="delta_matroid.loop_complement_input",
            message="loop-complement input is malformed",
        ) from exc
    n = len(matrix.ground)
    # Admission is complete before constructing a changed matrix or enumerating
    # any principal minor. This is exactly the bound used by binary().
    if n > MAX_BINARY_GROUND or (1 << n) * max(1, n) ** 3 > 250_000:
        raise OperationResourceAdmissionError(
            location=("matrix", "ground"),
            code="delta_matroid.binary_work",
            message="binary principal-minor work exceeds its envelope",
        )
    toggled = set(request.subset)
    entries = tuple(
        tuple(bit ^ int(i == j and i in toggled) for j, bit in enumerate(row))
        for i, row in enumerate(matrix.entries)
    )
    result_matrix = BinarySymmetricMatrix(ground=matrix.ground, entries=entries)
    return BinaryLoopComplementResult(
        source=matrix,
        subset=request.subset,
        result=binary(result_matrix),
    )


def twist_width_profile(d: FiniteDeltaMatroid) -> DeltaMatroidTwistWidthProfileResult:
    """Return the complete width profile of all twists of ``d``.

    Bit ``i`` in a profile index means that ground element ``i`` is in the
    twisting subset.  The profile is admissible only after both its state
    count and its exact feasible-set evaluation count have been bounded.
    """

    d = _admit_delta(d)
    state_count = 1 << len(d.ground)
    work = state_count * len(d.feasible)
    if state_count > MAX_TWIST_WIDTH_STATES or work > MAX_TWIST_WIDTH_WORK:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="delta_matroid.twist_width_profile_work",
            message="complete twist-width profile exceeds its state or work envelope",
        )
    _check(d)

    feasible_masks = tuple(sum(1 << element for element in row) for row in d.feasible)
    widths = []
    for twist_mask in range(state_count):
        sizes = tuple(
            (feasible_mask ^ twist_mask).bit_count() for feasible_mask in feasible_masks
        )
        widths.append(max(sizes) - min(sizes))
    profile = DeltaMatroidTwistWidthProfile(
        ground=d.ground,
        widths_by_mask=tuple(widths),
    )
    return DeltaMatroidTwistWidthProfileResult(delta_matroid=d, profile=profile)


def feasible_size_profile(d: FiniteDeltaMatroid) -> DeltaMatroidFeasibleSizeProfile:
    """Return the complete family's counts from size 0 through |E|.

    This profile depends only on the retained ground and feasible-set tables;
    symmetric exchange is not needed to establish the returned histogram.
    """

    d = _admit_delta(d)
    entries = len(d.ground) + 1
    try:
        label_bytes = sum(len(label.encode("utf-8")) for label in d.ground)
    except UnicodeEncodeError:
        raise OperationDomainValidationError(
            location=("delta_matroid", "ground"),
            code="delta_matroid.labels_not_utf8",
            message="delta-matroid ground labels must be UTF-8-representable",
        ) from None
    # The result retains its ground axis and emits each count as decimal JSON.
    # Since the input family is already materialized, its row count bounds each
    # coefficient; this conservative estimate includes keys and JSON syntax.
    output_bytes = 2 * label_bytes + entries * (12 + len(str(len(d.feasible)))) + 512
    if (
        entries > MAX_FEASIBLE_SIZE_PROFILE_ENTRIES
        or output_bytes > MAX_FEASIBLE_SIZE_PROFILE_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="delta_matroid.feasible_size_profile_output",
            message="complete feasible-size profile exceeds its output envelope",
        )
    counts = [0] * entries
    for row in d.feasible:
        counts[len(row)] += 1
    return DeltaMatroidFeasibleSizeProfile(
        ground=d.ground,
        counts_by_size=tuple(counts),
    )


__all__ = [
    "binary",
    "dual",
    "feasible_size_profile",
    "minor",
    "twist_width_profile",
]
