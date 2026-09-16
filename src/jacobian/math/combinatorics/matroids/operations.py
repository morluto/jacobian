"""Domain-owned linear matroid operations over the shared GF(p) kernels."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids._models import (
    MAX_WEIGHT_DIGITS,
    ExchangeLedgerRow,
    LinearMatroid,
    MatroidClosureResult,
    MaximumWeightBasisResult,
    validate_subset_indices,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    rank as pf_rank,
)

MAX_CLOSURE_RANK_WORK = 50_000_000


def _rank_work(rows: int, columns: int) -> int:
    return rows * columns * min(rows, columns)


def _require_closure_work(matroid: LinearMatroid, subset_size: int) -> None:
    rows = len(matroid.matrix.entries)
    ground_size = matroid.ground_size
    work = _rank_work(rows, subset_size) + (ground_size - subset_size) * _rank_work(
        rows, subset_size + 1
    )
    if work > MAX_CLOSURE_RANK_WORK:
        raise OperationDomainValidationError(
            location=("matroid", "subset"),
            code="matroid.closure.work_bound",
            message="closure rank computations exceed the exact work bound",
        )


def _selected_columns_matrix(
    matroid: LinearMatroid, column_indices: list[int]
) -> PrimeFieldMatrix:
    """The canonical matrix restricted to the selected ground elements."""
    rows = matroid.matrix.entries
    selected = [tuple(row[j] for j in column_indices) for row in rows]
    return PrimeFieldMatrix(
        prime=matroid.matrix.prime,
        entries=tuple(selected),
        columns=len(column_indices),
    )


def matroid_rank(matroid: LinearMatroid) -> int:
    """Exact rank of the linear matroid (dimension of the column span).

    Routes through the maintained shared ``rank`` kernel so the matroid
    domain never maintains a divergent elimination implementation.
    """
    return pf_rank(matroid.matrix)


def _closure_invariant(
    matroid: LinearMatroid, subset: list[int]
) -> tuple[tuple[int, ...], int]:
    """Pure closure core: the flat of ``subset`` and its rank.

    An element e joins the closure exactly when adding it does not raise
    the subset's rank; every intermediate rank routes through the shared
    kernel. Returns ``(sorted_closure, subset_rank)``.
    """
    subset_rank = pf_rank(_selected_columns_matrix(matroid, subset))
    closure = set(subset)
    for element in range(matroid.ground_size):
        if element in closure:
            continue
        test = [*subset, element]
        if pf_rank(_selected_columns_matrix(matroid, test)) == subset_rank:
            closure.add(element)
    return tuple(sorted(closure)), subset_rank


def matroid_closure(
    matroid: LinearMatroid, subset: list[int] | tuple[int, ...]
) -> tuple[tuple[int, ...], int]:
    """Public native entry: exact closure and subset rank.

    Applies the same subset admission as the wire request so negative or
    out-of-range indices never reach the kernel through Python indexing.
    """
    canonical_subset = tuple(subset)
    try:
        validate_subset_indices(matroid, list(canonical_subset))
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("subset",),
            code="matroid.subset.invalid",
            message=str(exc),
        ) from exc
    _require_closure_work(matroid, len(canonical_subset))
    return _closure_invariant(matroid, list(canonical_subset))


def closure_result(
    matroid: LinearMatroid, subset: list[int] | tuple[int, ...]
) -> MatroidClosureResult:
    """Return the canonical source-bound closure result."""

    canonical_subset = tuple(subset)
    closure, subset_rank = matroid_closure(matroid, canonical_subset)
    return MatroidClosureResult._from_kernel(
        matroid, canonical_subset, closure, subset_rank
    )


def verify_closure(claim: MatroidClosureResult) -> bool:
    """Verify a serialized closure and rank against its retained matroid."""

    try:
        return closure_result(claim.matroid, claim.subset) == claim
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


def _admit_weight_basis(
    matroid: LinearMatroid, weights: tuple[int, ...]
) -> tuple[int, ...]:
    """Shared greedy/exchange admission for native and catalog callers.

    Returns canonical weights after checking ground coverage, exact integer
    type, digit bounds, and rank-call work before any kernel expansion.
    """
    if not isinstance(weights, tuple):
        raise OperationDomainValidationError(
            location=("weights",),
            code="matroid.weights.carrier",
            message="weights must be a tuple with one entry per ground element",
        )
    if len(weights) != matroid.ground_size:
        raise OperationDomainValidationError(
            location=("weights",),
            code="matroid.weights.ground_coverage",
            message="weights must cover the matroid ground set exactly once",
        )
    if any(type(weight) is not int for weight in weights):
        raise OperationDomainValidationError(
            location=("weights",),
            code="matroid.weights.integer",
            message="matroid weights must be exact integers",
        )
    if any(abs(weight) >= 10**MAX_WEIGHT_DIGITS for weight in weights):
        raise OperationDomainValidationError(
            location=("weights",),
            code="matroid.weights.digits",
            message="matroid weights must have fewer than "
            f"{MAX_WEIGHT_DIGITS} decimal digits",
        )
    rows = len(matroid.matrix.entries)
    ground_size = matroid.ground_size
    work = _rank_work(rows, ground_size)
    work += ground_size * _rank_work(rows, ground_size)
    work += ground_size * ground_size * _rank_work(rows, ground_size)
    if work > MAX_CLOSURE_RANK_WORK:
        raise OperationResourceAdmissionError(
            location=("matroid", "weights"),
            code="matroid.maximum_weight_basis.work_bound",
            message="maximum-weight basis rank work exceeds the exact work bound",
        )
    return weights


def _independent(matroid: LinearMatroid, subset: list[int]) -> bool:
    """Return whether ``subset`` is independent via the shared rank kernel."""
    if not subset:
        return True
    return pf_rank(_selected_columns_matrix(matroid, subset)) == len(subset)


def _greedy_order(weights: tuple[int, ...]) -> tuple[int, ...]:
    """Deterministic greedy consideration order: weight desc, index asc."""
    return tuple(sorted(range(len(weights)), key=lambda e: (-weights[e], e)))


def _fundamental_circuit(
    matroid: LinearMatroid, basis: list[int], outside: int
) -> tuple[int, ...]:
    """Return the fundamental circuit of ``outside`` with respect to ``basis``.

    The circuit is ``outside`` plus exactly those basis elements whose removal
    keeps ``basis + outside - b`` spanning (equivalently, dependent). For a
    basis ``B`` and ``e`` outside, ``b`` is in ``C(e,B)`` iff ``B + e - b``
    is a basis.
    """
    basis_rank = pf_rank(_selected_columns_matrix(matroid, list(basis)))
    circuit = [outside]
    for candidate in basis:
        exchange = [b for b in basis if b != candidate] + [outside]
        if pf_rank(_selected_columns_matrix(matroid, exchange)) == basis_rank:
            circuit.append(candidate)
    return tuple(sorted(circuit))


def maximum_weight_basis_result(
    matroid: LinearMatroid, weights: tuple[int, ...] | list[int]
) -> MaximumWeightBasisResult:
    """Run the deterministic greedy kernel and replay exchange optimality.

    Selects elements in decreasing-weight order (ties by increasing index),
    keeping each element that preserves independence. Before return, replays
    that the selection is a basis and that no valid single-element exchange
    strictly improves the total weight.
    """
    canonical_weights = _admit_weight_basis(matroid, tuple(weights))
    order = _greedy_order(canonical_weights)
    basis: list[int] = []
    for element in order:
        if _independent(matroid, [*basis, element]):
            basis.append(element)
    basis_rank = pf_rank(_selected_columns_matrix(matroid, list(basis)))
    if basis_rank != len(basis):
        raise OperationDomainValidationError(
            location=("matroid",),
            code="matroid.maximum_weight_basis.selection",
            message="greedy selection failed the independence replay",
        )
    full_rank = pf_rank(matroid.matrix)
    if len(basis) != full_rank:
        raise OperationDomainValidationError(
            location=("matroid",),
            code="matroid.maximum_weight_basis.cardinality",
            message="greedy selection failed the basis cardinality replay",
        )
    total = sum(canonical_weights[element] for element in basis)
    ledger: list[ExchangeLedgerRow] = []
    basis_set = set(basis)
    for outside in range(matroid.ground_size):
        if outside in basis_set:
            continue
        circuit = _fundamental_circuit(matroid, basis, outside)
        if outside not in circuit or any(
            member not in basis_set and member != outside for member in circuit
        ):
            raise OperationDomainValidationError(
                location=("matroid",),
                code="matroid.maximum_weight_basis.circuit",
                message="fundamental circuit left the basis-plus-element domain",
            )
        best_delta = max(
            (
                canonical_weights[outside] - canonical_weights[member]
                for member in circuit
                if member != outside
            ),
            default=0,
        )
        if best_delta > 0:
            raise OperationDomainValidationError(
                location=("weights",),
                code="matroid.maximum_weight_basis.exchange",
                message="a single-element exchange strictly improves the total",
            )
        ledger.append(
            ExchangeLedgerRow.model_construct(
                outside=outside,
                circuit=circuit,
                best_delta=best_delta,
            )
        )
    return MaximumWeightBasisResult._from_kernel(
        matroid,
        canonical_weights,
        tuple(sorted(basis)),
        total,
        full_rank,
        order,
        tuple(ledger),
    )


def verify_maximum_weight_basis(claim: MaximumWeightBasisResult) -> bool:
    """Replay greedy selection and exchange optimality for a serialized claim."""

    try:
        return maximum_weight_basis_result(claim.matroid, claim.weights) == claim
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


__all__ = [
    "closure_result",
    "matroid_closure",
    "matroid_rank",
    "maximum_weight_basis_result",
    "verify_closure",
    "verify_maximum_weight_basis",
]
