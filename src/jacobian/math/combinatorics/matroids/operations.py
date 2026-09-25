"""Domain-owned linear matroid operations over the shared GF(p) kernels."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids._models import (
    MAX_GROUND_SIZE,
    MAX_SPLIT_WEIGHT_DIGITS,
    MAX_WEIGHT_DIGITS,
    ExchangeLedgerRow,
    LinearMatroid,
    MatroidClosureResult,
    MatroidWeightFunction,
    MaximumWeightBasisResult,
    MaximumWeightIndependentSetResult,
    validate_subset_indices,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    rank as pf_rank,
)

MAX_CLOSURE_RANK_WORK = 50_000_000
# A canonical maximum-weight independent-set result charges ``3 * n`` fixed
# units plus one unit per weight digit. The widest admitted weight table is the
# split-witness domain, so the envelope must cover ``(3 + D) * n`` digits; the
# narrower 12-digit objective domain still bounds every public single-matroid
# request below this cap.
MAX_MAXIMUM_WEIGHT_INDEPENDENT_SET_OUTPUT_UNITS = (
    3 + MAX_SPLIT_WEIGHT_DIGITS
) * MAX_GROUND_SIZE


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


def _canonical_weight_function(
    matroid: LinearMatroid,
    weight_function: MatroidWeightFunction,
    *,
    max_digits: int = MAX_WEIGHT_DIGITS,
) -> tuple[tuple[int, ...], MatroidWeightFunction]:
    """Admit exact keyed coverage and normalize values to the source axis."""
    if not isinstance(weight_function, MatroidWeightFunction):
        raise OperationDomainValidationError(
            location=("weight_function",),
            code="matroid.weights.carrier",
            message="weight_function must be a canonical ground-axis table",
        )
    if (
        len(weight_function.ground_axis) != len(weight_function.values)
        or len(set(weight_function.ground_axis)) != len(weight_function.ground_axis)
        or set(weight_function.ground_axis) != set(matroid.ground_axis)
    ):
        raise OperationDomainValidationError(
            location=("weight_function", "ground_axis"),
            code="matroid.weights.ground_coverage",
            message="weight keys must equal the exact matroid ground axis",
        )
    by_label = dict(
        zip(weight_function.ground_axis, weight_function.values, strict=True)
    )
    weights = tuple(by_label[label] for label in matroid.ground_axis)
    if any(type(weight) is not int for weight in weights):
        raise OperationDomainValidationError(
            location=("weights",),
            code="matroid.weights.integer",
            message="matroid weights must be exact integers",
        )
    if any(abs(weight) >= 10**max_digits for weight in weights):
        raise OperationDomainValidationError(
            location=("weights",),
            code="matroid.weights.digits",
            message=f"matroid weights must have fewer than {max_digits} decimal digits",
        )
    canonical_function = MatroidWeightFunction(
        ground_axis=matroid.ground_axis, values=weights
    )
    return weights, canonical_function


def _admit_weight_basis(
    matroid: LinearMatroid, weight_function: MatroidWeightFunction
) -> tuple[tuple[int, ...], MatroidWeightFunction]:
    """Check the weight digit and rank-work bounds before greedy expansion."""
    weights, canonical_function = _canonical_weight_function(matroid, weight_function)
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
    return weights, canonical_function


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
    matroid: LinearMatroid, weight_function: MatroidWeightFunction
) -> MaximumWeightBasisResult:
    """Run the deterministic greedy kernel and replay exchange optimality.

    Selects elements in decreasing-weight order (ties by increasing index),
    keeping each element that preserves independence. Before return, replays
    that the selection is a basis and that no valid single-element exchange
    strictly improves the total weight.
    """
    canonical_weights, canonical_function = _admit_weight_basis(
        matroid, weight_function
    )
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
        canonical_function,
        tuple(sorted(basis)),
        total,
        full_rank,
        order,
        tuple(ledger),
    )


def verify_maximum_weight_basis(claim: MaximumWeightBasisResult) -> bool:
    """Replay greedy selection and exchange optimality for a serialized claim."""

    try:
        return (
            maximum_weight_basis_result(claim.matroid, claim.weight_function) == claim
        )
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


def maximum_weight_independent_set_result(
    matroid: LinearMatroid,
    weight_function: MatroidWeightFunction,
    *,
    max_digits: int = MAX_WEIGHT_DIGITS,
) -> MaximumWeightIndependentSetResult:
    """Return a maximum-weight independent set by positive-weight greedy scan.

    The matroid greedy theorem applies to arbitrary integer weights after
    deleting nonpositive elements. This result is distinct from a basis:
    negative weights are never forced into the returned set.
    """
    if not isinstance(matroid, LinearMatroid):
        raise OperationDomainValidationError(
            location=("matroid",),
            code="matroid.carrier",
            message="matroid must be a LinearMatroid",
        )
    if not isinstance(weight_function, MatroidWeightFunction):
        raise OperationDomainValidationError(
            location=("weight_function",),
            code="matroid.weights.carrier",
            message="weight_function must be a canonical MatroidWeightFunction",
        )
    canonical_weights, canonical_function, work, output_units = (
        _prepare_maximum_weight_independent_set(
            matroid, weight_function, max_digits=max_digits
        )
    )
    _admit_maximum_weight_independent_set(work, output_units)
    return _maximum_weight_independent_set_admitted(
        matroid, canonical_weights, canonical_function
    )


def _prepare_maximum_weight_independent_set(
    matroid: LinearMatroid,
    weight_function: MatroidWeightFunction,
    *,
    max_digits: int = MAX_WEIGHT_DIGITS,
) -> tuple[tuple[int, ...], MatroidWeightFunction, int, int]:
    """Canonicalize and measure one greedy phase without rank expansion."""
    canonical_weights, canonical_function = _canonical_weight_function(
        matroid, weight_function, max_digits=max_digits
    )
    rows = len(matroid.matrix.entries)
    n = matroid.ground_size
    work = (sum(weight > 0 for weight in canonical_weights) + 1) * _rank_work(rows, n)
    output_units = 3 * n + sum(len(str(abs(weight))) for weight in canonical_weights)
    return canonical_weights, canonical_function, work, output_units


def _admit_maximum_weight_independent_set(work: int, output_units: int) -> None:
    if (
        work > MAX_CLOSURE_RANK_WORK
        or output_units > MAX_MAXIMUM_WEIGHT_INDEPENDENT_SET_OUTPUT_UNITS
    ):
        raise OperationResourceAdmissionError(
            location=("matroid", "weights"),
            code="matroid.maximum_weight_independent_set.work_bound",
            message=(
                "maximum-weight independent-set rank work or output exceeds "
                "the exact work envelope"
            ),
        )


def _maximum_weight_independent_set_admitted(
    matroid: LinearMatroid,
    canonical_weights: tuple[int, ...],
    canonical_function: MatroidWeightFunction,
) -> MaximumWeightIndependentSetResult:
    """Run greedy selection after its caller has admitted aggregate work."""
    order = tuple(
        sorted(
            (i for i, weight in enumerate(canonical_weights) if weight > 0),
            key=lambda i: (-canonical_weights[i], i),
        )
    )
    selected: list[int] = []
    for element in order:
        if _independent(matroid, [*selected, element]):
            selected.append(element)
    selected_rank = (
        pf_rank(_selected_columns_matrix(matroid, selected)) if selected else 0
    )
    if selected_rank != len(selected):
        raise OperationDomainValidationError(
            location=("matroid", "independent_set"),
            code="matroid.maximum_weight_independent_set.feasibility",
            message="greedy selection failed the exact rank feasibility check",
        )
    canonical_selected = tuple(sorted(selected))
    total = sum(canonical_weights[i] for i in canonical_selected)
    return MaximumWeightIndependentSetResult._from_kernel(
        matroid,
        canonical_function,
        canonical_selected,
        total,
        selected_rank,
        order,
    )


def verify_maximum_weight_independent_set(
    claim: MaximumWeightIndependentSetResult,
) -> bool:
    """Replay greedy optimization and exact source rank for a serialized claim."""
    try:
        return (
            maximum_weight_independent_set_result(
                claim.matroid,
                claim.weight_function,
                max_digits=MAX_SPLIT_WEIGHT_DIGITS,
            )
            == claim
        )
    except (
        OperationDomainValidationError,
        OperationResourceAdmissionError,
        TypeError,
        ValueError,
    ):
        return False


__all__ = [
    "closure_result",
    "matroid_closure",
    "matroid_rank",
    "maximum_weight_basis_result",
    "maximum_weight_independent_set_result",
    "verify_closure",
    "verify_maximum_weight_basis",
    "verify_maximum_weight_independent_set",
]
