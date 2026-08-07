"""Search-side integer-matrix reference plugin.

Implements the maintained integer-matrix reference scenarios:
- MAT-KERNEL-001: the 2x2 matrix [[2,4],[1,2]] is singular.
- MAT-MAXDET3-001: maximize |det A| over 3x3 matrices with entries in {-1,1}.

All outputs are unverified search results; checkers replay evidence separately.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterator
from copy import deepcopy
from fractions import Fraction
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.plugin_matrices import (
    MatrixCandidate,
    MatrixCapabilityRequest,
    MatrixClaim,
    MatrixEnumerationRequest,
    MatrixMaterializeRequest,
    MatrixReductionRequest,
    MatrixScope,
    MatrixTransformRequest,
)

MAX_SEARCHED_MATRICES = 65_536

# ---------------------------------------------------------------------------
# Canonical helpers
# ---------------------------------------------------------------------------


def _to_int(value: Any) -> int:
    return int(value)


def _to_int_matrix(entries: Any) -> list[list[int]]:
    if not isinstance(entries, (list, tuple)):
        raise ValueError("entries must be a list of rows")
    matrix: list[list[int]] = []
    for row in entries:
        if not isinstance(row, (list, tuple)):
            raise ValueError("each row must be a list")
        matrix.append([_to_int(x) for x in row])
    return matrix


def _candidate_matrix(candidate: MatrixCandidate) -> list[list[int]]:
    return _to_int_matrix(candidate.entries)


def _canonical_rational(frac: Fraction) -> dict[str, str]:
    return {
        "num": format_canonical_integer(frac.numerator),
        "den": format_canonical_integer(frac.denominator),
    }


def _enumerate_typed(
    bounds: MatrixScope,
    *,
    page_size: int,
    offset: int,
) -> dict[str, Any]:
    rows = bounds.rows
    cols = bounds.cols
    values = _scope_values(bounds)
    total = len(values) ** (rows * cols)
    stop = min(offset + page_size, total)
    candidates: list[dict[str, Any]] = []
    for flat_index in range(offset, stop):
        index = flat_index
        flat: list[str] = []
        for _ in range(rows * cols):
            flat.append(str(values[index % len(values)]))
            index //= len(values)
        entries = [flat[row * cols : (row + 1) * cols] for row in range(rows)]
        candidates.append({"rows": rows, "cols": cols, "entries": entries})

    complete = stop >= total
    return {
        "response_version": "1",
        "candidates": candidates,
        "next_cursor": None if complete else {"offset": stop},
        "complete": complete,
        "scope": {
            "rows": rows,
            "cols": cols,
            "entries": [str(value) for value in values],
            "labeled": True,
            "candidate_count": total,
        },
    }


def enumerate_candidates_capability(request: dict[str, Any]) -> dict[str, Any]:
    """Page through a finite rectangular integer-matrix scope."""

    try:
        selected = MatrixEnumerationRequest.model_validate(request)
    except ValidationError as exc:
        raise ValueError(
            "matrix enumeration request does not match its contract"
        ) from exc
    return _enumerate_typed(
        selected.bounds,
        page_size=selected.page_size,
        offset=selected.cursor.offset if selected.cursor is not None else 0,
    )


def transform_row_major_capability(request: dict[str, Any]) -> dict[str, Any]:
    """Propose a row-major representation of one integer matrix."""

    try:
        selected = MatrixTransformRequest.model_validate(request)
    except ValidationError as exc:
        raise ValueError(
            "matrix transform request does not match its contract"
        ) from exc
    rows = selected.source.rows
    cols = selected.source.cols
    values = [str(value) for row in _candidate_matrix(selected.source) for value in row]
    return {
        "response_version": "1",
        "transform_format": "matrix.row_major",
        "format_version": "1",
        "relation": "EQUIVALENT",
        "target_payload": {
            "rows": rows,
            "cols": cols,
            "values": values,
        },
        "obligation": {
            "kind": "row_major_bijection",
            "source_entry_count": rows * cols,
            "target_value_count": len(values),
        },
        "detail": "flattened entries in row-major order",
    }


# ---------------------------------------------------------------------------
# Exact linear algebra
# ---------------------------------------------------------------------------


def _det_fraction(matrix: list[list[int]]) -> Fraction:
    """Exact determinant using Fraction Gaussian elimination."""
    n = len(matrix)
    if n == 0:
        return Fraction(1)
    a = [[Fraction(x) for x in row] for row in matrix]
    det = Fraction(1)
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if a[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            return Fraction(0)
        if pivot != row:
            a[pivot], a[row] = a[row], a[pivot]
            det = -det
        piv = a[row][col]
        det *= piv
        for r in range(row + 1, n):
            if a[r][col] == 0:
                continue
            factor = a[r][col] / piv
            for c in range(col, n):
                a[r][c] -= factor * a[row][c]
        row += 1
    return det


def _kernel_vector(matrix: list[list[int]]) -> list[Fraction] | None:
    """Return a non-zero rational vector in the kernel, or None if trivial."""
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    a = [[Fraction(x) for x in row] for row in matrix]
    pivot_cols: list[int] = []
    pivot_rows: list[int] = []
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if a[i][c] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        a[pivot], a[r] = a[r], a[pivot]
        pivot_cols.append(c)
        pivot_rows.append(r)
        for i in range(r + 1, rows):
            if a[i][c] == 0:
                continue
            factor = a[i][c] / a[r][c]
            for j in range(c, cols):
                a[i][j] -= factor * a[r][j]
        r += 1

    if len(pivot_cols) == cols:
        return None

    free_cols = [c for c in range(cols) if c not in pivot_cols]
    sol = [Fraction(0) for _ in range(cols)]
    sol[free_cols[0]] = Fraction(1)

    for r_idx, c_idx in reversed(list(zip(pivot_rows, pivot_cols, strict=True))):
        total = Fraction(0)
        for j in range(c_idx + 1, cols):
            total += a[r_idx][j] * sol[j]
        sol[c_idx] = -total / a[r_idx][c_idx]

    return sol


def _is_singular(matrix: list[list[int]]) -> bool:
    return _det_fraction(matrix) == 0


# ---------------------------------------------------------------------------
# Scope enumeration
# ---------------------------------------------------------------------------


def _scope_values(scope: MatrixScope) -> list[int]:
    return [_to_int(value) for value in scope.entries]


def _scope_iterator(scope: MatrixScope) -> Iterator[tuple[int, list[list[int]]]]:
    rows = scope.rows
    cols = scope.cols
    values = _scope_values(scope)
    positions = rows * cols
    for index, combo in enumerate(itertools.product(values, repeat=positions)):
        mat = [list(combo[i * cols : (i + 1) * cols]) for i in range(rows)]
        yield index, mat


def _scope_total(scope: MatrixScope) -> int:
    values = _scope_values(scope)
    return int(len(values) ** (int(scope.rows) * int(scope.cols)))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _now_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _ok(start: float) -> dict[str, Any]:
    return {
        "execution": {"status": "COMPLETED", "runtime_ms": _now_ms(start)},
        "input": {"status": "ACCEPTED", "errors": [], "warnings": []},
        "verified": False,
    }


def _rejected(errors: list[str], start: float) -> dict[str, Any]:
    return {
        "execution": {"status": "COMPLETED", "runtime_ms": _now_ms(start)},
        "input": {"status": "REJECTED", "errors": errors, "warnings": []},
        "verified": False,
    }


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def _matrix_capability_request(request: dict[str, Any]) -> MatrixCapabilityRequest:
    try:
        return MatrixCapabilityRequest.model_validate(request)
    except ValidationError as exc:
        raise ValueError(
            "matrix capability request does not match its contract"
        ) from exc


def _matrix_reduction_request(request: dict[str, Any]) -> MatrixReductionRequest:
    try:
        return MatrixReductionRequest.model_validate(request)
    except ValidationError as exc:
        raise ValueError(
            "matrix reduction request does not match its contract"
        ) from exc


def _evaluate_typed(
    claim: MatrixClaim,
    candidate: MatrixCandidate,
) -> dict[str, Any]:
    matrix = _candidate_matrix(candidate)
    det = _det_fraction(matrix)
    if claim.predicate == "is_nonsingular":
        result: dict[str, Any] = {
            "objective": {
                "name": "determinant",
                "value": _canonical_rational(det),
            },
            "is_singular": det == 0,
            "proposed_witness": None,
            "coverage": "EXHAUSTIVE",
            "arithmetic": "EXACT_RATIONAL",
            "detail": "exact rational determinant",
        }
        if det == 0:
            vec = _kernel_vector(matrix)
            if vec is not None:
                result["proposed_witness"] = {
                    "witness_format": "matrix.kernel_vector",
                    "format_version": "1",
                    "role": "DEFEATS_CANDIDATE",
                    "payload": {"vector": [_canonical_rational(v) for v in vec]},
                }
                result["detail"] = "matrix is singular with a non-zero kernel vector"
        return result

    if claim.predicate == "maximize_absolute_determinant":
        return {
            "objective": {
                "name": "abs_determinant",
                "value": _canonical_rational(abs(det)),
            },
            "is_singular": det == 0,
            "proposed_witness": None,
            "coverage": "EXHAUSTIVE",
            "arithmetic": "EXACT_RATIONAL",
            "detail": "exact rational absolute determinant",
        }

    raise ValueError("unsupported matrix claim predicate")


def evaluate_capability(request: dict[str, Any]) -> dict[str, Any]:
    """Return one matrix evaluation in the generic evaluator contract."""

    try:
        selected = MatrixCapabilityRequest.model_validate(request)
    except ValidationError as exc:
        raise ValueError(
            "matrix evaluation request does not match its contract"
        ) from exc
    candidate_model = selected.candidate
    if candidate_model is None:
        raise ValueError("matrix evaluation request requires a candidate")
    result = _evaluate_typed(selected.claim, candidate_model)
    objective = result["objective"]
    predicate = selected.claim.predicate
    if predicate == "is_nonsingular":
        conclusion = "FALSE" if result["is_singular"] else "TRUE"
        method = "EXHAUSTIVE_FINITE"
        coverage = "EXHAUSTIVE"
    else:
        conclusion = "UNKNOWN"
        method = "BOUNDED_SEARCH"
        coverage = "BOUNDED"
    return {
        "response_version": "1",
        "conclusion": conclusion,
        "arithmetic": result["arithmetic"],
        "method": method,
        "coverage": coverage,
        "objectives": {objective["name"]: objective["value"]},
        "features": (
            {
                "rows": str(candidate_model.rows),
                "cols": str(candidate_model.cols),
            }
            if candidate_model is not None
            else {}
        ),
        "failure_classifications": (
            ["nontrivial_kernel"] if result.get("is_singular") else []
        ),
        "detail": result["detail"],
    }


def _find_kernel_witness_typed(
    candidate: MatrixCandidate | None,
) -> dict[str, Any]:
    if candidate is None:
        raise ValueError("is_nonsingular witness requires a candidate")
    matrix = _candidate_matrix(candidate)
    vec = _kernel_vector(matrix)
    if vec is None:
        return {
            "status": "SEARCH_EXHAUSTED",
            "witness": None,
            "coverage": "EXHAUSTIVE",
            "arithmetic": "EXACT_RATIONAL",
            "detail": "matrix has trivial kernel only",
        }
    return {
        "status": "FOUND",
        "witness": {"vector": [_canonical_rational(v) for v in vec]},
        "witness_format": "matrix.kernel_vector",
        "format_version": "1",
        "role": "DEFEATS_CANDIDATE",
        "coverage": "EXHAUSTIVE",
        "arithmetic": "EXACT_RATIONAL",
        "detail": "non-zero kernel vector found",
    }


def _find_maxdet_witness_typed(
    claim: MatrixClaim,
) -> dict[str, Any]:
    scope = claim.scope
    if scope is None:
        raise ValueError("maximize_absolute_determinant witness requires a scope")
    if _scope_total(scope) > MAX_SEARCHED_MATRICES:
        raise ValueError(
            f"scope exceeds witness search limit of {MAX_SEARCHED_MATRICES} candidates"
        )

    best_value = Fraction(-1)
    best_index = -1
    best_matrix: list[list[int]] = []
    for index, mat in _scope_iterator(scope):
        det = abs(_det_fraction(mat))
        if det > best_value:
            best_value = det
            best_index = index
            best_matrix = mat

    if best_index < 0:
        raise ValueError("scope contains no candidates")

    return {
        "status": "FOUND",
        "witness": {
            "matrix": {
                "rows": scope.rows,
                "cols": scope.cols,
                "entries": best_matrix,
            },
            "objective_value": _canonical_rational(best_value),
            "index": best_index,
        },
        "witness_format": "matrix.maximizer",
        "format_version": "1",
        "role": "SUPPORTS_CLAIM",
        "coverage": "EXHAUSTIVE",
        "arithmetic": "EXACT_RATIONAL",
        "detail": f"maximizer with |det| = {best_value}",
    }


def _find_witness_typed(
    claim: MatrixClaim,
    candidate: MatrixCandidate | None,
    requested_role: str,
) -> dict[str, Any]:
    if claim.predicate == "is_nonsingular":
        if requested_role != "DEFEATS_CANDIDATE":
            raise ValueError("is_nonsingular supports only DEFEATS_CANDIDATE witnesses")
        return _find_kernel_witness_typed(candidate)
    if claim.predicate == "maximize_absolute_determinant":
        if requested_role != "SUPPORTS_CLAIM":
            raise ValueError(
                "maximize_absolute_determinant supports only SUPPORTS_CLAIM witnesses"
            )
        return _find_maxdet_witness_typed(claim)
    raise ValueError("unsupported matrix claim predicate for witness search")


def find_witness_capability(request: dict[str, Any]) -> dict[str, Any]:
    """Return matrix witness search in the generic oracle contract."""

    selected = _matrix_capability_request(request)
    return _find_witness_typed(
        selected.claim,
        selected.candidate,
        selected.witness_role,
    )


def materialize(request: dict[str, Any]) -> dict[str, Any]:
    """Materialize a complete bounded family for a matrix claim."""
    start = time.monotonic()
    try:
        selected = MatrixMaterializeRequest.model_validate(request)
    except ValidationError:
        return _rejected(
            ["matrix materialization request does not match its contract"], start
        )
    if selected.claim.predicate != "maximize_absolute_determinant":
        return _rejected(
            ["matrix materialize supports maximize_absolute_determinant only"], start
        )

    scope = selected.claim.scope
    if scope is None:
        return _rejected(["missing scope"], start)
    if _scope_total(scope) > MAX_SEARCHED_MATRICES:
        return _rejected(
            [
                "matrix materialization scope exceeds the bounded search limit "
                f"of {MAX_SEARCHED_MATRICES} candidates"
            ],
            start,
        )

    family: list[dict[str, Any]] = []
    for index, mat in _scope_iterator(scope):
        family.append(
            {
                "index": index,
                "candidate": {
                    "rows": scope.rows,
                    "cols": scope.cols,
                    "entries": mat,
                },
            }
        )

    response = _ok(start)
    response["family"] = family
    response["coverage"] = "EXHAUSTIVE"
    response["arithmetic"] = "EXACT_INTEGER"
    response["detail"] = f"all {_scope_total(scope)} labeled matrices in scope"
    return response


def _singular_reduction_proposals(matrix: list[list[int]]) -> list[dict[str, Any]]:
    n = len(matrix)
    proposed: list[dict[str, Any]] = []

    for i in range(n):
        reduced = [
            [matrix[r][c] for c in range(n) if c != i] for r in range(n) if r != i
        ]
        if reduced and _is_singular(reduced) and _kernel_vector(reduced) is not None:
            proposed.append(
                {
                    "reduction_kind": "delete_row_column",
                    "index": i,
                    "objectives": {
                        "elements": (n - 1) * (n - 1),
                        "max_abs_entry": max(abs(x) for row in reduced for x in row),
                    },
                }
            )

    for i in range(n):
        for j in range(n):
            reduced = [row[:] for row in matrix]
            reduced[i][j] = 0
            if _is_singular(reduced) and _kernel_vector(reduced) is not None:
                proposed.append(
                    {
                        "reduction_kind": "zero_entry",
                        "row": i,
                        "col": j,
                        "objectives": {
                            "elements": n * n,
                            "max_abs_entry": max(
                                abs(x) for row in reduced for x in row
                            ),
                        },
                    }
                )

    proposed.sort(
        key=lambda r: (r["objectives"]["elements"], r["objectives"]["max_abs_entry"])
    )
    return proposed


def _reductions_typed(
    selected: MatrixReductionRequest,
    *,
    start: float,
) -> dict[str, Any]:
    if selected.target_kind != "candidate":
        response = _ok(start)
        response["reductions"] = []
        response["detail"] = "matrix plugin only supports candidate reduction"
        return response

    matrix = _candidate_matrix(selected.target)
    proposed = (
        _singular_reduction_proposals(matrix)
        if selected.claim.predicate == "is_nonsingular" and _is_singular(matrix)
        else []
    )

    response = _ok(start)
    response["reductions"] = proposed
    response["coverage"] = "BOUNDED"
    response["arithmetic"] = "EXACT_RATIONAL"
    response["detail"] = f"{len(proposed)} reduction(s) preserve singularity"
    return response


def reductions_capability(request: dict[str, Any]) -> dict[str, Any]:
    """Return complete reduced payloads for the generic shrinker."""

    selected = _matrix_reduction_request(request)
    target = selected.target.model_dump(mode="json")
    response = _reductions_typed(selected, start=time.monotonic())
    if response["input"]["status"] != "ACCEPTED":
        raise ValueError("; ".join(response["input"]["errors"]))
    requested = set(selected.reducers)
    objective_names = tuple(selected.objectives)
    proposals: list[dict[str, Any]] = []
    for operation in response["reductions"]:
        reducer = operation["reduction_kind"]
        if requested and reducer not in requested:
            continue
        payload = deepcopy(target)
        matrix = _to_int_matrix(payload["entries"])
        if reducer == "delete_row_column":
            index = operation["index"]
            matrix = [
                [value for column, value in enumerate(row) if column != index]
                for row_index, row in enumerate(matrix)
                if row_index != index
            ]
            payload = {
                "rows": payload["rows"] - 1,
                "cols": payload["cols"] - 1,
                "entries": matrix,
            }
        elif reducer == "zero_entry":
            matrix[operation["row"]][operation["col"]] = 0
            payload["entries"] = matrix
        else:
            continue
        proposals.append(
            {
                "reducer": reducer,
                "payload": payload,
                "objectives": {
                    name: operation["objectives"][name]
                    for name in objective_names
                    if name in operation["objectives"]
                },
            }
        )
    current_matrix = _to_int_matrix(target.get("entries", []))
    current = {
        "elements": target.get("rows", 0) * target.get("cols", 0),
        "max_abs_entry": max(
            (abs(value) for row in current_matrix for value in row),
            default=0,
        ),
    }
    return {
        "response_version": "1",
        "current_objectives": {
            name: current[name] for name in objective_names if name in current
        },
        "reductions": proposals,
        "detail": response["detail"],
    }
