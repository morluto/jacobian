"""Exact clique gluing without inverses of singular separators.

Vandenberghe--Andersen, Chordal Graphs and Semidefinite Optimization,
sections 10.1 and 10.3: PSD specified cliques characterize completability.
For a simplicial vertex v, solve C x = b on its later-neighbor separator.
The row x^T A[S,:] extends the already completed remainder. Clique PSD
ensures range membership and a nonnegative residual at v; any exact
solution works, including the RREF solution with free coordinates zero.
"""

import time
from fractions import Fraction
from math import lcm

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.chordal.operations import (
    _later_clique_failure,
    _maximum_cardinality_ordering,
)
from jacobian.math.matrices._flint import rational_rref
from jacobian.math.matrices.analysis.operations import _symmetric_inertia
from jacobian.math.matrices.completion._models import (
    ChordalPSDCompletionResult,
    CompletedChordalPSDCompletion,
    InfeasibleChordalPSDCompletion,
    PartialSymmetricRationalMatrix,
)
from jacobian.math.matrices.values import rational_matrix_from_fractions

__all__ = ["complete_chordal_psd"]


def _reject(message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("matrix",), code="matrix.chordal_completion.domain", message=message
    )


def _resource(message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("matrix",), code="matrix.chordal_completion.budget", message=message
    )


def _numeric_components(matrix: PartialSymmetricRationalMatrix) -> list[int]:
    parents = list(range(matrix.graph.vertex_count))

    def root(v: int) -> int:
        while parents[v] != v:
            parents[v] = parents[parents[v]]
            v = parents[v]
        return v

    for e in matrix.specified_entries:
        if e.row != e.column and e.value.num:
            parents[root(e.row)] = root(e.column)
    return [root(v) for v in range(len(parents))]


def _component_budget(
    values: list[CanonicalRational], sizes: list[int]
) -> tuple[int, int]:
    # Clearing denominators is bounded by the admitted source component bits.
    # For |B_ij| < 2**h every k-minor is below
    # 2**(k*(h+ceil(log2(k)))); all RREF solution coefficients share one
    # pivot-minor denominator, even when the separator is singular.
    denominator = 1
    for q in values:
        denominator = lcm(denominator, q.den)
    h = max(
        abs(q.num).bit_length() + (denominator // q.den).bit_length() for q in values
    )
    numerator_bits = max(abs(q.num).bit_length() for q in values)
    minors = [k * (h + k.bit_length()) for k in sizes if k]
    # The common output denominator divides D times the product of the
    # separator pivot minors. PSD bounds |A_ij| by the largest diagonal.
    # Empty separators contribute no determinant and create no growth.
    output_bits = denominator.bit_length() + sum(minors) + numerator_bits + 2
    clique_minor = max(
        ((k + 1) * (h + (k + 1).bit_length()) for k in sizes if k), default=0
    )
    # Inertia's Schur fractions are minor ratios; even unreduced 2x2
    # updates use at most eight minor factors. Gluing partial sums use
    # the current common denominator, one new pivot minor and log2(k).
    arithmetic_bits = max(
        8 * clique_minor,
        output_bits + 2 * max(minors, default=0) + len(sizes).bit_length(),
    )
    work = sum((k + 1) ** 3 + len(sizes) * k for k in sizes)
    if output_bits > 16000 or arithmetic_bits > 128000:
        raise _resource("exact completion exceeds rational bit budget")
    return len(sizes) ** 2 * 2 * output_bits, work * arithmetic_bits**2


def _admit(
    matrix: PartialSymmetricRationalMatrix,
) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]], list[int]]:
    n = matrix.graph.vertex_count
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for i, j in matrix.graph.edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    if n * n + sum(len(s) ** 2 for s in neighbors) > 2_000_000:
        raise _resource("chordal recognition exceeds 2,000,000 adjacency work units")
    if n * n > 65536:
        raise _resource("dense completion exceeds 65,536 output entries")
    ordering = _maximum_cardinality_ordering(n, neighbors)
    if _later_clique_failure(ordering, neighbors) is not None:
        raise _reject("the specified pattern must be chordal; no fill-in is performed")
    component = _numeric_components(matrix)
    position = {v: i for i, v in enumerate(ordering)}
    # Across numeric components all specified entries are zero. Choose zero
    # also for missing cross entries. Restricting a PEO to each induced
    # component remains a PEO, including extra zero-valued graph edges.
    separators = {
        v: tuple(
            sorted(
                w
                for w in neighbors[v]
                if position[w] > position[v] and component[w] == component[v]
            )
        )
        for v in ordering
    }
    input_bits = sum(
        abs(e.value.num).bit_length() + e.value.den.bit_length()
        for e in matrix.specified_entries
    )
    if input_bits > 1_000_000:
        raise _resource("source exceeds 1,000,000 rational component bits")
    grouped: dict[int, list[CanonicalRational]] = {c: [] for c in component}
    sizes: dict[int, list[int]] = {c: [] for c in component}
    for e in matrix.specified_entries:
        if component[e.row] == component[e.column]:
            grouped[component[e.row]].append(e.value)
    for v, separator in separators.items():
        sizes[component[v]].append(len(separator))
    budgets = [_component_budget(values, sizes[c]) for c, values in grouped.items()]
    if n * n + sum(bits for bits, _ in budgets) > 64_000_000:
        raise _resource("exact completion exceeds dense-output bit budget")
    if sum(work for _, work in budgets) > 1_000_000_000_000:
        raise _resource(
            "exact clique solves and gluing exceed rational arithmetic budget"
        )
    return ordering, separators, component


def complete_chordal_psd(
    matrix: PartialSymmetricRationalMatrix,
) -> ChordalPSDCompletionResult:
    """Complete a chordal partial matrix over QQ, or return a non-PSD clique.

    MCS ties use smallest vertex indices; each RREF solve sets free coordinates
    to zero. No minimum-rank, norm, or determinant optimum is promised.
    Nonchordal patterns are domain errors; resource excess is an execution rejection.
    """
    if not isinstance(matrix, PartialSymmetricRationalMatrix):
        raise TypeError("complete_chordal_psd expects PartialSymmetricRationalMatrix")
    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    deadline = started + 30.0
    if execution is not None and execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)

    def checkpoint(phase: str) -> None:
        request_checkpoint(phase)
        if time.monotonic() >= deadline:
            raise OperationExecutionTimeoutError(f"completion deadline expired {phase}")

    checkpoint("before admission")
    ordering, separators, component = _admit(matrix)
    checkpoint("after admission")
    n = matrix.graph.vertex_count
    source = {
        (e.row, e.column): e.value.as_fraction() for e in matrix.specified_entries
    }

    def entry(i: int, j: int) -> Fraction:
        return source[min(i, j), max(i, j)]

    for v in ordering:
        clique = tuple(sorted((v, *separators[v])))
        _, negative, _ = _symmetric_inertia(
            [[entry(i, j) for j in clique] for i in clique], checkpoint=checkpoint
        )
        if negative:
            obstruction = ChordalPSDCompletionResult(
                matrix=matrix,
                outcome=InfeasibleChordalPSDCompletion(obstruction_clique=clique),
            )
            checkpoint("after obstruction construction")
            return obstruction
    result = [[Fraction() for _ in range(n)] for _ in range(n)]
    processed: list[int] = []
    for v in reversed(ordering):
        checkpoint("during separator gluing")
        separator = separators[v]
        coefficients = [Fraction() for _ in separator]
        if separator:
            augmented = tuple(
                (*(entry(i, j) for j in separator), entry(i, v)) for i in separator
            )
            reduced, rank = rational_rref(augmented)
            checkpoint("after exact separator solve")
            for row in reduced[:rank]:
                pivot = next(i for i, value in enumerate(row) if value)
                if pivot == len(separator):
                    raise ArithmeticError(
                        "PSD clique violated separator range condition"
                    )
                coefficients[pivot] = row[-1]
        for j in processed:
            if component[v] != component[j]:
                continue
            value = sum(
                (
                    x * result[s][j]
                    for s, x in zip(separator, coefficients, strict=True)
                ),
                Fraction(),
            )
            result[v][j] = result[j][v] = value
        result[v][v] = entry(v, v)
        processed.append(v)
    checkpoint("before completion serialization")
    completion = rational_matrix_from_fractions(result)
    answer = ChordalPSDCompletionResult(
        matrix=matrix,
        outcome=CompletedChordalPSDCompletion(completion=completion),
    )
    checkpoint("after completion serialization")
    return answer
