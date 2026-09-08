"""Sparse exact LDL elimination without irrational square roots.

Vandenberghe and Andersen, Chordal Graphs and Semidefinite Optimization, §9.2.
FLINT's rational matrix interface is dense and exposes no sparse LDL kernel;
this sparse orchestration uses the standard library's exact Fraction arithmetic.
"""

from __future__ import annotations

import time
from fractions import Fraction
from math import lcm

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.chordal.operations import (
    _later_clique_failure,
    _maximum_cardinality_ordering,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.matrices.chordal_psd._models import (
    ChordalPSDDecomposition,
    CliquePSDTerm,
)
from jacobian.math.matrices.values import RationalMatrix, rational_matrix_from_fractions

_MAX_GRAPH_WORK = 16_777_216
_MAX_OUTPUT_ENTRIES = 131_072
_MAX_OUTPUT_BITS = 128_000_000
_SOURCE_CELL_JSON_BITS = 192
_RESULT_ENVELOPE_BITS = 8_192
_MAX_ARITHMETIC_WORK = 1_000_000_000_000
_MAX_SCALAR_BITS = 65_536
_WALL_SECONDS = 60.0
_JSON_HEIGHT_NUMERATOR = 12
_JSON_HEIGHT_DENOMINATOR = 5


def _json_scaled_bits(bits: int) -> int:
    """Bound decimal JSON digits of a binary integer (log10(2) < 12/25, so 2.4 bits)."""

    return (bits * _JSON_HEIGHT_NUMERATOR) // _JSON_HEIGHT_DENOMINATOR


def _reject(code: str, message: str) -> OperationDomainValidationError:
    error_type = (
        OperationResourceAdmissionError
        if code.endswith("_bound")
        else OperationDomainValidationError
    )
    return error_type(
        location=("matrix",), code=f"matrix.chordal_psd.{code}", message=message
    )


def decompose_chordal_psd(
    matrix: RationalMatrix, graph: IndexedSimpleUndirectedGraph
) -> ChordalPSDDecomposition:
    """Return at most n rational rank-one PSD clique terms summing to matrix.

    Reject nonsymmetric, non-PSD, nonchordal, or uncovered support. No negative
    mathematical conclusion is returned on resource exhaustion. All mandatory
    phases share a 60-second cooperative request deadline.
    """
    if not isinstance(matrix, RationalMatrix) or not isinstance(
        graph, IndexedSimpleUndirectedGraph
    ):
        raise TypeError("expected RationalMatrix and IndexedSimpleUndirectedGraph")
    if current_request_execution() is None:
        with request_execution(time.monotonic()):
            return _decompose(matrix, graph)
    return _decompose(matrix, graph)


def _decompose(
    matrix: RationalMatrix, graph: IndexedSimpleUndirectedGraph
) -> ChordalPSDDecomposition:
    execution = current_request_execution()
    assert execution is not None
    deadline = execution.started_at + _WALL_SECONDS
    bind_request_deadline(
        min(deadline, execution.deadline)
        if execution.deadline is not None
        else deadline
    )
    order, supports = _admit(matrix, graph)
    residual = [
        {j: value.as_fraction() for j, value in enumerate(row) if value.num != 0}
        for row in matrix.entries
    ]
    terms: list[CliquePSDTerm] = []
    for vertex, support in zip(order, supports, strict=True):
        request_checkpoint("during sparse PSD elimination")
        pivot = residual[vertex].get(vertex, Fraction(0))
        if pivot < 0:
            raise _reject("not_psd", "negative elimination pivot: matrix is not PSD")
        if not pivot:
            if any(residual[vertex].values()):
                raise _reject(
                    "not_psd",
                    "zero diagonal with nonzero residual row: matrix is not PSD",
                )
            continue
        row = [residual[vertex].get(axis, Fraction(0)) for axis in support]
        local = (
            [[pivot]]
            if len(support) == 1
            else [[left * right / pivot for right in row] for left in row]
        )
        for i, axis in enumerate(support):
            for j, other in enumerate(support):
                value = residual[axis].get(other, Fraction(0)) - local[i][j]
                if value:
                    residual[axis][other] = value
                else:
                    residual[axis].pop(other, None)
        terms.append(
            CliquePSDTerm(axes=support, matrix=rational_matrix_from_fractions(local))
        )
    request_checkpoint("before sparse PSD result construction")
    result = ChordalPSDDecomposition(matrix=matrix, graph=graph, terms=tuple(terms))
    request_checkpoint("after sparse PSD result construction")
    return result


def _admit(
    matrix: RationalMatrix, graph: IndexedSimpleUndirectedGraph
) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
    n = graph.vertex_count
    if matrix.row_count != n or matrix.column_count != n:
        raise _reject("shape", "matrix must be square on the graph vertices")
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for i, j in graph.edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    if n * n + sum(len(row) ** 2 for row in neighbors) > _MAX_GRAPH_WORK:
        raise _reject(
            "graph_work_bound", "graph ordering work exceeds the admitted bound"
        )
    order = _maximum_cardinality_ordering(n, neighbors)
    if _later_clique_failure(order, neighbors) is not None:
        raise _reject("nonchordal", "the supplied graph must be chordal")
    rank = {v: k for k, v in enumerate(order)}
    component = _scan(matrix, neighbors)
    heights = _component_heights(matrix, component)
    # Numeric components are independent even when the supplied graph adds
    # zero-valued edges between them. Restricting a PEO to each component
    # remains a PEO of its induced chordal graph.
    supports = [
        tuple(
            sorted(
                (
                    v,
                    *(
                        w
                        for w in neighbors[v]
                        if rank[w] > rank[v] and component[w] == component[v]
                    ),
                )
            )
        )
        for v in order
    ]
    cells = sum(len(support) ** 2 for support in supports)
    if cells > _MAX_OUTPUT_ENTRIES:
        raise _reject(
            "output_bound", "total local matrix entries exceed the admitted bound"
        )
    output_bits = sum(
        2 * len(support) ** 2 * _json_scaled_bits(heights[component[v]])
        for v, support in zip(order, supports, strict=True)
    )
    n = matrix.row_count
    source_bits = n * n * _SOURCE_CELL_JSON_BITS + n * 64 + _RESULT_ENVELOPE_BITS
    for row in matrix.entries:
        for value in row:
            source_bits += _json_scaled_bits(
                abs(value.num).bit_length()
            ) + _json_scaled_bits(value.den.bit_length())
    graph_bits = 64 + 2 * len(graph.edges) * max(n.bit_length(), 1)
    output_bits += source_bits + graph_bits
    arithmetic_work = sum(
        len(support) ** 2 * heights[component[v]] ** 2
        for v, support in zip(order, supports, strict=True)
    )
    if output_bits > _MAX_OUTPUT_BITS:
        raise _reject(
            "output_bound",
            "predicted exact local matrices exceed the output bit budget",
        )
    if arithmetic_work > _MAX_ARITHMETIC_WORK:
        raise _reject(
            "work_bound",
            "support updates and coefficient height exceed the arithmetic budget",
        )
    return order, supports


def _scan(matrix: RationalMatrix, neighbors: list[set[int]]) -> list[int]:
    n = matrix.row_count
    # Bound denominator clearing before building large integer intermediates.
    # Each lcm multiplies operands of at most the scalar cap, so even
    # the first rejected intermediate has at most twice that many bits.
    component = list(range(n))

    def root(v: int) -> int:
        while component[v] != v:
            component[v] = component[component[v]]
            v = component[v]
        return v

    for i, row in enumerate(matrix.entries):
        request_checkpoint("during sparse PSD admission")
        for j, value in enumerate(row):
            if value != matrix.entries[j][i]:
                raise _reject("symmetry", "matrix must be symmetric")
            q = value.as_fraction()
            if i != j and q and j not in neighbors[i]:
                raise _reject(
                    "support", "graph must cover every nonzero off-diagonal entry"
                )
            if (
                max(abs(q.numerator).bit_length(), q.denominator.bit_length())
                > _MAX_SCALAR_BITS
            ):
                raise _reject(
                    "height_bound",
                    "input coefficient height exceeds the admitted bound",
                )
            if i != j and q:
                component[root(i)] = root(j)
    return [root(v) for v in range(n)]


def _component_heights(matrix: RationalMatrix, component: list[int]) -> dict[int, int]:
    """Admit each numeric block independently; singleton terms copy one scalar."""
    denominators = dict.fromkeys(component, 1)
    numerators = dict.fromkeys(component, 1)
    sizes = dict.fromkeys(component, 0)
    for i, row in enumerate(matrix.entries):
        request_checkpoint("during component height admission")
        c = component[i]
        sizes[c] += 1
        for q in row:
            if not q.num:
                continue
            denominators[c] = lcm(denominators[c], q.den)
            if denominators[c].bit_length() > _MAX_SCALAR_BITS:
                raise _reject(
                    "height_bound",
                    "component common denominator exceeds the height bound",
                )
            numerators[c] = max(numerators[c], abs(q.num).bit_length())
    heights: dict[int, int] = {}
    for c, size in sizes.items():
        d_bits = denominators[c].bit_length()
        if size == 1:
            heights[c] = max(numerators[c], d_bits)
            continue
        # For A=D*X, a k-minor is bounded by k! max|A_ij|^k.
        # Schur entries are ratios of two minors with one D factor. Each
        # r_i*r_j/p has at most three minor factors and two D factors.
        # Four times (minor_bits+D_bits) bounds reduced numerator/denominator;
        # unreduced arithmetic uses at most twice this many bits.
        minor_bits = size * (numerators[c] + d_bits + size.bit_length()) + 1
        heights[c] = 4 * (minor_bits + d_bits)
        if heights[c] > _MAX_SCALAR_BITS:
            raise _reject(
                "height_bound",
                "predicted rational coefficient height exceeds the admitted bound",
            )
    return heights
