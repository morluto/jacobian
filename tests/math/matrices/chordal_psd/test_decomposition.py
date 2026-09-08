"""Independent sum, clique, and principal-minor evidence."""

import json
from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations

import pytest
import sympy

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.matrices.chordal_psd import (
    ChordalPSDDecomposition,
    decompose_chordal_psd,
)
from jacobian.math.matrices.chordal_psd._tools import TOOLS
from jacobian.math.matrices.values import RationalMatrix, rational_matrix_from_fractions


def _case(
    rows: Sequence[Sequence[int | Fraction]], edges: Sequence[tuple[int, int]]
) -> tuple[RationalMatrix, IndexedSimpleUndirectedGraph]:
    return rational_matrix_from_fractions(
        [[Fraction(x) for x in row] for row in rows]
    ), IndexedSimpleUndirectedGraph(vertex_count=len(rows), edges=tuple(sorted(edges)))


def _check(
    matrix: RationalMatrix, graph: IndexedSimpleUndirectedGraph
) -> ChordalPSDDecomposition:
    result = decompose_chordal_psd(matrix, graph)
    n = graph.vertex_count
    total = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for term in result.terms:
        assert all((a, b) in graph.edges for a, b in combinations(term.axes, 2))
        local = sympy.Matrix(
            [[sympy.Rational(x.num, x.den) for x in row] for row in term.matrix.entries]
        )
        # Symmetric rank one with nonnegative diagonal independently proves PSD.
        assert local == local.T
        assert local.rank() == 1
        assert all(x >= 0 for x in local.diagonal())
        for i, a in enumerate(term.axes):
            for j, b in enumerate(term.axes):
                total[a][b] += term.matrix.entries[i][j].as_fraction()
    assert total == [[x.as_fraction() for x in row] for row in matrix.entries]
    assert (
        ChordalPSDDecomposition.model_validate_json(result.model_dump_json()) == result
    )
    return result


def test_path_allocates_separator() -> None:
    result = _check(*_case([[1, 1, 0], [1, 2, 1], [0, 1, 1]], [(0, 1), (1, 2)]))
    assert len(result.terms) == 2
    assert all(
        all(x.as_fraction() == 1 for row in term.matrix.entries for x in row)
        for term in result.terms
    )


@pytest.mark.parametrize(
    "rows,edges",
    [
        ([], []),
        ([[0, 0], [0, 0]], [(0, 1)]),
        ([[2, 0, 0], [0, 0, 0], [0, 0, 3]], []),
        (
            [[Fraction(2, 3), Fraction(1, 3)], [Fraction(1, 3), Fraction(1, 6)]],
            [(0, 1)],
        ),
        ([[0, 0, 0], [0, 2, 1], [0, 1, 2]], [(0, 1), (1, 2)]),
    ],
)
def test_degenerate_and_rational(
    rows: list[list[int | Fraction]], edges: list[tuple[int, int]]
) -> None:
    _check(*_case(rows, edges))


@pytest.mark.parametrize(
    "rows,edges,code",
    [
        ([[1, 1], [0, 1]], [(0, 1)], "symmetry"),
        ([[1, 1], [1, 1]], [], "support"),
        ([[0, 1], [1, 0]], [(0, 1)], "not_psd"),
        ([[1, 2], [2, 1]], [(0, 1)], "not_psd"),
        (
            [[int(i == j) for j in range(4)] for i in range(4)],
            [(0, 1), (0, 3), (1, 2), (2, 3)],
            "nonchordal",
        ),
    ],
)
def test_reject_invalid(
    rows: list[list[int]], edges: list[tuple[int, int]], code: str
) -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        decompose_chordal_psd(*_case(rows, edges))
    assert error.value.errors()[0]["type"].endswith(code)


def test_native_wire_parity() -> None:
    tool = TOOLS[0]
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    assert tool.run(request) == decompose_chordal_psd(request.matrix, request.graph)


def test_permuted_axes_remain_correct() -> None:
    # Relabelling changes the owner's deterministic PEO; correctness is invariant.
    rows = [[2, 1, 0, 0], [1, 3, 1, 1], [0, 1, 2, 0], [0, 1, 0, 1]]
    for p in ((0, 1, 2, 3), (1, 3, 0, 2), (3, 2, 1, 0)):
        permuted = [[rows[i][j] for j in p] for i in p]
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4) if permuted[i][j]]
        _check(*_case(permuted, edges))


def test_sparse_path_256() -> None:
    n = 256
    rows = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        rows[i][i] += 1
        rows[i + 1][i + 1] += 1
        rows[i][i + 1] = rows[i + 1][i] = 1
    result = _check(*_case(rows, [(i, i + 1) for i in range(n - 1)]))
    assert len(result.terms) == n - 1


def test_dense_clique_24() -> None:
    n = 24
    _check(
        *_case(
            [[2 if i == j else 1 for j in range(n)] for i in range(n)],
            list(combinations(range(n), 2)),
        )
    )


def test_output_bound() -> None:
    n = 80
    matrix, graph = _case(
        [[2 if i == j else 1 for j in range(n)] for i in range(n)],
        list(combinations(range(n), 2)),
    )
    with pytest.raises(OperationDomainValidationError, match="total local"):
        decompose_chordal_psd(matrix, graph)


def test_retained_source_counts_toward_output_bits() -> None:
    n = 1024
    matrix, graph = _case([[0] * n for _ in range(n)], ())
    with pytest.raises(OperationDomainValidationError, match="output bit"):
        decompose_chordal_psd(matrix, graph)


def test_height_bound() -> None:
    # The nontrivial Schur denominator grows beyond 65,536 bits, although
    # each input scalar is below that limit.
    matrix, graph = _case(
        [[1, Fraction(1, 3**22000)], [Fraction(1, 3**22000), Fraction(1, 2**20000)]],
        [(0, 1)],
    )
    with pytest.raises(OperationDomainValidationError, match="height"):
        decompose_chordal_psd(matrix, graph)


def test_large_singleton_and_independent_denominators() -> None:
    _check(*_case([[Fraction(1, 2**20000)]], []))
    n = 128
    rows = [
        [Fraction(1, 2**128 - 1 + 2 * i) if i == j else 0 for j in range(n)]
        for i in range(n)
    ]
    result = _check(*_case(rows, []))
    assert len(result.terms) == n


def test_extra_zero_edges_do_not_inflate_diagonal_output() -> None:
    n = 80
    result = _check(
        *_case(
            [[int(i == j) for j in range(n)] for i in range(n)],
            list(combinations(range(n), 2)),
        )
    )
    assert all(len(term.axes) == 1 for term in result.terms)
