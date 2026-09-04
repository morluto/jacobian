"""Independent defining identities for issues 3206, 3207, and 3209."""

from itertools import combinations, pairwise
from random import Random

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.chip_firing import (
    abel_jacobi,
    parallel_step,
    q_reduced,
    stabilize,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def graph(n: int, edges: tuple[tuple[int, int], ...]) -> SimpleUndirectedGraph:
    labels = tuple(str(i) for i in range(n))
    return SimpleUndirectedGraph(
        vertices=labels,
        edges=tuple(
            (min(labels[i], labels[j]), max(labels[i], labels[j])) for i, j in edges
        ),
    )


EDGE = graph(2, ((0, 1),))
TRIANGLE = graph(3, ((0, 1), (0, 2), (1, 2)))
CYCLE = graph(4, ((0, 1), (1, 2), (2, 3), (0, 3)))
COMPLETE = graph(4, tuple(combinations(range(4), 2)))
PATH = graph(4, ((0, 1), (1, 2), (2, 3)))


def transport(
    g: SimpleUndirectedGraph, divisor: tuple[int, ...], f: tuple[int, ...]
) -> tuple[int, ...]:
    """Compute D-Lf edge by edge, independently of the production Laplacian."""
    result = list(divisor)
    for a, b in g.edges:
        i, j = g.vertices.index(a), g.vertices.index(b)
        result[i] -= f[i] - f[j]
        result[j] += f[i] - f[j]
    return tuple(result)


def assert_reduced(
    g: SimpleUndirectedGraph, divisor: tuple[int, ...], sink: str
) -> None:
    nonsink = set(g.vertices) - {sink}
    assert all(divisor[g.vertices.index(v)] >= 0 for v in nonsink)
    for size in range(1, len(nonsink) + 1):
        for subset in combinations(sorted(nonsink), size):
            assert any(
                divisor[g.vertices.index(v)]
                < sum(
                    (a == v and b not in subset) or (b == v and a not in subset)
                    for a, b in g.edges
                )
                for v in subset
            ), (divisor, subset)


@pytest.mark.parametrize("sink", EDGE.vertices)
@pytest.mark.parametrize("chips", [2, 3, 1_000_000])
def test_repeated_leaf_firing(sink: str, chips: int) -> None:
    initial = tuple(0 if v == sink else chips for v in EDGE.vertices)
    result = stabilize(EDGE, sink, initial)
    assert result.stable == tuple(chips if v == sink else 0 for v in EDGE.vertices)
    assert result.odometer == initial
    assert result.total_firings == chips


@pytest.mark.parametrize("g", [PATH, TRIANGLE, CYCLE, COMPLETE])
def test_stabilization_matches_bounded_parallel_oracle(
    g: SimpleUndirectedGraph,
) -> None:
    for sink in g.vertices:
        initial = tuple(
            0 if v == sink else 3 * (i + 1) for i, v in enumerate(g.vertices)
        )
        current = initial
        odometer = [0] * len(initial)
        for _ in range(sum(initial) * (len(initial) - 1) ** 2 + 1):
            step = parallel_step(g, sink, current)
            current = step.next_configuration
            for v in step.fired_vertices:
                odometer[g.vertices.index(v)] += 1
            if not step.fired_vertices:
                break
        else:
            pytest.fail("parallel oracle exceeded the Green-function firing bound")
        result = stabilize(g, sink, initial)
        assert result.stable == current
        assert result.odometer == tuple(odometer)
        assert transport(g, initial, result.odometer) == result.stable
        assert stabilize(g, sink, result.stable).total_firings == 0


@pytest.mark.parametrize(
    ("divisor", "expected"), [((0, 1, 1), (2, 0, 0)), ((0, -1, 0), (-2, 0, 1))]
)
def test_triangle_reduction(
    divisor: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    result = q_reduced(TRIANGLE, divisor, "0")
    assert result.reduced_divisor == expected
    assert transport(TRIANGLE, divisor, result.firing_vector) == expected


@pytest.mark.parametrize("g", [EDGE, PATH, TRIANGLE, CYCLE, COMPLETE])
def test_reduction_subset_definition_and_equivalence(g: SimpleUndirectedGraph) -> None:
    for sink in g.vertices:
        divisor = tuple((-1) ** i * (i + 2) for i in range(len(g.vertices)))
        result = q_reduced(g, divisor, sink)
        assert_reduced(g, result.reduced_divisor, sink)
        assert transport(g, divisor, result.firing_vector) == result.reduced_divisor
        assert sum(result.reduced_divisor) == sum(divisor)
        assert (
            q_reduced(g, result.reduced_divisor, sink).reduced_divisor
            == result.reduced_divisor
        )
        shifted = transport(g, divisor, tuple(10**30 * i for i in range(len(divisor))))
        assert q_reduced(g, shifted, sink).reduced_divisor == result.reduced_divisor


@pytest.mark.parametrize("g", [TRIANGLE, CYCLE, COMPLETE])
def test_every_principal_generator_maps_to_zero(g: SimpleUndirectedGraph) -> None:
    n = len(g.vertices)
    for sink in g.vertices:
        for j in range(n):
            principal = transport(g, (0,) * n, tuple(int(i == j) for i in range(n)))
            assert not any(abel_jacobi(g, principal, sink).coordinates)


def test_abel_jacobi_separates_classes_and_preserves_addition() -> None:
    for sink in TRIANGLE.vertices:
        images = [abel_jacobi(TRIANGLE, (k, -k, 0), sink).coordinates for k in range(3)]
        assert len(set(images)) == 3
        assert images[2] == tuple(2 * c % 3 for c in images[1])
        shifted = transport(TRIANGLE, (1, -1, 0), (7, -11, 23))
        assert abel_jacobi(TRIANGLE, shifted, sink).coordinates == images[1]


@pytest.mark.parametrize("g", [graph(3, ((1, 2),)), graph(2, ())])
@pytest.mark.parametrize("chips", [0, 1])
def test_disconnected_domain_is_specific_to_completion(
    g: SimpleUndirectedGraph, chips: int
) -> None:
    configuration = (chips,) * len(g.vertices)
    with pytest.raises(OperationDomainValidationError) as caught:
        stabilize(g, "0", configuration)
    assert caught.value.errors()[0]["type"] == "chip_firing.requires_connected_graph"
    with pytest.raises(OperationDomainValidationError) as caught:
        q_reduced(g, configuration, "0")
    assert caught.value.errors()[0]["type"] == "chip_firing.requires_connected_graph"
    with pytest.raises(OperationDomainValidationError) as caught:
        abel_jacobi(g, (0,) * len(g.vertices), "0")
    assert caught.value.errors()[0]["type"] == "chip_firing.requires_connected_graph"
    assert len(parallel_step(g, "0", configuration).next_configuration) == len(
        g.vertices
    )


def test_single_vertex_and_trivial_group() -> None:
    singleton = graph(1, ())
    assert stabilize(singleton, "0", (-10,)).stable == (-10,)
    assert q_reduced(singleton, (-10,), "0").reduced_divisor == (-10,)
    result = abel_jacobi(singleton, (0,), "0")
    assert result.coordinates == result.invariant_factors == ()
    assert result.nonsink_vertices == ()
    assert abel_jacobi(PATH, (1, -1, 0, 0), "1").coordinates == ()


def test_large_coefficient_reduction_and_rejected_height() -> None:
    from jacobian.math.graphs.chip_firing._models import MAX_COEFFICIENT_DIGITS

    large = 10 ** (MAX_COEFFICIENT_DIGITS - 1)
    divisor = (-large, large, 0)
    reduced = q_reduced(TRIANGLE, divisor, "0")
    assert_reduced(TRIANGLE, reduced.reduced_divisor, "0")
    assert (
        transport(TRIANGLE, divisor, reduced.firing_vector) == reduced.reduced_divisor
    )
    assert (
        abel_jacobi(TRIANGLE, divisor, "0").coordinates
        == abel_jacobi(TRIANGLE, (-large % 3, -(-large % 3), 0), "0").coordinates
    )
    excessive = 10**MAX_COEFFICIENT_DIGITS
    for operation in (q_reduced, abel_jacobi):
        with pytest.raises(OperationDomainValidationError) as caught:
            operation(TRIANGLE, (-excessive, excessive, 0), "0")
        assert caught.value.errors()[0]["type"] == "chip_firing.coefficient_bound"


def test_full_vertex_boundary_exact_cycle_coordinates() -> None:
    from jacobian.math.graphs.chip_firing._models import MAX_VERTICES

    labels = tuple(f"v{i:02d}" for i in range(MAX_VERTICES))
    g = SimpleUndirectedGraph(
        vertices=labels,
        edges=(*pairwise(labels), (labels[0], labels[-1])),
    )
    principal = (2, -1) + (0,) * (MAX_VERTICES - 3) + (-1,)
    result = abel_jacobi(g, principal, labels[-1])
    assert result.invariant_factors == (1,) * (MAX_VERTICES - 2) + (MAX_VERTICES,)
    assert result.coordinates == (0,)


@pytest.mark.parametrize("complete", [False, True])
def test_dense_fifty_vertex_column_quotient(complete: bool) -> None:
    rng = Random(3209)
    labels = tuple(f"v{i:02d}" for i in range(50))
    edges = tuple(
        (labels[i], labels[j])
        for i in range(50)
        for j in range(i + 1, 50)
        if complete or j == i + 1 or rng.random() < 0.3
    )
    g = SimpleUndirectedGraph(vertices=labels, edges=edges)
    divisor = (1, -1) + (0,) * 48
    result = abel_jacobi(g, divisor, labels[0])
    shifted = transport(g, divisor, tuple(10**30 * (i - 13) for i in range(50)))
    assert abel_jacobi(g, shifted, labels[0]).coordinates == result.coordinates
    principal = transport(g, (0,) * 50, tuple(int(i == 27) for i in range(50)))
    assert not any(abel_jacobi(g, principal, labels[0]).coordinates)
    if complete:
        assert result.invariant_factors == (1,) + (50,) * 48


def test_coupled_residual_and_interior_unit_axis() -> None:
    cases = [
        graph(5, ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4))),
        graph(
            6,
            (
                (0, 1),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 3),
                (3, 4),
                (3, 5),
                (4, 5),
            ),
        ),
    ]
    for g in cases:
        n = len(g.vertices)
        divisor = (1, -1) + (0,) * (n - 2)
        result = abel_jacobi(g, divisor, "0")
        shifted = transport(g, divisor, tuple(i * 17 for i in range(n)))
        assert abel_jacobi(g, shifted, "0").coordinates == result.coordinates


@pytest.mark.parametrize("sink", ["0", "3", "6"])
@pytest.mark.parametrize("vertices", [7, 16, 50])
def test_coupled_column_hnf_is_an_accepted_quotient(sink: str, vertices: int) -> None:
    # Four coupled directions in column HNF need not be four nontrivial
    # quotient generators. The group is cyclic of order 2520.
    g = graph(
        vertices,
        (
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 3),
            (2, 5),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 5),
            (5, 6),
            *((i, i + 1) for i in range(6, vertices - 1)),
        ),
    )
    # Attaching a tree changes presentation size but not the critical group.
    divisor = (1, -1) + (0,) * (vertices - 2)
    result = abel_jacobi(g, divisor, sink)
    assert result.invariant_factors == (1,) * (vertices - 2) + (2520,)
    assert any(result.coordinates)
    shifted = transport(
        g, divisor, tuple((-1) ** i * (2 * i + 3) for i in range(vertices))
    )
    assert abel_jacobi(g, shifted, sink).coordinates == result.coordinates
    principal = transport(g, (0,) * vertices, (0, 1) + (0,) * (vertices - 2))
    assert abel_jacobi(g, principal, sink).coordinates == (0,)
    doubled = tuple(2 * value for value in divisor)
    assert abel_jacobi(g, doubled, sink).coordinates == tuple(
        2 * value % factor
        for value, factor in zip(result.coordinates, (2520,), strict=True)
    )


def test_modular_hermite_kernel_work_matches_its_finite_loop_bound() -> None:
    from cProfile import Profile
    from math import prod

    from sympy.polys.matrices import normalforms
    from tests.fixtures.accounting import assert_charged_work_parity

    from jacobian.math.graphs.chip_firing._hermite import _column_hermite

    # K_50 has reduced Laplacian 50I-J and 50^48 spanning trees. This
    # supplements the public quotient identities with real backend work,
    # without replacing the kernel or its mathematical values.
    n = 49
    matrix = [[49 if i == j else -1 for j in range(n)] for i in range(n)]
    with Profile() as profile:
        hnf = _column_hermite(matrix, 50**48)
    assert prod(hnf[i][i] for i in range(n)) == 50**48
    names = {
        "_gcdex": "gcd",
        "add_columns_mod_R": "modular_column",
        "add_columns": "hnf_column",
    }
    executed = dict.fromkeys(names.values(), 0)
    for entry in profile.getstats():
        code = entry.code
        if (
            not isinstance(code, str)
            and code.co_filename == normalforms.__file__
            and code.co_name in names
        ):
            executed[names[code.co_name]] += entry.callcount
    assert all(count > 0 for count in executed.values())
    assert_charged_work_parity(
        charged={
            "gcd": n * (n + 1) // 2,
            "modular_column": n * (n - 1) // 2,
            "hnf_column": n * (n - 1) // 2,
        },
        executed=executed,
    )
