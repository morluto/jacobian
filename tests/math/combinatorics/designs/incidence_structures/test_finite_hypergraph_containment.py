import time
from collections import Counter
from itertools import combinations

import pytest

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.designs.incidence_structures.operations import (
    containment_profile,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs import FiniteHypergraph


def test_hypergraph_contains_duplicates_and_empty_order_zero() -> None:
    h = FiniteHypergraph(
        vertices=("a", "b", "c"),
        edges=(("e1", ("a", "b")), ("e2", ("a", "b")), ("empty", ())),
    )
    assert dict(containment_profile(h, 2).subset_profile)[("a", "b")] == 2
    assert containment_profile(h, 0).subset_profile == (((), 3),)


def test_hypergraph_bruteforce_orders() -> None:
    h = FiniteHypergraph(
        vertices=("a", "b", "c"), edges=(("e1", ("a", "b")), ("e2", ("b", "c")))
    )
    for t in range(4):
        expected = tuple(
            (subset, sum(set(subset) <= set(m) for _, m in h.edges))
            for subset in combinations(h.vertices, t)
        )
        result = containment_profile(h, t)
        assert result.subset_profile == expected
        counts = [count for _, count in expected]
        assert result.histogram == tuple(sorted(Counter(counts).items()))
        assert result.min_multiplicity == min(counts, default=0)
        assert result.max_multiplicity == max(counts, default=0)
        assert result.total_multiplicity == sum(counts)


def test_hypergraph_source_bound_refusal() -> None:
    h = FiniteHypergraph(
        vertices=tuple(f"v{i}" for i in range(100)),
        edges=tuple((f"e{i}", (f"v{i % 100}",)) for i in range(12000)),
    )
    with pytest.raises(OperationDomainValidationError):
        containment_profile(h, 10)


def test_large_indexed_hypergraph_with_cheap_order_is_accepted() -> None:
    vertices = tuple(f"v{i}" for i in range(256))
    h = FiniteHypergraph(
        vertices=vertices,
        edges=tuple((f"e{i}", (vertices[i % 256],)) for i in range(12000)),
    )
    assert containment_profile(h, 1).total_multiplicity == 12000


@pytest.mark.parametrize(
    "vertices,edges,t,expected",
    [
        ((), (), 0, (((), 0),)),
        ((), (), 1, ()),
        ((), (("empty", ()),), 0, (((), 1),)),
        (("a",), (), 1, ((("a",), 0),)),
        (("a",), (("empty", ()),), 2, ()),
    ],
)
def test_empty_and_out_of_axis_orders(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, tuple[str, ...]], ...],
    t: int,
    expected: tuple[tuple[tuple[str, ...], int], ...],
) -> None:
    result = containment_profile(FiniteHypergraph(vertices=vertices, edges=edges), t)
    assert result.subset_profile == expected
    assert result.is_constant
    assert result.constant_lambda == (expected[0][1] if expected else 0)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_shared_deadline_is_honored() -> None:
    source = FiniteHypergraph(vertices=(), edges=())
    with request_execution(time.monotonic()):
        bind_request_deadline(time.monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError):
            containment_profile(source, 0)
