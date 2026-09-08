from itertools import combinations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.designs.incidence_structures.operations import (
    containment_profile,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs import FiniteHypergraph


def test_hypergraph_contains_duplicates_and_empty_order_zero():
    h=FiniteHypergraph(vertices=("a","b","c"), edges=(("e1",("a","b")),("e2",("a","b")),("empty",())))
    r=containment_profile(h,2)
    assert dict(r.subset_profile)[("a","b")] == 2
    z=containment_profile(h,0)
    assert z.subset_profile == (((),3),)

def test_hypergraph_bruteforce_orders():
    h=FiniteHypergraph(vertices=("a","b","c"), edges=(("e1",("a","b")),("e2",("b","c"))))
    for t in range(4):
        r=containment_profile(h,t)
        expected=[]
        for subset in combinations(h.vertices,t):
            expected.append((subset,sum(set(subset)<=set(m) for _,m in h.edges)))
        assert r.subset_profile == tuple(expected)

def test_hypergraph_source_bound_refusal():
    h=FiniteHypergraph(vertices=tuple(f"v{i}" for i in range(100)), edges=tuple((f"e{i}",(f"v{i % 100}",)) for i in range(12000)))
    with pytest.raises(OperationDomainValidationError):
        containment_profile(h,10)

def test_large_indexed_hypergraph_with_cheap_order_is_accepted():
    vertices = tuple(f"v{i}" for i in range(256))
    hypergraph = FiniteHypergraph(
        vertices=vertices,
        edges=tuple((f"e{i}", (vertices[i % 256],)) for i in range(12000)),
    )
    result = containment_profile(hypergraph, 1)
    assert result.total_multiplicity == 12000
