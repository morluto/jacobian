"""Mathematical intent distinguishes optimization targets and interpolation bases."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchRequest


@pytest.mark.parametrize(
    "query",
    [
        "Given a finite bipartite graph and positive integer quotas on its two sides, compute a maximum flow and minimum cut with integer capacities",
        "Compute maximum flow and minimum cut with integer capacities",
    ],
)
def test_flow_cut_precedes_maximum_cut(query: str) -> None:
    ids = [
        m.operation_id
        for m in Catalog.open()
        .match(OperationMatchRequest(need=query, limit=10))
        .matches
    ]
    for operation in ("graph.flow.maximum.compute", "graph.cut.minimum_st.compute"):
        assert ids.index(operation) < ids.index("graph.cut.maximum.compute")


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Compute a maximum cut of an undirected graph", "graph.cut.maximum.compute"),
        (
            "Compute the minimum s-t cut of a capacitated graph",
            "graph.cut.minimum_st.compute",
        ),
        (
            "Compute the maximum flow of a capacitated graph",
            "graph.flow.maximum.compute",
        ),
        (
            "Compute minimum cost flow with node demands",
            "network.min_cost_flow.compute",
        ),
        ("Compute an integer binomial coefficient", "combinatorics.compute.binomial"),
        (
            "Compute exact Bernstein basis coefficients of a polynomial on a box",
            "polynomial.bernstein.coefficients.compute",
        ),
    ],
)
def test_neighboring_intents(query: str, expected: str) -> None:
    matches = Catalog.open().match(OperationMatchRequest(need=query, limit=5)).matches
    assert matches[0].operation_id == expected


@pytest.mark.parametrize(
    "query",
    [
        "Compute binomial-basis coefficients from exact rational values at consecutive integer nodes using forward finite differences",
        "Return the exact forward difference table for values 35/16, 1, 5/16, 0 at nodes 0, 1, 2, 3",
        "Recover falling factorial coefficients from values at equally spaced nodes",
        "Compute divided differences at arbitrary rational nodes",
    ],
)
def test_finite_difference_interpolation(query: str) -> None:
    ids = {
        m.operation_id
        for m in Catalog.open()
        .match(OperationMatchRequest(need=query, limit=5))
        .matches
    }
    assert ids & {
        "polynomial.interpolation.newton_form.compute",
        "polynomial.interpolation.divided_differences.compute",
    }
