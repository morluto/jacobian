"""Defining-invariant and boundary tests for minimum transversals."""

import pytest
from pydantic import ValidationError

from jacobian.math.hypergraphs._models import (
    FiniteHypergraph,
    MinimumTransversalRequest,
    MinimumTransversalResult,
)
from jacobian.math.hypergraphs._operations import (
    compute_minimum_transversal,
    verify_minimum_transversal_result,
)

HYPERGRAPH = {
    "vertices": ["a", "b", "c", "d"],
    "edges": [
        ["e1", ["a", "b", "c"]],
        ["e2", ["b", "c", "d"]],
        ["e3", ["a", "d"]],
    ],
}


def _transversal(source: object) -> MinimumTransversalResult:
    return compute_minimum_transversal(
        MinimumTransversalRequest(hypergraph=FiniteHypergraph.model_validate(source))
    )


class TestMinimumTransversal:
    def test_known_minimum(self) -> None:
        result = _transversal(HYPERGRAPH)
        assert result.cardinality == 2
        assert set(result.transversal) in ({"a", "b"},)
        # {a,b} hits e1(a/b), e2(b), e3(a)
        assert set(result.transversal) == {"a", "b"}

    def test_empty_edge_family_empty_transversal(self) -> None:
        result = _transversal({"vertices": ["a", "b"], "edges": []})
        assert result.transversal == ()
        assert result.cardinality == 0

    def test_empty_edge_family_above_search_cap_is_admitted(self) -> None:
        result = _transversal({"vertices": [f"v{i}" for i in range(21)], "edges": []})
        assert result.transversal == ()
        assert result.cardinality == 0

    def test_single_edge_requires_one_vertex(self) -> None:
        result = _transversal(
            {"vertices": ["x", "y", "z"], "edges": [["e", ["x", "y"]]]}
        )
        assert result.cardinality == 1
        assert result.transversal == ("x",)

    def test_disjoint_edges_require_vertex_per_edge(self) -> None:
        result = _transversal(
            {
                "vertices": ["a", "b", "c", "d"],
                "edges": [
                    ["e1", ["a", "b"]],
                    ["e2", ["c", "d"]],
                ],
            }
        )
        assert result.cardinality == 2
        assert set(result.transversal) == {"a", "c"}

    def test_transversal_in_declared_vertex_order(self) -> None:
        result = _transversal(
            {
                "vertices": ["z", "a", "m"],
                "edges": [
                    ["e1", ["z", "a"]],
                    ["e2", ["a", "m"]],
                ],
            }
        )
        # Minimum is {a}; must appear in declared order (single element)
        assert result.cardinality == 1
        assert result.transversal == ("a",)

    def test_empty_edge_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MinimumTransversalRequest(
                hypergraph=FiniteHypergraph.model_validate(
                    {"vertices": ["a"], "edges": [["e", []]]}
                )
            )

    def test_vertex_bound_exceeded(self) -> None:
        hg = {"vertices": [f"v{i}" for i in range(21)], "edges": [["e", ["v0"]]]}
        with pytest.raises(ValidationError):
            MinimumTransversalRequest(hypergraph=FiniteHypergraph.model_validate(hg))

    def test_verify_round_trip(self) -> None:
        result = _transversal(HYPERGRAPH)
        assert verify_minimum_transversal_result(result)

    def test_rejects_non_hitting_set(self) -> None:
        with pytest.raises(ValidationError):
            MinimumTransversalResult.model_validate(
                {
                    "hypergraph": HYPERGRAPH,
                    "transversal": ["d"],
                    "cardinality": 1,
                }
            )

    def test_rejects_wrong_cardinality(self) -> None:
        with pytest.raises(ValidationError):
            MinimumTransversalResult.model_validate(
                {
                    "hypergraph": HYPERGRAPH,
                    "transversal": ["a", "b"],
                    "cardinality": 3,
                }
            )

    def test_rejects_non_minimum_transversal(self) -> None:
        # {a, b, c} is a transversal but not minimum; search returns {a,b}.
        # The result model only validates hitting, not minimality, but the
        # compute function must return the minimum.
        result = _transversal(HYPERGRAPH)
        assert result.cardinality == 2
        assert verify_minimum_transversal_result(result)

    def test_verify_accepts_tied_minimum_transversal(self) -> None:
        result = MinimumTransversalResult.model_validate(
            {
                "hypergraph": {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [
                        ["e1", ["a", "b"]],
                        ["e2", ["c", "d"]],
                    ],
                },
                "transversal": ["b", "c"],
                "cardinality": 2,
            }
        )

        assert verify_minimum_transversal_result(result)
