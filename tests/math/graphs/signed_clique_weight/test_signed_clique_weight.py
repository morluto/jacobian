"""Defining-invariant tests for signed clique-weight maximization."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.optimization._models import (
    RationalWeightedEdge,
    RationalWeightedGraph,
)
from jacobian.math.graphs.signed_clique_weight._models import (
    SignedCliqueWeightResult,
)
from jacobian.math.graphs.signed_clique_weight.operations import (
    signed_clique_weight_maximum,
)


def _edge(a: str, b: str, w: int | Fraction) -> RationalWeightedEdge:
    return RationalWeightedEdge(
        endpoints=(a, b) if a < b else (b, a),
        weight=CanonicalRational.from_fraction(Fraction(w)),
    )


def _graph(
    vertices: Sequence[str], specs: Sequence[tuple[str, str, int | Fraction]]
) -> RationalWeightedGraph:
    return RationalWeightedGraph(
        vertices=tuple(vertices),
        edges=tuple(_edge(a, b, w) for a, b, w in specs),
    )


def _brute_force(
    graph: RationalWeightedGraph,
) -> tuple[Fraction | int, tuple[str, ...]] | None:
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    weight_of = {}
    for edge in graph.edges:
        left, right = edge.endpoints
        adjacency[left].add(right)
        adjacency[right].add(left)
        weight_of[(left, right)] = edge.weight.as_fraction()
    best = None
    for size in range(2, len(graph.vertices) + 1):
        for combo in combinations(graph.vertices, size):
            if any(
                combo[right] not in adjacency[combo[left]]
                for left in range(len(combo))
                for right in range(left + 1, len(combo))
            ):
                continue
            total = sum(
                weight_of[
                    (combo[left], combo[right])
                    if combo[left] < combo[right]
                    else (combo[right], combo[left])
                ]
                for left in range(len(combo))
                for right in range(left + 1, len(combo))
            )
            if (
                best is None
                or total > best[0]
                or (total == best[0] and combo < best[1])
            ):
                best = (total, combo)
    return best


class TestSignedCliqueWeight:
    def test_k3_nonmaximal_optimum(self) -> None:
        result = signed_clique_weight_maximum(
            _graph(["a", "b", "c"], [("a", "b", 2), ("a", "c", -2), ("b", "c", -2)])
        )
        assert result.optimum_value is not None
        assert result.optimum_value.as_fraction() == Fraction(2)
        assert result.clique == ("a", "b")

    def test_all_negative_optimum(self) -> None:
        result = signed_clique_weight_maximum(_graph(["a", "b"], [("a", "b", -1)]))
        assert result.optimum_value is not None
        assert result.optimum_value.as_fraction() == Fraction(-1)
        assert result.clique == ("a", "b")

    def test_edgeless_reports_missing_optimum(self) -> None:
        result = signed_clique_weight_maximum(_graph(["a", "b"], []))
        assert result.optimum_value is None
        assert result.clique == ()

    def test_unrestricted_maximum_differs(self) -> None:
        from jacobian.math.graphs.signed_induced_weight.operations import (
            signed_induced_weight_extrema,
        )

        graph = _graph(["a", "b", "c"], [("a", "b", -1)])
        restricted = signed_clique_weight_maximum(graph)
        assert restricted.optimum_value is not None
        assert restricted.optimum_value.as_fraction() == Fraction(-1)
        unrestricted = signed_induced_weight_extrema(graph)
        assert unrestricted.maximum.value.as_fraction() == Fraction(0)

    def test_exhaustive_small_graphs_match_brute_force(self) -> None:

        vertices = ("a", "b", "c", "d")
        pairs = [
            (vertices[left], vertices[right])
            for left in range(4)
            for right in range(left + 1, 4)
        ]
        for mask in range(1 << len(pairs)):
            chosen = [pairs[i] for i in range(len(pairs)) if mask & (1 << i)]
            signs = tuple(1 if i % 2 == 0 else -1 for i in range(len(chosen)))
            specs = [
                (left, right, sign)
                for (left, right), sign in zip(chosen, signs, strict=True)
            ]
            graph = _graph(vertices, specs)
            result = signed_clique_weight_maximum(graph)
            expected = _brute_force(graph)
            if expected is None:
                assert result.optimum_value is None
                assert result.clique == ()
            else:
                assert result.optimum_value is not None
                assert result.optimum_value.as_fraction() == expected[0]
                assert result.clique == expected[1]

    def test_result_reparses(self) -> None:
        result = signed_clique_weight_maximum(_graph(["a", "b"], [("a", "b", 1)]))
        assert (
            SignedCliqueWeightResult.model_validate(result.model_dump(mode="json"))
            == result
        )

    def test_forged_binding_rejected(self) -> None:
        graph = _graph(["a", "b"], [("a", "b", 1)])
        with pytest.raises(ValidationError):
            SignedCliqueWeightResult.model_validate(
                {
                    "graph": graph.model_dump(mode="json"),
                    "optimum_value": {"num": "1", "den": "1"},
                    "clique": ("a",),
                }
            )


def test_five_overlapping_blocks_public_operation() -> None:
    from jacobian.math.graphs.signed_clique_weight._tools import TOOLS

    vertices = ["z"]
    specs: list[tuple[str, str, int | Fraction]] = []
    for t in range(5):
        block = ["z", f"a{t}", f"b{t}", f"u{t}", f"v{t}"]
        vertices.extend(block[1:])
        for i, j in combinations(range(5), 2):
            if (i, j) != (3, 4):
                specs.append((block[i], block[j], Fraction(2 if j >= 3 else -1, 3)))
    graph = _graph(vertices, specs)
    tool = TOOLS[0]
    result = tool.run(tool.request_type.model_validate({"graph": graph.model_dump()}))
    assert result.optimum_value.as_fraction() == 1
    assert result.clique == ("a0", "b0", "u0")


@pytest.mark.parametrize("weight", [-2, 0, 3])
def test_overlapping_blocks_and_bridges_match_subsets(weight: int) -> None:
    graph = _graph(
        ["z", "b", "a", "e", "d", "c", "isolated"],
        [
            ("z", "a", weight),
            ("a", "b", -1),
            ("b", "z", 2),
            ("b", "c", weight),
            ("c", "d", -3),
            ("d", "e", weight),
            ("e", "c", -1),
        ],
    )
    result = signed_clique_weight_maximum(graph)
    assert result.optimum_value is not None
    assert (result.optimum_value.as_fraction(), result.clique) == _brute_force(graph)


def test_large_biconnected_block_rejected() -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    vertices = [str(i) for i in range(21)]
    graph = _graph(
        vertices, [(vertices[i], vertices[(i + 1) % 21], 1) for i in range(21)]
    )
    with pytest.raises(OperationDomainValidationError, match="exhaustive"):
        signed_clique_weight_maximum(graph)
