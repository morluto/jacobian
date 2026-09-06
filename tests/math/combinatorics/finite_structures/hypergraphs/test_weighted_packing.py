"""Defining-invariant tests for maximum-weight edge packings."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import combinations

from jacobian._exact import CanonicalRational
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    EdgeWeight,
    FiniteHypergraph,
    WeightedPackingRequest,
    WeightedPackingResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.operations import (
    maximum_edge_matching,
    maximum_weight_packing,
    verify_weighted_packing,
)


def _weight(edge_id: str, value: int) -> EdgeWeight:
    return EdgeWeight(
        edge_id=edge_id,
        weight=CanonicalRational.from_fraction(Fraction(value)),
    )


def _pack(
    vertices: list[str],
    edges: list[tuple[str, tuple[str, ...]]],
    weights: dict[str, int],
) -> WeightedPackingResult:
    hypergraph = FiniteHypergraph(
        vertices=tuple(vertices),
        edges=tuple((edge_id, members) for edge_id, members in edges),
    )
    return maximum_weight_packing(
        hypergraph,
        tuple(_weight(edge_id, weights[edge_id]) for edge_id, _ in edges),
    )


def _brute_force_optimum(
    edges: list[tuple[str, frozenset[str]]],
    weights: dict[str, Fraction],
) -> tuple[tuple[str, ...], Fraction]:
    """Reference optimum with the documented component-wise tie-break."""

    from jacobian.math.combinatorics.finite_structures.hypergraphs.operations import (
        _conflict_components,
    )

    edge_sets = tuple(members for _, members in edges)
    names = tuple(edge_id for edge_id, _ in edges)
    best_ids: list[str] = []
    best_weight = Fraction(0)
    for component in _conflict_components(edge_sets):
        local_best: tuple[str, ...] = ()
        local_weight = Fraction(0)
        for size in range(len(component) + 1):
            for combo in combinations(component, size):
                picked = [edge_sets[i] for i in combo]
                if any(
                    picked[left] & picked[right]
                    for left in range(len(picked))
                    for right in range(left + 1, len(picked))
                ):
                    continue
                total = sum((weights[names[i]] for i in combo), start=Fraction(0))
                ids = tuple(names[i] for i in combo)
                if total > local_weight or (total == local_weight and ids < local_best):
                    local_best, local_weight = ids, total
        best_ids.extend(local_best)
        best_weight += local_weight
    order = [edge_id for edge_id, _ in edges]
    return tuple(edge_id for edge_id in order if edge_id in set(best_ids)), best_weight


class TestWeightedPacking:
    def test_weight_beats_cardinality_on_nine_candidate_fixture(self) -> None:
        vertices = [
            "C:ux",
            "C:uy",
            "C:vx",
            "C:vy",
            "C:wx",
            "C:wy",
            "J:uv",
            "J:uw",
            "J:vw",
            "K:xy",
        ]
        edges: list[tuple[str, tuple[str, ...]]] = []
        for pair in ("uv", "uw", "vw"):
            left, right = pair
            edges.append((f"T_{pair}_x", (f"J:{pair}", f"C:{left}x", f"C:{right}x")))
            edges.append((f"T_{pair}_y", (f"J:{pair}", f"C:{left}y", f"C:{right}y")))
            edges.append(
                (
                    f"Q_{pair}_xy",
                    (
                        f"J:{pair}",
                        "K:xy",
                        f"C:{left}x",
                        f"C:{left}y",
                        f"C:{right}x",
                        f"C:{right}y",
                    ),
                )
            )
        weights = {
            edge_id: (5 if edge_id.startswith("Q_") else 2) for edge_id, _ in edges
        }
        result = _pack(vertices, edges, weights)
        assert result.total_weight.as_fraction() == Fraction(5)
        assert result.packing == ("Q_uv_xy",)
        hypergraph = FiniteHypergraph(
            vertices=tuple(vertices),
            edges=tuple((edge_id, members) for edge_id, members in edges),
        )
        assert maximum_edge_matching(hypergraph).count == 2

    def test_unit_weights_agree_with_cardinality(self) -> None:
        vertices = ["a", "b", "c", "d"]
        edges = [
            ("e1", ("a", "b", "c")),
            ("e2", ("b", "c", "d")),
            ("e3", ("a", "d")),
        ]
        result = _pack(vertices, edges, {edge_id: 1 for edge_id, _ in edges})
        hypergraph = FiniteHypergraph(
            vertices=tuple(vertices),
            edges=tuple((edge_id, members) for edge_id, members in edges),
        )
        assert result.total_weight.as_fraction() == Fraction(
            maximum_edge_matching(hypergraph).count
        )

    def test_zero_weights_excluded_by_tie_break(self) -> None:
        result = _pack(
            ["a", "b"],
            [("e1", ("a", "b")), ("e2", ("a",))],
            {"e1": 0, "e2": 0},
        )
        assert result.total_weight.as_fraction() == Fraction(0)
        assert result.packing == ()

    def test_empty_source(self) -> None:
        result = _pack(["a"], [], {})
        assert result.packing == ()
        assert result.total_weight.as_fraction() == Fraction(0)

    def test_duplicate_member_sets_conflict(self) -> None:
        result = _pack(
            ["a", "b"],
            [("e1", ("a", "b")), ("e2", ("a", "b"))],
            {"e1": 3, "e2": 5},
        )
        assert result.packing == ("e2",)
        assert result.total_weight.as_fraction() == Fraction(5)

    def test_exhaustive_small_instances_match_brute_force(self) -> None:
        vertices = ("a", "b", "c")
        pool = [
            frozenset(combo)
            for width in range(1, 4)
            for combo in combinations(vertices, width)
        ]
        weight_choices = (0, 1, 2)
        for mask in range(1 << len(pool)):
            family = [pool[index] for index in range(len(pool)) if mask & (1 << index)]
            edges = [
                (f"e{index}", tuple(sorted(members)))
                for index, members in enumerate(family)
            ]
            weight_pattern = {
                edge_id: weight_choices[(mask + index) % len(weight_choices)]
                for index, (edge_id, _) in enumerate(edges)
            }
            result = _pack(list(vertices), edges, weight_pattern)
            expected_ids, expected_weight = _brute_force_optimum(
                [(edge_id, frozenset(members)) for edge_id, members in edges],
                {edge_id: Fraction(value) for edge_id, value in weight_pattern.items()},
            )
            assert result.packing == expected_ids
            assert result.total_weight.as_fraction() == expected_weight

    def test_result_reparses(self) -> None:
        result = _pack(
            ["a", "b", "c"],
            [("e1", ("a", "b")), ("e2", ("b", "c"))],
            {"e1": 2, "e2": 3},
        )
        assert (
            WeightedPackingResult.model_validate(result.model_dump(mode="json"))
            == result
        )

    def test_serialized_packing_claim_is_structural_and_verifiable(self) -> None:
        result = _pack(
            ["a", "b", "c"],
            [("e1", ("a", "b")), ("e2", ("b", "c"))],
            {"e1": 2, "e2": 3},
        )
        decoded = type(result).model_validate_json(result.model_dump_json())
        assert verify_weighted_packing(decoded)

        forged = result.model_dump(mode="json")
        forged["total_weight"] = {"num": "0", "den": "1"}
        forged_decoded = type(result).model_validate_json(json.dumps(forged))
        assert not verify_weighted_packing(forged_decoded)

    def test_verifier_enforces_canonical_optimum_tie_break(self) -> None:
        result = _pack(
            ["a"],
            [("e1", ("a",)), ("e2", ("a",))],
            {"e1": 1, "e2": 1},
        )
        assert result.packing == ("e1",)
        forged = result.model_dump(mode="json")
        forged["packing"] = ["e2"]
        claim = WeightedPackingResult.model_validate(forged)
        assert not verify_weighted_packing(claim)

    def test_request_rejects_partial_and_duplicate_weights(self) -> None:
        import pytest
        from pydantic import ValidationError

        hypergraph = FiniteHypergraph(
            vertices=("a", "b"),
            edges=(("e1", ("a", "b")), ("e2", ("a",))),
        )
        with pytest.raises(ValidationError):
            WeightedPackingRequest(hypergraph=hypergraph, weights=(_weight("e1", 1),))
        with pytest.raises(ValidationError):
            WeightedPackingRequest(
                hypergraph=hypergraph,
                weights=(_weight("e1", 1), _weight("e1", 1), _weight("e2", 1)),
            )
