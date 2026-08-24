"""Contract and mathematical tests for hypergraph independence search."""

from itertools import combinations

import pytest
import z3  # type: ignore[import-untyped]
from pydantic import ValidationError

from jacobian.math.hypergraphs._models import (
    FiniteHypergraph,
    HypergraphIndependenceBudget,
    HypergraphIndependenceRequest,
    HypergraphIndependenceResult,
)
from jacobian.math.hypergraphs._operations import compute_independence_number


def _compute(
    hypergraph: FiniteHypergraph | dict[str, object],
    *,
    max_solver_calls: int = 100,
) -> HypergraphIndependenceResult:
    return compute_independence_number(
        HypergraphIndependenceRequest(
            hypergraph=hypergraph,
            resource_budget=HypergraphIndependenceBudget(
                wall_seconds=5,
                max_solver_calls=max_solver_calls,
            ),
        )
    )


def _brute_force_witness(hypergraph: FiniteHypergraph) -> tuple[str, ...]:
    best: tuple[str, ...] = ()
    vertices = hypergraph.vertices
    edge_sets = tuple(set(members) for _, members in hypergraph.edges)
    for mask in range(1 << len(vertices)):
        candidate = tuple(
            vertex for index, vertex in enumerate(vertices) if mask & (1 << index)
        )
        candidate_set = set(candidate)
        if len(candidate) > len(best) and not any(
            edge <= candidate_set for edge in edge_sets
        ):
            best = candidate
    return best


def test_rejects_empty_hyperedge_before_solver() -> None:
    with pytest.raises(ValidationError, match="does not admit empty edges"):
        HypergraphIndependenceRequest(
            hypergraph={"vertices": ["v"], "edges": [["empty", []]]}
        )


def test_lone_surrogate_vertex_label_rejected_before_execution() -> None:
    with pytest.raises(ValidationError, match="must be valid UTF-8"):
        HypergraphIndependenceRequest(hypergraph={"vertices": ["\ud800"], "edges": []})


def test_lone_surrogate_edge_id_rejected_before_execution() -> None:
    with pytest.raises(ValidationError, match="must be valid UTF-8"):
        HypergraphIndependenceRequest(
            hypergraph={"vertices": ["a"], "edges": [["\udbff", ["a"]]]}
        )


def test_astral_plane_label_computes_through_edge_free_special_case() -> None:
    result = _compute({"vertices": ["\U0001d5a0"], "edges": []})
    assert result.status == "EXACT"
    assert result.independence_number == 1
    assert result.incumbent_vertices == ("\U0001d5a0",)
    assert result.termination_reason == "SPECIAL_CASE"


def test_full_structural_encoding_boundary_is_admitted() -> None:
    vertices = [f"v{index:03d}" for index in range(100)]
    request = HypergraphIndependenceRequest(
        hypergraph={
            "vertices": vertices,
            "edges": [[f"e{index:03d}", vertices] for index in range(100)],
        }
    )
    assert len(request.hypergraph.vertices) == 100
    assert sum(len(edge) for _, edge in request.hypergraph.edges) == 10_000


def test_vertex_boundary_rejects_immediately_unsupported_input() -> None:
    with pytest.raises(ValidationError):
        HypergraphIndependenceRequest(
            hypergraph={
                "vertices": [f"v{index}" for index in range(101)],
                "edges": [],
            }
        )


def test_edge_free_hypergraph_returns_all_vertices() -> None:
    result = _compute({"vertices": ["c", "a", "b"], "edges": []})
    assert result.status == "EXACT"
    assert result.independence_number == 3
    assert result.incumbent_vertices == ("c", "a", "b")
    assert result.lower_bound == result.upper_bound == 3
    assert result.termination_reason == "SPECIAL_CASE"


def test_empty_hypergraph_returns_zero() -> None:
    result = _compute({"vertices": [], "edges": []})
    assert result.status == "EXACT"
    assert result.independence_number == 0
    assert result.incumbent_vertices == ()
    assert result.lower_bound == result.upper_bound == 0


def test_singleton_edges_forbid_their_vertices() -> None:
    result = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["forbid-b", ["b"]], ["duplicate", ["b"]]],
        }
    )
    assert result.status == "EXACT"
    assert result.independence_number == 2
    assert result.incumbent_vertices == ("a", "c")


def test_one_three_edge_differs_from_clique_expansion_independence() -> None:
    source = FiniteHypergraph(
        vertices=["a", "b", "c"],
        edges=[["triple", ["c", "a", "b"]]],
    )
    result = _compute(source)
    assert result.independence_number == 2
    assert set(result.incumbent_vertices) < set(source.edges[0][1])
    assert len(result.incumbent_vertices) == 2


def test_isolated_vertices_remain_available_to_the_incumbent() -> None:
    result = _compute(
        {
            "vertices": ["a", "b", "isolated"],
            "edges": [["pair", ["a", "b"]]],
        }
    )
    assert result.independence_number == 2
    assert "isolated" in result.incumbent_vertices


def test_complete_three_uniform_hypergraph_has_independence_number_two() -> None:
    vertices = ["a", "b", "c", "d", "e"]
    result = _compute(
        {
            "vertices": vertices,
            "edges": [
                [f"e{index}", edge]
                for index, edge in enumerate(combinations(vertices, 3))
            ],
        }
    )
    assert result.status == "EXACT"
    assert result.independence_number == 2
    assert len(result.incumbent_vertices) == 2


def test_reordered_and_repeated_indexed_edges_have_same_invariant() -> None:
    first = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["left", ["c", "a", "b"]], ["right", ["a", "b", "c"]]],
        }
    )
    second = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["right", ["c", "b", "a"]], ["left", ["b", "a", "c"]]],
        }
    )
    assert first.independence_number == second.independence_number == 2


def test_erdos_536_reduced_forbidden_subset_fixture() -> None:
    # Reduced from sanity_small_capacity in ShouqiaoW/erdos, commit
    # d28713ac8245ca86a686b8c67370a8d19d81b242,
    # 536/numerical_verifier.py:666-699. Vertices are squarefree products
    # through 15; each edge is a triple whose prime supports have equal
    # pairwise unions.
    source = FiniteHypergraph(
        vertices=["1", "2", "3", "5", "6", "10", "15"],
        edges=[
            ["union-6", ["2", "3", "6"]],
            ["union-10", ["2", "5", "10"]],
            ["union-15", ["3", "5", "15"]],
            ["union-30", ["6", "10", "15"]],
        ],
    )
    result = _compute(source)
    brute_force = _brute_force_witness(source)
    assert result.status == "EXACT"
    assert result.independence_number == len(brute_force) == 5
    assert all(
        not set(members) <= set(result.incumbent_vertices)
        for _, members in source.edges
    )


def test_all_three_vertex_hypergraphs_match_exhaustive_search() -> None:
    vertices = ("a", "b", "c")
    possible_edges = tuple(
        edge
        for width in range(1, len(vertices) + 1)
        for edge in combinations(vertices, width)
    )
    for edge_mask in range(1 << len(possible_edges)):
        source = FiniteHypergraph(
            vertices=vertices,
            edges=[
                [f"e{index}", edge]
                for index, edge in enumerate(possible_edges)
                if edge_mask & (1 << index)
            ],
        )
        result = _compute(source)
        assert result.status == "EXACT"
        assert result.independence_number == len(_brute_force_witness(source))


def test_result_rejects_stale_source_digest() -> None:
    result = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["triple", ["a", "b", "c"]]],
        }
    )
    payload = result.model_dump(mode="json")
    payload["hypergraph"]["edges"][0][0] = "renamed"
    with pytest.raises(ValidationError, match="digest must bind"):
        HypergraphIndependenceResult.model_validate(payload)


def test_source_digest_preserves_distinct_unicode_wire_values() -> None:
    decomposed = _compute({"vertices": ["e\u0301"], "edges": []})
    composed = _compute({"vertices": ["\u00e9"], "edges": []})
    assert decomposed.hypergraph != composed.hypergraph
    assert decomposed.hypergraph_digest != composed.hypergraph_digest

    payload = decomposed.model_dump(mode="json")
    payload["hypergraph"] = composed.hypergraph.model_dump(mode="json")
    payload["incumbent_vertices"] = list(composed.incumbent_vertices)
    with pytest.raises(ValidationError, match="digest must bind"):
        HypergraphIndependenceResult.model_validate(payload)


def test_result_replays_witness_after_source_hyperedge_mutation() -> None:
    original = _compute({"vertices": ["a", "b", "c"], "edges": []})
    mutated = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["forbidden", ["a", "b", "c"]]],
        }
    )
    payload = mutated.model_dump(mode="json")
    payload.update(
        {
            "incumbent_vertices": list(original.incumbent_vertices),
            "lower_bound": 3,
            "upper_bound": 3,
            "independence_number": 3,
        }
    )
    with pytest.raises(ValidationError, match="no complete hyperedge"):
        HypergraphIndependenceResult.model_validate(payload)


def test_result_rejects_authored_upper_bound() -> None:
    result = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["triple", ["a", "b", "c"]]],
        }
    )
    payload = result.model_dump(mode="json")
    payload["upper_bound"] = 3
    with pytest.raises(ValidationError):
        HypergraphIndependenceResult.model_validate(payload)


def test_result_rejects_authored_exact_termination_reason() -> None:
    result = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["triple", ["a", "b", "c"]]],
        }
    )
    payload = result.model_dump(mode="json")
    payload["termination_reason"] = "SPECIAL_CASE"
    with pytest.raises(ValidationError, match="special-case exactness"):
        HypergraphIndependenceResult.model_validate(payload)


def test_result_rejects_authored_exact_solver_call_count() -> None:
    result = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["triple", ["a", "b", "c"]]],
        }
    )
    payload = result.model_dump(mode="json")
    payload["solver_calls"] += 1
    with pytest.raises(ValidationError, match="descending thresholds"):
        HypergraphIndependenceResult.model_validate(payload)


def test_result_replays_authored_sharper_upper_bound_against_source() -> None:
    result = _compute(
        {
            "vertices": ["a", "b", "c", "d"],
            "edges": [
                ["ab", ["a", "b"]],
                ["ac", ["a", "c"]],
                ["ad", ["a", "d"]],
            ],
        }
    )
    payload = result.model_dump(mode="json")
    payload.update(
        {
            "incumbent_vertices": ["a"],
            "lower_bound": 1,
            "upper_bound": 1,
            "independence_number": 1,
            "termination_reason": "OPTIMUM_ESTABLISHED",
        }
    )
    with pytest.raises(ValidationError, match="failed its bounded source replay"):
        HypergraphIndependenceResult.model_validate(payload)


def test_solver_call_limit_returns_only_sound_partial_bounds() -> None:
    vertices = ["a", "b", "c", "d"]
    result = _compute(
        {
            "vertices": vertices,
            "edges": [
                [f"e{index}", edge]
                for index, edge in enumerate(combinations(vertices, 2))
            ],
        },
        max_solver_calls=1,
    )
    assert result.status == "UNKNOWN"
    assert result.independence_number is None
    assert len(result.incumbent_vertices) == result.lower_bound == 1
    assert result.upper_bound == 3
    assert result.termination_reason == "SOLVER_CALL_LIMIT"
    assert result.solver_calls == 1
    assert not result.wall_budget_exhausted


def test_result_rejects_authored_unknown_termination_reason() -> None:
    vertices = ["a", "b", "c", "d"]
    result = _compute(
        {
            "vertices": vertices,
            "edges": [
                [f"e{index}", edge]
                for index, edge in enumerate(combinations(vertices, 2))
            ],
        },
        max_solver_calls=1,
    )
    payload = result.model_dump(mode="json")
    payload["termination_reason"] = "WALL_TIME"
    with pytest.raises(ValidationError, match="wall-time termination"):
        HypergraphIndependenceResult.model_validate(payload)


def test_wall_expiry_returns_unknown_without_a_false_optimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.hypergraphs import _independence_z3

    monkeypatch.setattr(_independence_z3, "_remaining_ms", lambda *_args: 0)
    result = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["triple", ["a", "b", "c"]]],
        }
    )
    assert result.status == "UNKNOWN"
    assert result.independence_number is None
    assert result.lower_bound == 2
    assert result.upper_bound == 3
    assert result.termination_reason == "WALL_TIME"
    assert result.wall_budget_exhausted


def test_interrupted_solver_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from jacobian.math.hypergraphs import _independence_z3

    monkeypatch.setattr(
        _independence_z3,
        "_check_threshold",
        lambda *_args: (z3.unknown, (), "interrupted"),
    )
    result = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["triple", ["a", "b", "c"]]],
        }
    )
    assert result.status == "UNKNOWN"
    assert result.independence_number is None
    assert result.termination_reason == "SOLVER_UNKNOWN"


def test_backend_exception_returns_typed_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.hypergraphs import _independence_z3

    def fail(*_args: object) -> object:
        raise z3.Z3Exception("forced backend failure")

    monkeypatch.setattr(_independence_z3, "_check_threshold", fail)
    result = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["triple", ["a", "b", "c"]]],
        }
    )
    assert result.status == "UNKNOWN"
    assert result.independence_number is None
    assert result.termination_reason == "SOLVER_ERROR"


def test_producer_does_not_repeat_an_established_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.hypergraphs import _independence_z3

    def fail_replay(*_args: object) -> bool:
        raise AssertionError("producer repeated an established threshold")

    monkeypatch.setattr(_independence_z3, "verify_upper_bound", fail_replay)
    result = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["triple", ["a", "b", "c"]]],
        }
    )
    assert result.status == "EXACT"
    assert result.lower_bound == result.upper_bound == 2


def test_produced_result_satisfies_the_full_independent_validator() -> None:
    result = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["triple", ["a", "b", "c"]]],
        }
    )
    assert HypergraphIndependenceResult.model_validate(result.model_dump(mode="json"))


def test_producer_rejects_infeasible_backend_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.hypergraphs import _independence_z3

    def regressed(*_args: object) -> object:
        return z3.sat, ("a", "b", "c"), ""

    monkeypatch.setattr(_independence_z3, "_check_threshold", regressed)
    with pytest.raises(ValidationError, match="no complete hyperedge"):
        _compute(
            {
                "vertices": ["a", "b", "c"],
                "edges": [["triple", ["a", "b", "c"]]],
            }
        )


def test_producer_rejects_forged_optimum_below_greedy_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.hypergraphs import _independence_z3

    def regressed(*_args: object) -> object:
        return z3.sat, ("a",), ""

    monkeypatch.setattr(_independence_z3, "_check_threshold", regressed)
    result = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["triple", ["a", "b", "c"]]],
        }
    )
    assert result.status == "UNKNOWN"
    assert result.independence_number is None
    assert result.incumbent_vertices == ("a", "b")
    assert result.upper_bound == 3
    assert result.solver_calls == 1
    assert result.termination_reason == "SOLVER_ERROR"


def test_producer_rejects_solver_calls_inconsistent_with_established_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.hypergraphs import _independence_z3

    def regressed(*_args: object) -> object:
        return z3.sat, ("b", "c"), ""

    monkeypatch.setattr(_independence_z3, "_check_threshold", regressed)
    result = _compute(
        {
            "vertices": ["a", "b", "c", "d"],
            "edges": [
                ["ab", ["a", "b"]],
                ["ac", ["a", "c"]],
                ["ad", ["a", "d"]],
            ],
        }
    )
    assert result.status == "UNKNOWN"
    assert result.independence_number is None
    assert result.upper_bound == 4
    assert result.solver_calls == 1
    assert result.termination_reason == "SOLVER_ERROR"


def test_producer_projects_solver_error_when_witness_misses_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.hypergraphs import _independence_z3

    thresholds: list[int] = []

    def regressed(
        _solver: object,
        _selected: object,
        _cardinality: object,
        threshold: int,
        *_rest: object,
    ) -> object:
        thresholds.append(threshold)
        if threshold == 3:
            return z3.unsat, (), ""
        return z3.sat, ("c",), ""

    monkeypatch.setattr(_independence_z3, "_check_threshold", regressed)
    result = _compute(
        {
            "vertices": ["c", "a", "b"],
            "edges": [["ca", ["c", "a"]], ["cb", ["c", "b"]]],
        }
    )
    assert thresholds == [3, 2]
    assert result.status == "UNKNOWN"
    assert result.independence_number is None
    assert result.incumbent_vertices == ("c",)
    assert result.lower_bound == 1
    assert result.upper_bound == 3
    assert result.solver_calls == 2
    assert not result.wall_budget_exhausted
    assert result.termination_reason == "SOLVER_ERROR"
    assert HypergraphIndependenceResult.model_validate(result.model_dump(mode="json"))


def test_produced_exact_result_meets_the_queried_threshold() -> None:
    source = FiniteHypergraph(
        vertices=["c", "a", "b"],
        edges=[["ca", ["c", "a"]], ["cb", ["c", "b"]]],
    )
    result = _compute(source)
    assert result.status == "EXACT"
    assert result.independence_number == 2
    assert len(result.incumbent_vertices) >= 2
    brute_force = _brute_force_witness(result.hypergraph)
    assert result.independence_number == len(brute_force)


def test_backend_witness_choice_is_repeatable() -> None:
    source = {
        "vertices": ["a", "b", "c", "d"],
        "edges": [["ab", ["a", "b"]], ["ac", ["a", "c"]], ["ad", ["a", "d"]]],
    }
    first = _compute(source)
    second = _compute(source)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
