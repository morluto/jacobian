"""Contract and mathematical tests for hypergraph independence search."""

from itertools import combinations
from pathlib import Path

import pytest
import z3  # type: ignore[import-untyped]
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs import _independence_z3
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_HYPERGRAPH_INDEPENDENCE_SOLVER_CALLS,
    DualRequest,
    FiniteHypergraph,
    HypergraphIndependenceBudget,
    HypergraphIndependenceRequest,
    HypergraphIndependenceResult,
    ParametersRequest,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._operations import (
    compute_dual,
    compute_independence_number,
    compute_parameters,
)
from jacobian.process import BoundedProcessResult, ProcessResourceLimits


def _compute(
    hypergraph: FiniteHypergraph | dict[str, object],
    *,
    max_solver_calls: int = MAX_HYPERGRAPH_INDEPENDENCE_SOLVER_CALLS,
) -> HypergraphIndependenceResult:
    return compute_independence_number(
        HypergraphIndependenceRequest.model_validate(
            {
                "hypergraph": hypergraph,
                "resource_budget": {
                    "wall_seconds": 5,
                    "max_solver_calls": max_solver_calls,
                },
            }
        )
    )


def _kernel_compute(
    hypergraph: FiniteHypergraph | dict[str, object],
    *,
    max_solver_calls: int = MAX_HYPERGRAPH_INDEPENDENCE_SOLVER_CALLS,
) -> HypergraphIndependenceResult:
    """Exercise Z3 fault injection at its isolated owner-kernel seam."""

    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3,
    )

    request = HypergraphIndependenceRequest.model_validate(
        {
            "hypergraph": hypergraph,
            "resource_budget": {
                "wall_seconds": 5,
                "max_solver_calls": max_solver_calls,
            },
        }
    )
    return _independence_z3._solve_independence_number_kernel(request)


def _independence_worker_result(payload: dict[str, object]) -> BoundedProcessResult:
    import json

    return BoundedProcessResult(
        returncode=0,
        stdout=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        stderr=b"",
        stdout_exceeded=False,
        stderr_exceeded=False,
        timed_out=False,
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
    request = HypergraphIndependenceRequest.model_validate(
        {"hypergraph": {"vertices": ["v"], "edges": [["empty", []]]}}
    )
    with pytest.raises(OperationDomainValidationError, match="empty edges"):
        compute_independence_number(request)


def test_solver_budget_is_separate_from_the_hypergraph_carrier_limit() -> None:
    source = {
        "vertices": [f"v{index}" for index in range(100)],
        "edges": [],
    }
    admitted = HypergraphIndependenceRequest.model_validate(
        {
            "hypergraph": source,
            "resource_budget": {
                "max_solver_calls": MAX_HYPERGRAPH_INDEPENDENCE_SOLVER_CALLS
            },
        }
    )

    assert len(admitted.hypergraph.vertices) == 100
    with pytest.raises(ValidationError, match="less than or equal to 16"):
        HypergraphIndependenceRequest.model_validate(
            {
                "hypergraph": source,
                "resource_budget": {
                    "max_solver_calls": MAX_HYPERGRAPH_INDEPENDENCE_SOLVER_CALLS + 1
                },
            }
        )


def test_lone_surrogate_vertex_label_rejected_before_execution() -> None:
    with pytest.raises(ValidationError):
        HypergraphIndependenceRequest.model_validate(
            {"hypergraph": {"vertices": ["\ud800"], "edges": []}}
        )


def test_lone_surrogate_edge_id_rejected_before_execution() -> None:
    with pytest.raises(ValidationError):
        HypergraphIndependenceRequest.model_validate(
            {"hypergraph": {"vertices": ["a"], "edges": [["\udbff", ["a"]]]}}
        )


def test_astral_plane_label_computes_through_edge_free_special_case() -> None:
    result = _compute({"vertices": ["\U0001d5a0"], "edges": []})
    assert result.status == "EXACT"
    assert result.independence_number == 1
    assert result.incumbent_vertices == ("\U0001d5a0",)
    assert result.termination_reason == "SPECIAL_CASE"


def test_full_structural_encoding_boundary_is_admitted() -> None:
    vertices = [f"v{index:03d}" for index in range(100)]
    request = HypergraphIndependenceRequest.model_validate(
        {
            "hypergraph": {
                "vertices": vertices,
                "edges": [[f"e{index:03d}", vertices] for index in range(100)],
            }
        }
    )
    assert len(request.hypergraph.vertices) == 100
    assert sum(len(edge) for _, edge in request.hypergraph.edges) == 10_000


def test_carrier_rejects_vertices_above_its_representation_bound() -> None:
    with pytest.raises(ValidationError):
        FiniteHypergraph.model_validate(
            {
                "vertices": [f"v{index}" for index in range(257)],
                "edges": [],
            }
        )


def test_large_ap_carrier_reaches_linear_consumers_not_independence_search() -> None:
    vertices = [str(index) for index in range(1, 213)]
    edges = [
        [
            f"ap-{index}",
            [str(start), str(start + difference), str(start + 2 * difference)],
        ]
        for difference in range(1, 106)
        for start in range(1, 213 - 2 * difference)
        for index in [
            sum(212 - 2 * prior for prior in range(1, difference)) + start - 1
        ]
    ]
    source = FiniteHypergraph.model_validate({"vertices": vertices, "edges": edges})

    assert len(source.vertices) == 212
    assert len(source.edges) == 11_130
    assert sum(len(members) for _, members in source.edges) == 33_390
    assert FiniteHypergraph.model_validate(source.model_dump(mode="json")) == source
    parameters = compute_parameters(ParametersRequest(hypergraph=source))
    assert (
        parameters.vertex_count,
        parameters.edge_count,
        parameters.total_incidences,
    ) == (
        212,
        11_130,
        33_390,
    )
    with pytest.raises(OperationDomainValidationError, match="100-vertex solver bound"):
        compute_independence_number(HypergraphIndependenceRequest(hypergraph=source))
    with pytest.raises(
        OperationDomainValidationError, match="256-vertex representation bound"
    ):
        compute_dual(DualRequest(hypergraph=source))


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
    source = FiniteHypergraph.model_validate(
        {"vertices": ["a", "b", "c"], "edges": [["triple", ["c", "a", "b"]]]}
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
    source = FiniteHypergraph.model_validate(
        {
            "vertices": ["1", "2", "3", "5", "6", "10", "15"],
            "edges": [
                ["union-6", ["2", "3", "6"]],
                ["union-10", ["2", "5", "10"]],
                ["union-15", ["3", "5", "15"]],
                ["union-30", ["6", "10", "15"]],
            ],
        }
    )
    result = _compute(source)
    brute_force = _brute_force_witness(source)
    assert result.status == "EXACT"
    assert result.independence_number == len(brute_force) == 5
    assert all(
        not set(members) <= set(result.incumbent_vertices)
        for _, members in source.edges
    )


@pytest.mark.exhaustive
def test_all_three_vertex_hypergraphs_match_exhaustive_search() -> None:
    vertices = ("a", "b", "c")
    possible_edges = tuple(
        edge
        for width in range(1, len(vertices) + 1)
        for edge in combinations(vertices, width)
    )
    for edge_mask in range(1 << len(possible_edges)):
        source = FiniteHypergraph.model_validate(
            {
                "vertices": vertices,
                "edges": [
                    [f"e{index}", edge]
                    for index, edge in enumerate(possible_edges)
                    if edge_mask & (1 << index)
                ],
            }
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
    with pytest.raises(ValidationError):
        HypergraphIndependenceResult.model_validate(payload)


def test_source_digest_preserves_distinct_unicode_wire_values() -> None:
    decomposed = _compute({"vertices": ["e\u0301"], "edges": []})
    composed = _compute({"vertices": ["\u00e9"], "edges": []})
    assert decomposed.hypergraph != composed.hypergraph
    assert decomposed.hypergraph_digest != composed.hypergraph_digest

    payload = decomposed.model_dump(mode="json")
    payload["hypergraph"] = composed.hypergraph.model_dump(mode="json")
    payload["incumbent_vertices"] = list(composed.incumbent_vertices)
    with pytest.raises(ValidationError):
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


def test_wall_expiry_returns_unknown_without_a_false_optimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3,
    )

    monkeypatch.setattr(_independence_z3, "_remaining_ms", lambda *_args: 0)
    result = _kernel_compute(
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


def test_public_independence_path_bounds_the_entire_z3_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3,
    )

    request = HypergraphIndependenceRequest.model_validate(
        {
            "hypergraph": {
                "vertices": ["a", "b", "c"],
                "edges": [["triple", ["a", "b", "c"]]],
            },
            "resource_budget": {"wall_seconds": 3, "max_solver_calls": 5},
        }
    )
    expected = _independence_z3._solve_independence_number_kernel(request)
    recorded: dict[str, object] = {}

    def complete_worker(*args: object, **kwargs: object) -> BoundedProcessResult:
        recorded["args"] = args
        recorded.update(kwargs)
        return _independence_worker_result(
            expected.model_dump(
                mode="json",
                exclude={"hypergraph", "hypergraph_digest", "resource_budget"},
            )
        )

    monkeypatch.setattr(_independence_z3, "run_bounded_process", complete_worker)

    result = compute_independence_number(request)

    assert result == expected
    timeout_seconds = recorded["timeout_seconds"]
    assert isinstance(timeout_seconds, int | float)
    assert 0 < timeout_seconds <= 3
    assert Path(str(recorded["cwd"])).name.startswith(
        "jacobian-hypergraph-independence-"
    )
    limits = recorded["resource_limits"]
    assert isinstance(limits, ProcessResourceLimits)
    assert limits.cpu_seconds == 3
    assert limits.address_space_bytes == 1_536 * 1024 * 1024
    assert limits.file_size_bytes == 1_024 * 1_024


def test_independence_worker_projection_cannot_replace_the_submitted_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3,
    )

    request = HypergraphIndependenceRequest.model_validate(
        {
            "hypergraph": {"vertices": ["a", "b"], "edges": [["pair", ["a", "b"]]]},
            "resource_budget": {"wall_seconds": 3, "max_solver_calls": 5},
        }
    )
    wrong_request = HypergraphIndependenceRequest.model_validate(
        {
            "hypergraph": {"vertices": ["x"], "edges": []},
            "resource_budget": {"wall_seconds": 3, "max_solver_calls": 5},
        }
    )
    wrong_result = _independence_z3._solve_independence_number_kernel(wrong_request)
    monkeypatch.setattr(
        _independence_z3,
        "run_bounded_process",
        lambda *_args, **_kwargs: _independence_worker_result(
            wrong_result.model_dump(
                mode="json",
                exclude={"hypergraph", "hypergraph_digest", "resource_budget"},
            )
        ),
    )

    result = compute_independence_number(request)

    assert result.status == "UNKNOWN"
    assert result.hypergraph == request.hypergraph


def test_threshold_encoding_rechecks_the_wall_budget_before_solver_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3,
    )

    class Cardinality:
        def __ge__(self, _threshold: int) -> object:
            return object()

    class Solver:
        def __init__(self) -> None:
            self.set_called = False
            self.check_called = False

        def push(self) -> None:
            return None

        def pop(self) -> None:
            return None

        def add(self, _constraint: object) -> None:
            return None

        def set(self, **_settings: int) -> None:
            self.set_called = True

        def check(self) -> object:
            self.check_called = True
            return z3.sat

    monkeypatch.setattr(_independence_z3, "_remaining_ms", lambda *_args: 0)
    solver = Solver()

    status, witness, detail = _independence_z3._check_threshold(
        solver,
        {},
        Cardinality(),
        1,
        started=0.0,
        wall_seconds=1,
        vertex_order=(),
    )

    assert status == z3.unknown
    assert witness == ()
    assert "expired during encoding" in detail
    assert not solver.set_called
    assert not solver.check_called


def test_interrupted_solver_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3,
    )

    monkeypatch.setattr(
        _independence_z3,
        "_check_threshold",
        lambda *_args: (z3.unknown, (), "interrupted"),
    )
    result = _kernel_compute(
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
    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3,
    )

    def fail(*_args: object) -> object:
        raise z3.Z3Exception("forced backend failure")

    monkeypatch.setattr(_independence_z3, "_check_threshold", fail)
    result = _kernel_compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["triple", ["a", "b", "c"]]],
        }
    )
    assert result.status == "UNKNOWN"
    assert result.independence_number is None
    assert result.termination_reason == "SOLVER_ERROR"


def test_produced_result_satisfies_structural_and_explicit_verification() -> None:
    result = _compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["triple", ["a", "b", "c"]]],
        }
    )
    restored = HypergraphIndependenceResult.model_validate(
        result.model_dump(mode="json")
    )
    assert _independence_z3.verify_independence_result(restored)


def test_forged_structural_upper_bound_requires_explicit_verification() -> None:
    source = FiniteHypergraph.model_validate(
        {
            "vertices": ["a", "b", "c", "d"],
            "edges": [
                ["ab", ["a", "b"]],
                ["ac", ["a", "c"]],
                ["ad", ["a", "d"]],
            ],
        }
    )
    result = HypergraphIndependenceResult(
        hypergraph=source,
        hypergraph_digest=_compute(source).hypergraph_digest,
        resource_budget=HypergraphIndependenceBudget(max_solver_calls=3),
        status="UNKNOWN",
        independence_number=None,
        incumbent_vertices=("a",),
        lower_bound=1,
        upper_bound=2,
        solver_calls=3,
        wall_budget_exhausted=False,
        termination_reason="SOLVER_UNKNOWN",
        detail="an independently supplied bounded claim",
    )
    assert _independence_z3.verify_independence_result(result) is False


def test_producer_rejects_infeasible_backend_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3,
    )

    def regressed(*_args: object) -> object:
        return z3.sat, ("a", "b", "c"), ""

    monkeypatch.setattr(_independence_z3, "_check_threshold", regressed)
    result = _kernel_compute(
        {
            "vertices": ["a", "b", "c"],
            "edges": [["triple", ["a", "b", "c"]]],
        }
    )
    assert result.status == "UNKNOWN"
    assert result.termination_reason == "SOLVER_ERROR"


def test_producer_rejects_forged_optimum_below_greedy_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3,
    )

    def regressed(*_args: object) -> object:
        return z3.sat, ("a",), ""

    monkeypatch.setattr(_independence_z3, "_check_threshold", regressed)
    result = _kernel_compute(
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
    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3,
    )

    def regressed(*_args: object) -> object:
        return z3.sat, ("b", "c"), ""

    monkeypatch.setattr(_independence_z3, "_check_threshold", regressed)
    result = _kernel_compute(
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
    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3,
    )

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
    result = _kernel_compute(
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
    source = FiniteHypergraph.model_validate(
        {
            "vertices": ["c", "a", "b"],
            "edges": [["ca", ["c", "a"]], ["cb", ["c", "b"]]],
        }
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
