"""Tests for exact maximal-independent-set decision."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.coloring._models import (
    EdgeKColorabilityResult,
    MaximalIndependentSetRequest,
    MaximalIndependentSetResult,
)
from jacobian.math.graphs.coloring._operations import (
    compute_maximal_independent_set_decision,
)


def _request(
    *,
    vertex_count: int,
    edges: list[list[int]],
    candidate_set: list[int],
) -> MaximalIndependentSetRequest:
    return MaximalIndependentSetRequest.model_validate(
        {
            "graph": {"vertex_count": vertex_count, "edges": edges},
            "candidate_set": candidate_set,
        }
    )


def test_path_candidate_is_maximal() -> None:
    result = compute_maximal_independent_set_decision(
        _request(
            vertex_count=4,
            edges=[[0, 1], [1, 2], [2, 3]],
            candidate_set=[0, 2],
        )
    )

    assert result == MaximalIndependentSetResult(decision="MAXIMAL")


def test_non_independent_candidate_returns_canonical_blocking_edge() -> None:
    result = compute_maximal_independent_set_decision(
        _request(
            vertex_count=3,
            edges=[[1, 0], [1, 2]],
            candidate_set=[0, 1],
        )
    )

    assert result.decision == "NOT_INDEPENDENT"
    assert result.blocking_edge == (0, 1)
    assert result.addable_vertex is None


def test_nonmaximal_candidate_returns_smallest_addable_vertex() -> None:
    result = compute_maximal_independent_set_decision(
        _request(
            vertex_count=4,
            edges=[[0, 1], [1, 2], [2, 3]],
            candidate_set=[0],
        )
    )

    assert result.decision == "INDEPENDENT_NOT_MAXIMAL"
    assert result.blocking_edge is None
    assert result.addable_vertex == 2


def test_empty_candidate_in_nonempty_graph_is_not_maximal() -> None:
    result = compute_maximal_independent_set_decision(
        _request(vertex_count=1, edges=[], candidate_set=[])
    )

    assert result.decision == "INDEPENDENT_NOT_MAXIMAL"
    assert result.addable_vertex == 0


def test_singleton_in_complete_graph_is_maximal() -> None:
    result = compute_maximal_independent_set_decision(
        _request(
            vertex_count=3,
            edges=[[0, 1], [0, 2], [1, 2]],
            candidate_set=[1],
        )
    )

    assert result.decision == "MAXIMAL"


def test_all_vertices_of_empty_graph_form_a_maximal_set() -> None:
    result = compute_maximal_independent_set_decision(
        _request(vertex_count=3, edges=[], candidate_set=[0, 1, 2])
    )

    assert result.decision == "MAXIMAL"


@pytest.mark.parametrize(
    ("candidate_set", "message"),
    [
        ([1, 0], "strictly increasing"),
        ([0, 0], "duplicate"),
        ([0, 3], "0..vertex_count-1"),
    ],
)
def test_candidate_set_is_canonical_and_in_range(
    candidate_set: list[int],
    message: str,
) -> None:
    with pytest.raises(ValidationError):
        _request(vertex_count=3, edges=[], candidate_set=candidate_set)


def test_result_rejects_witness_for_maximal_decision() -> None:
    with pytest.raises(ValidationError):
        MaximalIndependentSetResult(
            decision="MAXIMAL",
            addable_vertex=0,
        )


def test_result_requires_matching_rejection_witness() -> None:
    with pytest.raises(ValidationError):
        MaximalIndependentSetResult(decision="NOT_INDEPENDENT")
    with pytest.raises(ValidationError):
        MaximalIndependentSetResult(decision="INDEPENDENT_NOT_MAXIMAL")


class TestEdgeKColorability:
    def _petersen(self):
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        return SimpleUndirectedGraph(
            vertices=("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),
            edges=(
                ("0", "1"),
                ("1", "2"),
                ("2", "3"),
                ("3", "4"),
                ("0", "4"),
                ("5", "7"),
                ("7", "9"),
                ("6", "9"),
                ("6", "8"),
                ("5", "8"),
                ("0", "5"),
                ("1", "6"),
                ("2", "7"),
                ("3", "8"),
                ("4", "9"),
            ),
        )

    def test_petersen_not_3_edge_colorable(self):
        from jacobian.math.graphs.coloring._models import EdgeKColorabilityRequest
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_k_colorability,
        )

        result = compute_edge_k_colorability(
            EdgeKColorabilityRequest(graph=self._petersen(), colors=3),
        )
        assert result.colorable is False
        assert result.coloring is None

    def test_petersen_4_edge_colorable_and_proper(self):
        from jacobian.math.graphs.coloring._models import (
            EdgeColoringCheckRequest,
            EdgeKColorabilityRequest,
        )
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_coloring_check,
            compute_edge_k_colorability,
        )

        g = self._petersen()
        result = compute_edge_k_colorability(
            EdgeKColorabilityRequest(graph=g, colors=4)
        )
        assert result.colorable is True
        assert result.coloring is not None
        assert result.coloring.graph == g
        assert result.coloring.colors == 4
        assert len(result.coloring.coloring) == 15
        check = compute_edge_coloring_check(
            EdgeColoringCheckRequest(assignment=result.coloring),
        )
        assert check.proper is True

    def test_triangle_needs_3_edge_colors(self):
        from jacobian.math.graphs.coloring._models import EdgeKColorabilityRequest
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_k_colorability,
        )
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        g = SimpleUndirectedGraph(
            vertices=("0", "1", "2"),
            edges=(("0", "1"), ("1", "2"), ("0", "2")),
        )
        assert (
            compute_edge_k_colorability(
                EdgeKColorabilityRequest(graph=g, colors=2)
            ).colorable
            is False
        )
        assert (
            compute_edge_k_colorability(
                EdgeKColorabilityRequest(graph=g, colors=3)
            ).colorable
            is True
        )


class TestEdgeColoringRequestSchema:
    MAX_VERTICES = 64

    def _path_graph(self, vertex_count: int):
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        labels = tuple(f"{index:02d}" for index in range(vertex_count))
        edges = tuple((labels[i], labels[i + 1]) for i in range(vertex_count - 1))
        return SimpleUndirectedGraph(vertices=labels, edges=edges)

    def test_published_schema_advertises_operation_specific_bounds(self):
        from jacobian.math.graphs.coloring._models import (
            EdgeColoringCheckRequest,
            EdgeKColorabilityRequest,
        )

        decide_graph = EdgeKColorabilityRequest.model_json_schema()["properties"][
            "graph"
        ]
        assert decide_graph["properties"]["vertices"]["maxItems"] == self.MAX_VERTICES
        assert decide_graph["properties"]["edges"]["maxItems"] == 2016
        assert "at most 64 vertices" in decide_graph["description"]

        check_schema = EdgeColoringCheckRequest.model_json_schema()
        assignment = check_schema["$defs"]["EdgeColoringAssignment"]
        assert check_schema["properties"]["assignment"]["$ref"] == (
            "#/$defs/EdgeColoringAssignment"
        )
        assert assignment["properties"]["graph"] == decide_graph

    def test_21_vertex_graph_is_schema_bound_and_validator_rejected(self):
        from jacobian.math.graphs.coloring._models import (
            EdgeColoringCheckRequest,
            EdgeKColorabilityRequest,
        )

        g = self._path_graph(65)
        assert len(g.vertices) == 65 < 256
        with pytest.raises(ValidationError):
            EdgeKColorabilityRequest.model_validate(
                {"graph": g.model_dump(), "colors": 3}
            )
        with pytest.raises(ValidationError):
            EdgeColoringCheckRequest.model_validate(
                {
                    "assignment": {
                        "graph": g.model_dump(),
                        "colors": 3,
                        "coloring": tuple(range(64)),
                    }
                }
            )

    def test_direct_construction_still_enforces_vertex_bound(self):
        from jacobian.math.graphs.coloring._models import EdgeKColorabilityRequest

        with pytest.raises(ValidationError):
            EdgeKColorabilityRequest(graph=self._path_graph(65), colors=3)

    def test_20_vertex_boundary_request_is_admitted(self):
        from jacobian.math.graphs.coloring._models import (
            EdgeColoringAssignment,
            EdgeColoringCheckRequest,
            EdgeKColorabilityRequest,
        )
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_coloring_check,
            compute_edge_k_colorability,
        )

        g = self._path_graph(20)
        decided = compute_edge_k_colorability(
            EdgeKColorabilityRequest(graph=g, colors=2)
        )
        assert decided.colorable is True
        checked = compute_edge_coloring_check(
            EdgeColoringCheckRequest(
                assignment=EdgeColoringAssignment(
                    graph=g, colors=2, coloring=(0, 1) * 9 + (0,)
                )
            )
        )
        assert checked.proper is True


class TestEdgeColoringCheck:
    def _petersen(self):
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        return SimpleUndirectedGraph(
            vertices=("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),
            edges=(
                ("0", "1"),
                ("1", "2"),
                ("2", "3"),
                ("3", "4"),
                ("0", "4"),
                ("5", "7"),
                ("7", "9"),
                ("6", "9"),
                ("6", "8"),
                ("5", "8"),
                ("0", "5"),
                ("1", "6"),
                ("2", "7"),
                ("3", "8"),
                ("4", "9"),
            ),
        )

    def test_proper_coloring_accepted(self):
        from jacobian.math.graphs.coloring._models import (
            EdgeColoringAssignment,
            EdgeColoringCheckRequest,
        )
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_coloring_check,
        )

        g = self._petersen()
        coloring = (1, 0, 1, 3, 2, 3, 0, 3, 1, 2, 0, 2, 2, 0, 1)
        result = compute_edge_coloring_check(
            EdgeColoringCheckRequest(
                assignment=EdgeColoringAssignment(graph=g, colors=4, coloring=coloring)
            ),
        )
        assert result.proper is True
        assert result.blocking_edge is None
        assert result.assignment.colors == 4

    def test_improper_coloring_reports_blocking_edge(self):
        from jacobian.math.graphs.coloring._models import (
            EdgeColoringAssignment,
            EdgeColoringCheckRequest,
        )
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_coloring_check,
        )
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        g = SimpleUndirectedGraph(
            vertices=("0", "1", "2"),
            edges=(("0", "1"), ("1", "2")),
        )
        # edges (0,1) and (1,2) share vertex 1; same color is improper.
        result = compute_edge_coloring_check(
            EdgeColoringCheckRequest(
                assignment=EdgeColoringAssignment(graph=g, colors=2, coloring=(0, 0))
            ),
        )
        assert result.proper is False
        assert result.blocking_edge is not None
        assert result.conflicting_edge is not None

    def test_rejects_wrong_length_assignment(self):
        from jacobian.math.graphs.coloring._models import EdgeColoringAssignment
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        g = SimpleUndirectedGraph(
            vertices=("0", "1"),
            edges=(("0", "1"),),
        )
        with pytest.raises(ValidationError):
            EdgeColoringAssignment(graph=g, colors=2, coloring=(0, 1))


def _petersen_graph():
    from jacobian.math.graphs.values import SimpleUndirectedGraph

    return SimpleUndirectedGraph(
        vertices=("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),
        edges=(
            ("0", "1"),
            ("1", "2"),
            ("2", "3"),
            ("3", "4"),
            ("0", "4"),
            ("5", "7"),
            ("7", "9"),
            ("6", "9"),
            ("6", "8"),
            ("5", "8"),
            ("0", "5"),
            ("1", "6"),
            ("2", "7"),
            ("3", "8"),
            ("4", "9"),
        ),
    )


class TestCanonicalEdgeColoringValue:
    """The k_decide witness composes into graph.edge_coloring.check unchanged."""

    def test_serialized_witness_feeds_the_checker_unchanged(self):
        from jacobian.math.graphs.coloring._models import (
            EdgeColoringCheckRequest,
            EdgeKColorabilityRequest,
            EdgeKColorabilityResult,
        )
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_coloring_check,
            compute_edge_k_colorability,
        )

        result = compute_edge_k_colorability(
            EdgeKColorabilityRequest(graph=_petersen_graph(), colors=4)
        )
        serialized = EdgeKColorabilityResult.model_validate(result.model_dump())
        request = EdgeColoringCheckRequest.model_validate(
            {"assignment": serialized.model_dump()["coloring"]}
        )
        check = compute_edge_coloring_check(request)
        assert check.proper is True
        assert check.assignment == serialized.coloring

    def test_forged_witness_must_bind_the_result_source(self):
        from jacobian.math.graphs.coloring._models import (
            EdgeColoringAssignment,
            EdgeKColorabilityResult,
        )

        triangle = _triangle_graph()
        petersen = _petersen_graph()
        with pytest.raises(ValidationError):
            EdgeKColorabilityResult(
                graph=petersen,
                colors=3,
                status="DECIDED",
                colorable=True,
                coloring=EdgeColoringAssignment(
                    graph=triangle, colors=3, coloring=(0, 1, 2)
                ),
                edge_count=len(petersen.edges),
            )

    def test_forged_improper_witness_rejected_for_colorable_claim(self):
        from jacobian.math.graphs.coloring._models import (
            EdgeColoringAssignment,
            EdgeKColorabilityResult,
        )

        g = _triangle_graph()
        with pytest.raises(ValidationError):
            EdgeKColorabilityResult(
                graph=g,
                colors=2,
                status="DECIDED",
                colorable=True,
                coloring=EdgeColoringAssignment(graph=g, colors=2, coloring=(0, 0, 1)),
                edge_count=3,
            )


def _triangle_graph():
    from jacobian.math.graphs.values import SimpleUndirectedGraph

    return SimpleUndirectedGraph(
        vertices=("0", "1", "2"),
        edges=(("0", "1"), ("1", "2"), ("0", "2")),
    )


class TestSolverConflictBudget:
    def test_budget_exceeded_returns_typed_unknown_outcome(self):
        """A conflict budget of 1 cannot decide the Petersen graph; the
        operation must report SOLVER_BUDGET_EXCEEDED with no colorability
        claim instead of an unbounded wait or a false negative."""
        from jacobian.math.graphs.coloring._models import (
            EdgeKColorabilityRequest,
            EdgeKColorabilityResult,
        )
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_k_colorability,
        )

        request = EdgeKColorabilityRequest(
            graph=_petersen_graph(), colors=3, solver_conflicts=1
        )
        result = compute_edge_k_colorability(request)
        assert result.status == "SOLVER_BUDGET_EXCEEDED"
        assert result.colorable is None
        assert result.coloring is None
        assert EdgeKColorabilityResult.model_validate(result.model_dump()) == result

    def test_forged_budget_exceeded_rejected_when_decidable(self):
        """An authored budget-exceeded label on a trivially decidable graph
        must not validate."""
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        with pytest.raises(ValidationError):
            EdgeKColorabilityResult(
                graph=SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),)),
                colors=2,
                solver_conflicts=1000,
                status="SOLVER_BUDGET_EXCEEDED",
                colorable=None,
                coloring=None,
                edge_count=1,
            )

    def test_budget_exceeded_cannot_claim_colorable(self):
        petersen = _petersen_graph()
        with pytest.raises(ValidationError):
            EdgeKColorabilityResult(
                graph=petersen,
                colors=3,
                solver_conflicts=100000,
                status="SOLVER_BUDGET_EXCEEDED",
                colorable=False,
                coloring=None,
                edge_count=len(petersen.edges),
            )

    def test_negative_claim_requires_explicit_unsat_within_budget(self):
        """A non-colorable claim replayed under a too-small budget (which
        returns unknown) must not validate: negatives need explicit unsat."""
        petersen = _petersen_graph()
        payload = {
            "graph": petersen.model_dump(),
            "colors": 3,
            "solver_conflicts": 1,
            "status": "DECIDED",
            "colorable": False,
            "coloring": None,
            "edge_count": len(petersen.edges),
        }
        with pytest.raises(ValidationError):
            EdgeKColorabilityResult.model_validate(payload)

    def test_default_budget_still_decides_petersen_negative(self):
        from jacobian.math.graphs.coloring._models import EdgeKColorabilityRequest
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_k_colorability,
        )

        result = compute_edge_k_colorability(
            EdgeKColorabilityRequest(graph=_petersen_graph(), colors=3)
        )
        assert result.status == "DECIDED"
        assert result.colorable is False

    def test_decided_negative_runs_the_bounded_solver_exactly_once(self, monkeypatch):
        """The producing solve must not pay a second full-budget replay:
        one declared budget covers all solver work on the request."""
        import jacobian.math.graphs.coloring._models as coloring_models
        from jacobian.math.graphs.coloring._models import EdgeKColorabilityRequest
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_k_colorability,
        )

        calls: list[int] = []
        original = coloring_models._run_edge_coloring_solver

        def counting(graph, colors, solver_conflicts):
            calls.append(solver_conflicts)
            return original(graph, colors, solver_conflicts)

        monkeypatch.setattr(coloring_models, "_run_edge_coloring_solver", counting)
        result = compute_edge_k_colorability(
            EdgeKColorabilityRequest(graph=_petersen_graph(), colors=3)
        )
        assert result.status == "DECIDED"
        assert result.colorable is False
        assert calls == [result.solver_conflicts]

    def test_budget_exceeded_outcome_runs_the_bounded_solver_exactly_once(
        self, monkeypatch
    ):
        import jacobian.math.graphs.coloring._models as coloring_models
        from jacobian.math.graphs.coloring._models import EdgeKColorabilityRequest
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_k_colorability,
        )

        calls: list[int] = []
        original = coloring_models._run_edge_coloring_solver

        def counting(graph, colors, solver_conflicts):
            calls.append(solver_conflicts)
            return original(graph, colors, solver_conflicts)

        monkeypatch.setattr(coloring_models, "_run_edge_coloring_solver", counting)
        result = compute_edge_k_colorability(
            EdgeKColorabilityRequest(
                graph=_petersen_graph(), colors=3, solver_conflicts=1
            )
        )
        assert result.status == "SOLVER_BUDGET_EXCEEDED"
        assert result.colorable is None
        assert calls == [1]

    def test_produced_outcomes_round_trip_through_full_validation(self):
        """Results built from the producing solve must equal their fully
        validated reconstruction, so the skipped replay invariant holds."""
        from jacobian.math.graphs.coloring._models import EdgeKColorabilityRequest
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_k_colorability,
        )

        petersen = _petersen_graph()
        negative = compute_edge_k_colorability(
            EdgeKColorabilityRequest(graph=petersen, colors=3)
        )
        assert EdgeKColorabilityResult.model_validate(negative.model_dump()) == negative
        undecided = compute_edge_k_colorability(
            EdgeKColorabilityRequest(graph=petersen, colors=3, solver_conflicts=1)
        )
        assert (
            EdgeKColorabilityResult.model_validate(undecided.model_dump()) == undecided
        )


class TestVertexKColorability:
    def _k4(self):
        from jacobian.math.graphs.coloring._models import GraphEdgeList

        return GraphEdgeList(
            vertex_count=4,
            edges=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        )

    def test_triangle_decision_carries_a_proper_witness(self):
        from jacobian.math.graphs.coloring._models import KColorabilityRequest
        from jacobian.math.graphs.coloring._operations import compute_k_colorability

        result = compute_k_colorability(
            KColorabilityRequest(
                graph={"vertex_count": 3, "edges": [[0, 1], [1, 2], [2, 0]]},
                colors=3,
            )
        )
        assert result.status == "DECIDED"
        assert result.colorable is True
        assert len(result.coloring) == 3

    def test_budget_exceeded_returns_typed_unknown_outcome(self):
        """A conflict budget of 1 cannot decide the complete graph K4 under
        three colors; the operation must report SOLVER_BUDGET_EXCEEDED with
        no colorability claim instead of an unbounded wait or a false
        negative."""
        from jacobian.math.graphs.coloring._models import (
            KColorabilityRequest,
            KColorabilityResult,
        )
        from jacobian.math.graphs.coloring._operations import compute_k_colorability

        request = KColorabilityRequest(graph=self._k4(), colors=3, solver_conflicts=1)
        result = compute_k_colorability(request)
        assert result.status == "SOLVER_BUDGET_EXCEEDED"
        assert result.colorable is None
        assert result.coloring is None
        assert KColorabilityResult.model_validate(result.model_dump()) == result

    def test_forged_budget_exceeded_rejected_when_decidable(self):
        """An authored budget-exceeded label on a trivially decidable graph
        must not validate."""
        from jacobian.math.graphs.coloring._models import (
            GraphEdgeList,
            KColorabilityResult,
        )

        with pytest.raises(ValidationError):
            KColorabilityResult(
                graph=GraphEdgeList(vertex_count=2, edges=((0, 1),)),
                colors=2,
                solver_conflicts=1000,
                status="SOLVER_BUDGET_EXCEEDED",
                colorable=None,
                coloring=None,
                vertex_count=2,
            )

    def test_budget_exceeded_cannot_claim_colorable(self):
        from jacobian.math.graphs.coloring._models import KColorabilityResult

        k4 = self._k4()
        with pytest.raises(ValidationError):
            KColorabilityResult(
                graph=k4,
                colors=3,
                solver_conflicts=100000,
                status="SOLVER_BUDGET_EXCEEDED",
                colorable=False,
                coloring=None,
                vertex_count=4,
            )

    def test_negative_claim_requires_explicit_unsat_within_budget(self):
        """A non-colorable claim replayed under a too-small budget (which
        returns unknown) must not validate: negatives need explicit unsat."""
        from jacobian.math.graphs.coloring._models import KColorabilityResult

        k4 = self._k4()
        payload = {
            "graph": k4.model_dump(),
            "colors": 3,
            "solver_conflicts": 1,
            "status": "DECIDED",
            "colorable": False,
            "coloring": None,
            "vertex_count": 4,
        }
        with pytest.raises(ValidationError):
            KColorabilityResult.model_validate(payload)

    def test_default_budget_still_decides_k4_negative(self):
        from jacobian.math.graphs.coloring._models import KColorabilityRequest
        from jacobian.math.graphs.coloring._operations import compute_k_colorability

        result = compute_k_colorability(
            KColorabilityRequest(graph=self._k4(), colors=3)
        )
        assert result.status == "DECIDED"
        assert result.colorable is False
        assert result.coloring is None

    def test_decided_negative_runs_the_bounded_solver_exactly_once(self, monkeypatch):
        """The producing solve must not pay a second full-budget replay:
        one declared budget covers all solver work on the request."""
        import jacobian.math.graphs.coloring._models as coloring_models
        from jacobian.math.graphs.coloring._models import KColorabilityRequest
        from jacobian.math.graphs.coloring._operations import compute_k_colorability

        calls: list[int] = []
        original = coloring_models._run_k_colorability_solver

        def counting(graph, colors, solver_conflicts):
            calls.append(solver_conflicts)
            return original(graph, colors, solver_conflicts)

        monkeypatch.setattr(coloring_models, "_run_k_colorability_solver", counting)
        result = compute_k_colorability(
            KColorabilityRequest(graph=self._k4(), colors=3)
        )
        assert result.status == "DECIDED"
        assert result.colorable is False
        assert calls == [result.solver_conflicts]

    def test_forged_positive_witnesses_are_rejected(self):
        """A colorable claim must carry one in-range color per vertex and the
        witness must be proper for the result's own graph."""
        from jacobian.math.graphs.coloring._models import (
            GraphEdgeList,
            KColorabilityResult,
        )

        path = GraphEdgeList(vertex_count=3, edges=((0, 1), (1, 2)))
        base = {
            "graph": path,
            "colors": 2,
            "status": "DECIDED",
            "colorable": True,
            "vertex_count": 3,
        }
        with pytest.raises(ValidationError):
            KColorabilityResult(**{**base, "coloring": (0, 1)})
        with pytest.raises(ValidationError):
            KColorabilityResult(**{**base, "coloring": (0, 1, 2)})
        with pytest.raises(ValidationError):
            KColorabilityResult(**{**base, "coloring": (0, 0, 1)})
        with pytest.raises(ValidationError):
            KColorabilityResult(**{**base, "coloring": (0, 1, 0), "vertex_count": 2})
