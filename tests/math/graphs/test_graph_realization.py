"""Tests for graph realization operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.realization._models import (
    DegreeSequenceRequest,
    DegreeSequenceResult,
    GraphicalityCheckRequest,
    GraphicalityCheckResult,
    GraphRealizationRequest,
    GraphRealizationResult,
    RealizationCheckRequest,
    RealizationCheckResult,
)
from jacobian.math.graphs.realization._operations import (
    compute_degree_sequence,
    compute_graph_realization,
    compute_graphicality_check,
    compute_realization_check,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_graphical(degrees: list[int]) -> DegreeSequenceResult:
    return compute_degree_sequence(
        DegreeSequenceRequest.model_validate({"sequence": {"degrees": degrees}})
    )


def _realize(degrees: list[int]) -> GraphRealizationResult:
    return compute_graph_realization(
        GraphRealizationRequest.model_validate({"sequence": {"degrees": degrees}})
    )


def _graphicality_check(degrees: list[int]) -> GraphicalityCheckResult:
    return compute_graphicality_check(
        GraphicalityCheckRequest.model_validate({"sequence": {"degrees": degrees}})
    )


def _check(
    degrees: list[int],
    vertex_count: int,
    edges: list[tuple[int, int]],
) -> RealizationCheckResult:
    return compute_realization_check(
        RealizationCheckRequest.model_validate(
            {
                "sequence": {"degrees": degrees},
                "graph": {"vertex_count": vertex_count, "edges": edges},
            }
        )
    )


# ---------------------------------------------------------------------------
# is_graphical (Erdos-Gallai)
# ---------------------------------------------------------------------------


class TestIsGraphical:
    def test_empty_sequence_is_graphical(self) -> None:
        """A single isolated vertex with degree 0 is graphical."""
        result = _is_graphical([0])
        assert result.is_graphical is True
        assert result.vertex_count == 1
        assert result.degree_sum == 0

    def test_simple_path_is_graphical(self) -> None:
        result = _is_graphical([1, 2, 2, 1])
        assert result.is_graphical is True

    def test_cycle_is_graphical(self) -> None:
        result = _is_graphical([2, 2, 2, 2])
        assert result.is_graphical is True

    def test_complete_graph_is_graphical(self) -> None:
        result = _is_graphical([3, 3, 3, 3])
        assert result.is_graphical is True

    def test_odd_sum_not_graphical(self) -> None:
        result = _is_graphical([3, 3, 3])
        assert result.is_graphical is False

    def test_impossible_degree_too_large(self) -> None:
        """A vertex cannot have degree >= n."""
        result = _is_graphical([6, 1, 1, 1, 1])
        assert result.is_graphical is False

    def test_classic_non_graphical_example(self) -> None:
        """[3, 3, 3, 1] has even sum (10) but violates Erdos-Gallai at k=1."""
        result = _is_graphical([3, 3, 3, 1])
        assert result.is_graphical is False

    def test_star_is_graphical(self) -> None:
        result = _is_graphical([3, 1, 1, 1])
        assert result.is_graphical is True

    def test_two_vertices_one_edge(self) -> None:
        result = _is_graphical([1, 1])
        assert result.is_graphical is True

    def test_two_vertices_disconnected(self) -> None:
        result = _is_graphical([0, 0])
        assert result.is_graphical is True

    def test_degree_sum_reported(self) -> None:
        result = _is_graphical([2, 2, 2, 2])
        assert result.degree_sum == 8

    def test_contract_rejects_negative_degree(self) -> None:
        with pytest.raises(ValidationError):
            DegreeSequenceRequest.model_validate(
                {"sequence": {"degrees": [-1, 2, 2, 1]}}
            )

    def test_contract_rejects_empty_sequence(self) -> None:
        with pytest.raises(ValidationError):
            DegreeSequenceRequest.model_validate({"sequence": {"degrees": []}})

    def test_contract_rejects_degree_exceeding_bound(self) -> None:
        with pytest.raises(ValidationError):
            DegreeSequenceRequest.model_validate({"sequence": {"degrees": [64, 1, 1]}})


# ---------------------------------------------------------------------------
# realization (Havel-Hakimi construction)
# ---------------------------------------------------------------------------


class TestGraphRealization:
    def test_realizes_simple_path(self) -> None:
        result = _realize([1, 2, 2, 1])
        assert result.is_graphical is True
        assert result.vertex_count == 4
        # A path on 4 vertices has 3 edges
        assert len(result.edges) == 3

    def test_realized_edges_match_degree_sequence(self) -> None:
        """The realized graph's degree sequence must match the input."""
        degrees = [2, 2, 2, 2]
        result = _realize(degrees)
        assert result.is_graphical is True
        actual = [0] * 4
        for source, target in result.edges:
            actual[source] += 1
            actual[target] += 1
        assert actual == degrees

    def test_realizes_complete_graph(self) -> None:
        result = _realize([3, 3, 3, 3])
        assert result.is_graphical is True
        # K_4 has 6 edges
        assert len(result.edges) == 6

    def test_realizes_star(self) -> None:
        result = _realize([3, 1, 1, 1])
        assert result.is_graphical is True
        assert len(result.edges) == 3

    def test_realizes_isolated_vertices(self) -> None:
        result = _realize([0, 0, 0])
        assert result.is_graphical is True
        assert result.edges == ()

    def test_non_graphical_returns_empty_edges(self) -> None:
        result = _realize([3, 3, 3])
        assert result.is_graphical is False
        assert result.edges == ()
        assert result.vertex_count == 3

    def test_realized_graph_is_simple(self) -> None:
        """No self-loops or duplicate edges."""
        result = _realize([2, 2, 2, 2, 2])
        assert result.is_graphical is True
        seen: set[tuple[int, int]] = set()
        for source, target in result.edges:
            assert source != target, "self-loop detected"
            edge = (min(source, target), max(source, target))
            assert edge not in seen, "duplicate edge detected"
            seen.add(edge)


# ---------------------------------------------------------------------------
# graphicality check (with certificate)
# ---------------------------------------------------------------------------


class TestGraphicalityCheck:
    def test_graphical_returns_erdos_gallai_certificate(self) -> None:
        result = _graphicality_check([1, 2, 2, 1])
        assert result.is_graphical is True
        assert result.certificate == "ERDOS-GALLAI"

    def test_odd_sum_returns_certificate(self) -> None:
        result = _graphicality_check([3, 3, 3])
        assert result.is_graphical is False
        assert "odd-sum" in result.certificate

    def test_degree_too_large_returns_certificate(self) -> None:
        result = _graphicality_check([6, 1, 1, 1, 1])
        assert result.is_graphical is False
        assert "exceeds" in result.certificate

    def test_erdos_gallai_violation_returns_certificate(self) -> None:
        """[3, 3, 3, 1] has even sum but violates an inequality."""
        result = _graphicality_check([3, 3, 3, 1])
        assert result.is_graphical is False
        assert "erdos-gallai violation" in result.certificate

    def test_certificate_agrees_with_is_graphical(self) -> None:
        """The check must agree with the standalone is_graphical operation."""
        for degrees in (
            [1, 2, 2, 1],
            [3, 3, 3],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [3, 1, 1, 1],
            [3, 3, 3, 1],
            [0, 0],
        ):
            assert _is_graphical(degrees).is_graphical == (
                _graphicality_check(degrees).is_graphical
            ), f"mismatch for {degrees}"

    def test_degree_sum_reported(self) -> None:
        result = _graphicality_check([2, 2, 2, 2])
        assert result.degree_sum == 8


# ---------------------------------------------------------------------------
# realization check
# ---------------------------------------------------------------------------


class TestRealizationCheck:
    def test_valid_realization(self) -> None:
        result = _check([1, 2, 2, 1], 4, [(0, 1), (1, 2), (2, 3)])
        assert result.is_realization is True
        assert result.expected_degrees == (1, 2, 2, 1)
        assert result.actual_degrees == (1, 2, 2, 1)

    def test_invalid_realization(self) -> None:
        result = _check([2, 2, 2, 2], 4, [(0, 1), (1, 2), (2, 3)])
        assert result.is_realization is False

    def test_empty_graph_matches_zero_degrees(self) -> None:
        result = _check([0, 0, 0], 3, [])
        assert result.is_realization is True
        assert result.actual_degrees == (0, 0, 0)

    def test_complete_graph_matches(self) -> None:
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        result = _check([3, 3, 3, 3], 4, edges)
        assert result.is_realization is True

    def test_actual_degrees_computed_correctly(self) -> None:
        """The actual degrees must reflect the input edges."""
        edges = [(0, 1), (0, 2), (1, 2)]
        result = _check([2, 2, 2], 3, edges)
        assert result.actual_degrees == (2, 2, 2)

    def test_contract_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValidationError):
            RealizationCheckRequest.model_validate(
                {
                    "sequence": {"degrees": [1, 2, 2, 1]},
                    "graph": {"vertex_count": 3, "edges": [(0, 1), (1, 2)]},
                }
            )

    def test_contract_rejects_self_loop(self) -> None:
        with pytest.raises(ValidationError):
            RealizationCheckRequest.model_validate(
                {
                    "sequence": {"degrees": [0, 0]},
                    "graph": {"vertex_count": 2, "edges": [(0, 0)]},
                }
            )

    def test_contract_rejects_duplicate_edges(self) -> None:
        with pytest.raises(ValidationError):
            RealizationCheckRequest.model_validate(
                {
                    "sequence": {"degrees": [1, 1]},
                    "graph": {"vertex_count": 2, "edges": [(0, 1), (1, 0)]},
                }
            )

    def test_contract_rejects_out_of_range_vertex(self) -> None:
        with pytest.raises(ValidationError):
            RealizationCheckRequest.model_validate(
                {
                    "sequence": {"degrees": [0, 0]},
                    "graph": {"vertex_count": 2, "edges": [(0, 5)]},
                }
            )


# ---------------------------------------------------------------------------
# Cross-consistency: construct + check
# ---------------------------------------------------------------------------


class TestCrossConsistency:
    def test_constructed_graph_passes_check(self) -> None:
        """A graph constructed by Havel-Hakimi must pass the realization check."""
        degrees = [2, 2, 2, 2, 2]
        realized = _realize(degrees)
        assert realized.is_graphical is True
        check_result = _check(degrees, len(degrees), list(realized.edges))
        assert check_result.is_realization is True

    def test_constructed_complete_graph_passes_check(self) -> None:
        n = 5
        degrees = [n - 1] * n
        realized = _realize(degrees)
        assert realized.is_graphical is True
        check_result = _check(degrees, n, list(realized.edges))
        assert check_result.is_realization is True

    def test_all_operations_agree_on_graphicality(self) -> None:
        """is_graphical, construct, and check must agree on graphicality."""
        test_cases = [
            [1, 2, 2, 1],
            [3, 3, 3],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [3, 1, 1, 1],
            [4, 4, 4, 4, 4],
        ]
        for degrees in test_cases:
            is_graphical = _is_graphical(degrees).is_graphical
            realized = _realize(degrees)
            assert realized.is_graphical == is_graphical, (
                f"construct disagrees with is_graphical for {degrees}"
            )
