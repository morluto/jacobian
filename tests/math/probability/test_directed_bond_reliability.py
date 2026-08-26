"""Exact directed bond-reliability operation contracts."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from importlib import import_module
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.directed._models import (
    MAX_DIRECTED_GRAPH_PARSE_EDGES,
    DirectedGraph,
)
from jacobian.math.probability._directed_bond_reliability import (
    MAX_DIRECTED_BOND_RELIABILITY_ARCS,
    MAX_DIRECTED_BOND_RELIABILITY_DECLARED_VERTICES,
    MAX_DIRECTED_BOND_RELIABILITY_LOGICAL_WORK,
    MAX_DIRECTED_BOND_RELIABILITY_RELEVANT_VERTICES,
    MAX_DIRECTED_BOND_RELIABILITY_STATES,
    DirectedBondConnectionProbabilityRequest,
    DirectedBondConnectionProbabilityResult,
    DirectedBondConnectionProbabilitySource,
    DirectedBondReliabilityArcProbability,
    _directed_bond_connection_probability,
    verify_directed_bond_connection_probability_result,
)


def _rational(value: Fraction) -> dict[str, str]:
    return {"num": str(value.numerator), "den": str(value.denominator)}


def _request(
    *,
    vertex_count: int,
    arcs: tuple[tuple[int, int], ...],
    probabilities: tuple[Fraction, ...],
    source: int,
    target: int,
) -> DirectedBondConnectionProbabilityRequest:
    return DirectedBondConnectionProbabilityRequest(
        graph=DirectedGraph(vertex_count=vertex_count, edges=arcs),
        arc_probabilities=tuple(
            DirectedBondReliabilityArcProbability(
                arc=arc,
                open_probability=CanonicalRational.from_fraction(probability),
            )
            for arc, probability in zip(arcs, probabilities, strict=True)
        ),
        source=source,
        target=target,
    )


def _probability(result: DirectedBondConnectionProbabilityResult) -> Fraction:
    return result.connection_probability.as_fraction()


def test_single_directed_arc_has_its_open_probability() -> None:
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=2,
            arcs=((0, 1),),
            probabilities=(Fraction(2, 7),),
            source=0,
            target=1,
        )
    )

    assert _probability(result) == Fraction(2, 7)
    assert result.visited_states == 2
    assert tuple(state.open_arcs for state in result.states) == ((), ((0, 1),))
    assert tuple(state.source_reaches_target for state in result.states) == (
        False,
        True,
    )


def test_reverse_arc_does_not_create_forward_directed_connection() -> None:
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=2,
            arcs=((1, 0),),
            probabilities=(Fraction(1, 1),),
            source=0,
            target=1,
        )
    )

    assert _probability(result) == 0
    assert all(not state.source_reaches_target for state in result.states)


def test_two_arc_directed_series_multiplies_exact_probabilities() -> None:
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=3,
            arcs=((0, 1), (1, 2)),
            probabilities=(Fraction(2, 3), Fraction(3, 5)),
            source=0,
            target=2,
        )
    )

    assert _probability(result) == Fraction(2, 5)


def test_independent_directed_routes_use_exact_union_mass() -> None:
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=4,
            arcs=((0, 1), (1, 3), (0, 2), (2, 3)),
            probabilities=(
                Fraction(1, 2),
                Fraction(2, 3),
                Fraction(3, 4),
                Fraction(4, 5),
            ),
            source=0,
            target=3,
        )
    )

    assert _probability(result) == Fraction(11, 15)
    assert (
        sum(
            (state.state_probability.as_fraction() for state in result.states),
            Fraction(),
        )
        == 1
    )


def test_reversing_terminals_changes_the_directed_event() -> None:
    request = _request(
        vertex_count=2,
        arcs=((0, 1), (1, 0)),
        probabilities=(Fraction(2, 3), Fraction(3, 5)),
        source=0,
        target=1,
    )
    reverse = _request(
        vertex_count=2,
        arcs=((0, 1), (1, 0)),
        probabilities=(Fraction(2, 3), Fraction(3, 5)),
        source=1,
        target=0,
    )

    assert _probability(_directed_bond_connection_probability(request)) == Fraction(
        2, 3
    )
    assert _probability(_directed_bond_connection_probability(reverse)) == Fraction(
        3, 5
    )


def test_component_order_is_canonical_and_does_not_change_result() -> None:
    canonical = _request(
        vertex_count=3,
        arcs=((0, 1), (1, 2)),
        probabilities=(Fraction(2, 3), Fraction(3, 5)),
        source=0,
        target=2,
    )
    permuted = _request(
        vertex_count=3,
        arcs=((1, 2), (0, 1)),
        probabilities=(Fraction(3, 5), Fraction(2, 3)),
        source=0,
        target=2,
    )

    expected = _directed_bond_connection_probability(canonical)
    actual = _directed_bond_connection_probability(permuted)
    assert actual == expected
    assert actual.source.graph.edges == ((0, 1), (1, 2))


def test_sparse_graph_above_sixteen_vertices_is_admitted() -> None:
    """The derived work budget, not a fixed vertex ceiling, bounds vertices."""
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=17,
            arcs=((3, 16),),
            probabilities=(Fraction(2, 7),),
            source=3,
            target=16,
        )
    )

    assert _probability(result) == Fraction(2, 7)
    assert result.visited_states == 2


def test_edgeless_graph_above_the_shared_vertex_cap_is_admitted() -> None:
    """A 257-vertex edgeless source stays inside the derived work budget."""
    result = _directed_bond_connection_probability(
        _request(vertex_count=257, arcs=(), probabilities=(), source=0, target=1)
    )

    assert _probability(result) == 0
    assert result.arc_count == 0
    assert result.visited_states == 1


def test_single_arc_above_the_shared_vertex_cap_is_admitted() -> None:
    """Sparse sources above 64 vertices are not rejected by the shared graph."""
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=100,
            arcs=((63, 64),),
            probabilities=(Fraction(2, 7),),
            source=63,
            target=64,
        )
    )

    assert _probability(result) == Fraction(2, 7)
    assert result.visited_states == 2


def test_edgeless_graph_has_exact_zero_connection_probability() -> None:
    """Zero arcs keep the complete single-state ledger with probability zero."""
    result = _directed_bond_connection_probability(
        _request(vertex_count=2, arcs=(), probabilities=(), source=0, target=1)
    )

    assert _probability(result) == 0
    assert result.arc_count == 0
    assert result.visited_states == 1
    assert tuple(state.open_arcs for state in result.states) == ((),)
    assert tuple(state.source_reaches_target for state in result.states) == (False,)
    assert tuple(state.state_probability.as_fraction() for state in result.states) == (
        Fraction(1),
    )


def test_ledger_estimate_bounds_subset_numerator_growth_not_per_state_maxima() -> None:
    """A tall-numerator chain is charged by subset occurrence, not per state."""
    denominator = int("9" * 90)
    probability = Fraction(denominator - 1, denominator)
    arcs = tuple((index, index + 1) for index in range(12))
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=13,
            arcs=arcs,
            probabilities=(probability,) * len(arcs),
            source=0,
            target=12,
        )
    )

    assert _probability(result) == probability**12
    assert result.visited_states == 4096


def _mutate_open_arcs(payload: dict[str, Any]) -> None:
    payload["states"][1]["open_arcs"] = []


def _mutate_reachability(payload: dict[str, Any]) -> None:
    payload["states"][3]["source_reaches_target"] = False


def _mutate_connection_probability(payload: dict[str, Any]) -> None:
    payload["connection_probability"] = _rational(Fraction())


@pytest.mark.parametrize(
    "mutation",
    (
        _mutate_open_arcs,
        _mutate_reachability,
        _mutate_connection_probability,
    ),
    ids=("open-arcs", "reachability", "aggregate-probability"),
)
def test_explicit_result_verifier_rejects_mutated_source_bound_conclusions(
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=3,
            arcs=((0, 1), (1, 2)),
            probabilities=(Fraction(1, 2), Fraction(1, 2)),
            source=0,
            target=2,
        )
    )
    payload = result.model_dump(mode="json")
    mutation(payload)

    claim = DirectedBondConnectionProbabilityResult.model_validate(payload)

    assert not verify_directed_bond_connection_probability_result(claim)


def test_probability_bound_rejects_thirteenth_arc_before_enumeration() -> None:
    arcs = tuple((index, index + 1) for index in range(13))
    with pytest.raises(ValidationError):
        _request(
            vertex_count=14,
            arcs=arcs,
            probabilities=(Fraction(1, 2),) * len(arcs),
            source=0,
            target=13,
        )


def test_logical_work_bound_covers_one_producer_pass() -> None:
    """The 12-arc envelope charges its one subset-enumeration pass."""
    per_state_kernel_work = (
        6 * MAX_DIRECTED_BOND_RELIABILITY_ARCS
        + 4 * MAX_DIRECTED_BOND_RELIABILITY_RELEVANT_VERTICES
    )
    per_state_record_work = MAX_DIRECTED_BOND_RELIABILITY_ARCS + 3

    assert (
        MAX_DIRECTED_BOND_RELIABILITY_STATES
        * (per_state_kernel_work + per_state_record_work)
        + 8
    ) == MAX_DIRECTED_BOND_RELIABILITY_LOGICAL_WORK


def test_successful_compute_enumerates_each_arc_subset_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trusted result construction does not replay the complete state ledger."""

    reliability_module = import_module(
        "jacobian.math.probability._directed_bond_reliability"
    )

    request = _request(
        vertex_count=4,
        arcs=((0, 1), (0, 2), (1, 3), (2, 3)),
        probabilities=(Fraction(1, 2),) * 4,
        source=0,
        target=3,
    )
    call_count = 0
    original = reliability_module._directed_path_exists

    def counted(**kwargs: object) -> bool:
        nonlocal call_count
        call_count += 1
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(reliability_module, "_directed_path_exists", counted)
    _directed_bond_connection_probability(request)

    assert call_count == 1 << len(request.graph.edges)


def test_ledger_bound_rejects_large_exact_probability_products_before_enumeration() -> (
    None
):
    arcs = tuple((index, index + 1) for index in range(12))
    large_denominator = int("9" * 128)
    large_probability = Fraction(large_denominator - 1, large_denominator)
    with pytest.raises(ValidationError):
        _request(
            vertex_count=13,
            arcs=arcs,
            probabilities=(large_probability,) * len(arcs),
            source=0,
            target=12,
        )


def test_zero_and_one_probabilities_keep_complete_state_convention() -> None:
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=3,
            arcs=((0, 1), (1, 2)),
            probabilities=(Fraction(1), Fraction()),
            source=0,
            target=2,
        )
    )

    assert _probability(result) == 0
    assert result.visited_states == 4
    assert tuple(state.state_index for state in result.states) == (0, 1, 2, 3)


def test_edgeless_million_vertex_request_matches_small_equivalent() -> None:
    """Sparse admission prices executed work, not the declared index space."""

    large = _directed_bond_connection_probability(
        _request(
            vertex_count=1_000_000,
            arcs=(),
            probabilities=(),
            source=0,
            target=999_999,
        )
    )
    small = _directed_bond_connection_probability(
        _request(vertex_count=2, arcs=(), probabilities=(), source=0, target=1)
    )

    assert _probability(large) == _probability(small) == 0
    assert [
        (
            state.state_index,
            state.open_arcs,
            state.source_reaches_target,
            state.state_probability.as_fraction(),
        )
        for state in large.states
    ] == [
        (
            state.state_index,
            state.open_arcs,
            state.source_reaches_target,
            state.state_probability.as_fraction(),
        )
        for state in small.states
    ]
    replay = DirectedBondConnectionProbabilityResult.model_validate(
        large.model_dump(mode="json")
    )
    assert replay == large


def test_edgeless_source_at_the_declared_vertex_bound_stays_exact() -> None:
    """Sparse admission materializes terminals, not every declared vertex."""

    result = _directed_bond_connection_probability(
        _request(
            vertex_count=MAX_DIRECTED_BOND_RELIABILITY_DECLARED_VERTICES,
            arcs=(),
            probabilities=(),
            source=0,
            target=MAX_DIRECTED_BOND_RELIABILITY_DECLARED_VERTICES - 1,
        )
    )

    assert _probability(result) == 0
    assert result.arc_count == 0
    assert result.visited_states == 1
    assert tuple(state.source_reaches_target for state in result.states) == (False,)
    replay = DirectedBondConnectionProbabilityResult.model_validate(
        result.model_dump(mode="json")
    )
    assert replay == result


def test_dense_source_at_the_relevant_vertex_bound_is_admitted() -> None:
    """Twelve disjoint arcs plus outside terminals fill the relevant set."""

    arcs = tuple((2 * index, 2 * index + 1) for index in range(12))
    vertex_count = MAX_DIRECTED_BOND_RELIABILITY_RELEVANT_VERTICES
    result = _directed_bond_connection_probability(
        _request(
            vertex_count=vertex_count,
            arcs=arcs,
            probabilities=(Fraction(1, 2),) * len(arcs),
            source=vertex_count - 2,
            target=vertex_count - 1,
        )
    )

    assert result.arc_count == MAX_DIRECTED_BOND_RELIABILITY_ARCS
    assert result.visited_states == MAX_DIRECTED_BOND_RELIABILITY_STATES
    assert all(not state.source_reaches_target for state in result.states)
    assert _probability(result) == 0
    replay = DirectedBondConnectionProbabilityResult.model_validate(
        result.model_dump(mode="json")
    )
    assert replay == result


def test_declared_vertex_admission_rejects_one_past_the_label_bound() -> None:
    """The scalar transport ceiling, not traversal work, caps vertex labels."""

    with pytest.raises(ValidationError):
        _request(
            vertex_count=MAX_DIRECTED_BOND_RELIABILITY_DECLARED_VERTICES + 1,
            arcs=(),
            probabilities=(),
            source=0,
            target=1,
        )


class TestPublishedBondReliabilityEnvelope:
    """Published schemas advertise the bond-reliability graph envelope."""

    def test_request_and_source_schemas_project_the_arc_and_vertex_bounds(
        self,
    ) -> None:
        for model_type in (
            DirectedBondConnectionProbabilityRequest,
            DirectedBondConnectionProbabilitySource,
        ):
            graph_schema = model_type.model_json_schema()["properties"]["graph"]
            assert (
                graph_schema["properties"]["edges"]["maxItems"]
                == MAX_DIRECTED_BOND_RELIABILITY_ARCS
            )
            assert (
                graph_schema["properties"]["vertex_count"]["maximum"]
                == MAX_DIRECTED_BOND_RELIABILITY_DECLARED_VERTICES
            )
            assert "relevant vertices" in graph_schema["description"]
            assert "work budget" not in graph_schema["description"]

    def test_shared_carrier_schema_stays_free_of_the_reliability_caps(self) -> None:
        carrier_properties = DirectedGraph.model_json_schema()["properties"]
        assert "maxItems" not in carrier_properties["edges"]
        assert "maximum" not in carrier_properties["vertex_count"]

    def test_carrier_parse_envelope_cannot_bind_reliability_admission(self) -> None:
        """The shared parse guard sits orders of magnitude above 12 arcs."""

        assert (
            MAX_DIRECTED_GRAPH_PARSE_EDGES > 1000 * MAX_DIRECTED_BOND_RELIABILITY_ARCS
        )

    def test_one_arc_over_the_admission_bound_has_a_structured_error(self) -> None:
        schema_max_items = DirectedBondConnectionProbabilityRequest.model_json_schema()[
            "properties"
        ]["graph"]["properties"]["edges"]["maxItems"]
        assert schema_max_items == MAX_DIRECTED_BOND_RELIABILITY_ARCS
        arcs = tuple(
            (index, index + 1)
            for index in range(MAX_DIRECTED_BOND_RELIABILITY_ARCS + 1)
        )
        with pytest.raises(ValidationError) as raised:
            _request(
                vertex_count=len(arcs) + 1,
                arcs=arcs,
                probabilities=(Fraction(1, 2),) * len(arcs),
                source=0,
                target=len(arcs),
            )
        (error,) = raised.value.errors()
        assert error["type"] == "too_long"
        assert error["loc"] == ("arc_probabilities",)
        assert error["ctx"] == {
            "field_type": "Tuple",
            "max_length": MAX_DIRECTED_BOND_RELIABILITY_ARCS,
            "actual_length": MAX_DIRECTED_BOND_RELIABILITY_ARCS + 1,
        }
