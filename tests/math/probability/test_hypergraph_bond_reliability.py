"""Exact hyperedge bond reliability on finite simple hypergraphs (#1688)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.probability import (
    _hypergraph_bond_reliability as hypergraph_reliability_module,
)
from jacobian.math.probability._hypergraph_bond_reliability import (
    HYPERGRAPH_BOND_CONNECTION_PROBABILITY_OPERATION,
    HyperedgeOpenProbability,
    HypergraphBondConnectionProbabilityResult,
    HypergraphBondReliabilitySource,
    compute_hypergraph_bond_connection_probability,
)


def _probability(value: Fraction) -> CanonicalRational:
    return CanonicalRational(num=value.numerator, den=value.denominator)


def _source(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, tuple[str, ...]], ...],
    probabilities: dict[str, Fraction],
    terminals: tuple[str, str],
) -> HypergraphBondReliabilitySource:
    hypergraph = FiniteHypergraph(vertices=vertices, edges=edges)
    return HypergraphBondReliabilitySource(
        hypergraph=hypergraph,
        hyperedge_probabilities=tuple(
            HyperedgeOpenProbability(
                hyperedge_id=edge_id,
                open_probability=_probability(probabilities[edge_id]),
            )
            for edge_id, _ in hypergraph.edges
        ),
        terminals=terminals,
    )


def _brute_force(
    edges: tuple[tuple[str, tuple[str, ...]], ...],
    probabilities: dict[str, Fraction],
    terminals: tuple[str, str],
) -> Fraction:
    total = Fraction(0)
    for mask in range(1 << len(edges)):
        open_members = [
            set(members)
            for index, (_, members) in enumerate(edges)
            if mask & (1 << index)
        ]
        mass = Fraction(1)
        for index, (edge_id, _) in enumerate(edges):
            probability = probabilities[edge_id]
            mass *= probability if mask & (1 << index) else 1 - probability
        reached = {terminals[0]}
        changed = True
        while changed:
            changed = False
            for members in open_members:
                if members & reached and not members <= reached:
                    reached |= members
                    changed = True
        if terminals[1] in reached:
            total += mass
    return total


def test_chain_through_a_shared_vertex_connects() -> None:
    """a-b and b-d connect a to d only when both hyperedges are open."""
    result = compute_hypergraph_bond_connection_probability(
        _source(
            ("a", "b", "d"),
            (("ab", ("a", "b")), ("bd", ("b", "d"))),
            {"ab": Fraction(1, 2), "bd": Fraction(1, 2)},
            ("a", "d"),
        )
    )
    assert result.connection_probability.as_fraction() == Fraction(1, 4)
    assert result.connectivity_convention == "INCIDENCE_GRAPH_CHAIN"


def test_high_height_inputs_are_admitted_by_the_complete_result_carrier() -> None:
    """Five 128-digit factors exceed the old 512-digit late validator cap."""
    denominator = 10**128 - 1
    vertices = tuple("v" + str(index) for index in range(6))
    edges = tuple(
        ("e" + str(index), ("v" + str(index), "v" + str(index + 1)))
        for index in range(5)
    )
    result = compute_hypergraph_bond_connection_probability(
        _source(
            vertices,
            edges,
            {edge_id: Fraction(1, denominator) for edge_id, _ in edges},
            ("v0", "v5"),
        )
    )
    assert result.visited_states == 32
    assert len(str(result.states[-1].state_probability.as_fraction().denominator)) > 512


def test_disjoint_open_hyperedges_do_not_connect() -> None:
    """Overlapping vertex sets are required for a chain, not mere disjointness."""
    result = compute_hypergraph_bond_connection_probability(
        _source(
            ("a", "b", "c", "d"),
            (("ab", ("a", "b")), ("cd", ("c", "d"))),
            {"ab": Fraction(1), "cd": Fraction(1)},
            ("a", "d"),
        )
    )
    assert result.connection_probability.as_fraction() == Fraction(0)


def test_clique_expansion_would_differ_and_is_not_used() -> None:
    """Incidence-chain connectivity is not the clique-expansion convention.

    Hyperedge {a, b, c} alone does not connect a and d under either convention,
    but it also does not imply every pair inside it becomes a separate bond in
    the state ledger; the ledger records open hyperedges, not expanded pairs.
    """
    result = compute_hypergraph_bond_connection_probability(
        _source(
            ("a", "b", "c"),
            (("abc", ("a", "b", "c")),),
            {"abc": Fraction(1, 2)},
            ("a", "b"),
        )
    )
    # One open hyperedge already joins both terminals.
    assert result.connection_probability.as_fraction() == Fraction(1, 2)
    assert result.states[1].open_hyperedge_ids == ("abc",)


def test_three_hyperedges_form_a_chain_in_order() -> None:
    """Connectivity propagates transitively across a chain of three hyperedges."""
    result = compute_hypergraph_bond_connection_probability(
        _source(
            ("a", "b", "c", "d"),
            (("ab", ("a", "b")), ("bc", ("b", "c")), ("cd", ("c", "d"))),
            {"ab": Fraction(1), "bc": Fraction(1), "cd": Fraction(1)},
            ("a", "d"),
        )
    )
    assert result.connection_probability.as_fraction() == Fraction(1)


def test_states_partition_and_successful_mass_matches_the_scalar() -> None:
    """The ledger is a probability partition and its success mass is the total."""
    result = compute_hypergraph_bond_connection_probability(
        _source(
            ("a", "b", "c"),
            (("ab", ("a", "b")), ("bc", ("b", "c"))),
            {"ab": Fraction(1, 3), "bc": Fraction(2, 5)},
            ("a", "c"),
        )
    )
    total = sum(
        (state.state_probability.as_fraction() for state in result.states),
        Fraction(0),
    )
    assert total == Fraction(1)
    successful = sum(
        (
            state.state_probability.as_fraction()
            for state in result.states
            if state.terminals_connected
        ),
        Fraction(0),
    )
    assert successful == result.connection_probability.as_fraction()
    assert result.visited_states == 4


def test_matches_independent_brute_force() -> None:
    """Several small hypergraphs agree with an independent subset enumeration."""
    cases = (
        (("a", "b"), (("ab", ("a", "b")),), {"ab": Fraction(1, 4)}, ("a", "b")),
        (
            ("a", "b", "c", "d"),
            (("ab", ("a", "b")), ("bc", ("b", "c")), ("cd", ("c", "d"))),
            {"ab": Fraction(1, 3), "bc": Fraction(1, 2), "cd": Fraction(2, 3)},
            ("a", "d"),
        ),
        (
            ("a", "b", "c"),
            (("ac", ("a", "c")), ("bc", ("b", "c")), ("abc", ("a", "b", "c"))),
            {"ac": Fraction(1, 2), "bc": Fraction(1, 3), "abc": Fraction(1, 5)},
            ("a", "b"),
        ),
    )
    for vertices, edges, probabilities, terminals in cases:
        result = compute_hypergraph_bond_connection_probability(
            _source(vertices, edges, probabilities, terminals)
        )
        assert result.connection_probability.as_fraction() == _brute_force(
            edges, probabilities, terminals
        )


def test_duplicate_or_unknown_terminals_are_rejected() -> None:
    """Terminals must be two distinct declared vertices of the hypergraph."""
    with pytest.raises(ValidationError):
        _source(("a", "b"), (("ab", ("a", "b")),), {"ab": Fraction(1, 2)}, ("a", "a"))


def test_probability_outside_unit_interval_is_rejected() -> None:
    """Component probabilities must lie in [0, 1]."""
    with pytest.raises(OperationDomainValidationError):
        compute_hypergraph_bond_connection_probability(
            _source(
                ("a", "b"), (("ab", ("a", "b")),), {"ab": Fraction(3, 2)}, ("a", "b")
            )
        )


def test_result_round_trips_through_strict_json() -> None:
    """The complete ledger survives strict JSON serialization unchanged."""
    result = compute_hypergraph_bond_connection_probability(
        _source(
            ("a", "b"),
            (("ab", ("a", "b")),),
            {"ab": Fraction(1, 2)},
            ("a", "b"),
        )
    )
    restored = HypergraphBondConnectionProbabilityResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result


def test_forged_hyperedge_state_ids_are_rejected_after_json_round_trip() -> None:
    result = compute_hypergraph_bond_connection_probability(
        _source(
            ("a", "b"),
            (("ab", ("a", "b")),),
            {"ab": Fraction(1, 2)},
            ("a", "b"),
        )
    )
    payload = result.model_dump(mode="json")
    payload["states"][1]["open_hyperedge_ids"] = []
    with pytest.raises(ValidationError, match="state IDs"):
        HypergraphBondConnectionProbabilityResult.model_validate_json(
            encode_strict_json(payload), strict=True
        )
    payload = result.model_dump(mode="json")
    payload["states"][1]["open_hyperedge_ids"] = ["unknown"]
    with pytest.raises(ValidationError, match="state IDs"):
        HypergraphBondConnectionProbabilityResult.model_validate_json(
            encode_strict_json(payload), strict=True
        )


def test_hyperedge_bound_is_owned_by_operation_admission() -> None:
    vertices = tuple(f"v{index}" for index in range(14))
    edges = tuple((f"e{index}", (f"v{index}", f"v{index + 1}")) for index in range(13))
    source = _source(
        vertices,
        edges,
        {edge_id: Fraction(1, 2) for edge_id, _ in edges},
        ("v0", "v1"),
    )
    with pytest.raises(OperationResourceAdmissionError, match="hyperedge bound"):
        compute_hypergraph_bond_connection_probability(source)


def test_isolated_declared_vertices_are_charged_as_retained_source_only() -> None:
    vertices = tuple(f"v{index}" for index in range(25))
    source = _source(
        vertices,
        (("edge", ("v0", "v1")),),
        {"edge": Fraction(1, 2)},
        ("v0", "v1"),
    )
    result = compute_hypergraph_bond_connection_probability(source)
    assert result.visited_states == 2
    assert result.connection_probability.as_fraction() == Fraction(1, 2)
    assert result.source.hypergraph.vertices == vertices


def test_hypergraph_ledger_allocation_is_admitted_before_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hypergraph_reliability_module,
        "MAX_HYPERGRAPH_RELIABILITY_LEDGER_UNITS",
        1,
    )
    source = _source(
        ("a", "b"),
        (("ab", ("a", "b")),),
        {"ab": Fraction(1, 2)},
        ("a", "b"),
    )
    with pytest.raises(OperationResourceAdmissionError, match="allocation bound"):
        compute_hypergraph_bond_connection_probability(source)


def test_empty_hyperedges_are_rejected_by_the_reliability_contract() -> None:
    with pytest.raises(ValidationError, match="nonempty"):
        _source(
            ("a", "b"),
            (("empty", ()),),
            {"empty": Fraction(1, 2)},
            ("a", "b"),
        )


def test_duplicate_hyperedge_vertex_sets_are_rejected_by_the_reliability_contract() -> (
    None
):
    with pytest.raises(ValidationError, match="distinct vertex sets"):
        _source(
            ("a", "b"),
            (("first", ("a", "b")), ("second", ("a", "b"))),
            {"first": Fraction(1, 2), "second": Fraction(1, 2)},
            ("a", "b"),
        )


def test_published_hypergraph_example_states_axis_and_terminal_preconditions() -> None:
    example = HYPERGRAPH_BOND_CONNECTION_PROBABILITY_OPERATION.examples[0]
    description = example.description
    assert "ab and bd" in description
    assert "hyperedge axis" in description
    assert "distinct declared vertices" in description
