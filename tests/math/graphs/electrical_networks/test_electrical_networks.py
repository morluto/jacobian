from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.electrical_networks._models import (
    MAX_LAPLACIAN_VERTICES,
    ConductanceEdge,
    ConductanceNetwork,
    EffectiveResistanceRequest,
    LaplacianNetwork,
    LaplacianRequest,
    NodePotentialRequest,
)
from jacobian.math.graphs.electrical_networks._tools import (
    compute_effective_resistance,
    compute_laplacian,
    compute_node_potentials,
)

C = CanonicalRational


def _edge(source: int, target: int, num: str, den: str) -> ConductanceEdge:
    return ConductanceEdge(
        source=source, target=target, conductance=C(num=num, den=den)
    )


def _net(vertex_count: int, *edges: ConductanceEdge) -> ConductanceNetwork:
    return ConductanceNetwork(vertex_count=vertex_count, edges=edges)


def _laplacian_net(vertex_count: int, *edges: ConductanceEdge) -> LaplacianNetwork:
    return LaplacianNetwork(vertex_count=vertex_count, edges=edges)


def _star_of_distinct_fifty_digit_dens(leaf_count: int = 215) -> ConductanceNetwork:
    """Connected star whose hub diagonal sums ``leaf_count`` distinct 50-digit dens."""

    return _net(
        leaf_count + 1,
        *(
            _edge(0, leaf + 1, "1", str(10**49 + 2 * leaf + 1))
            for leaf in range(leaf_count)
        ),
    )


# ------------------------------------------------------------------ effective resistance


def test_single_edge_resistance_is_one() -> None:
    net = _net(2, _edge(0, 1, "1", "1"))
    req = EffectiveResistanceRequest(network=net, terminal_a=0, terminal_b=1)
    result = compute_effective_resistance(req)
    assert result.effective_resistance.as_fraction() == Fraction(1)
    assert result.terminal_a == 0
    assert result.terminal_b == 1


def test_triangle_unit_resistances_gives_two_thirds() -> None:
    net = _net(3, _edge(0, 1, "1", "1"), _edge(1, 2, "1", "1"), _edge(0, 2, "1", "1"))
    req = EffectiveResistanceRequest(network=net, terminal_a=0, terminal_b=1)
    assert compute_effective_resistance(
        req
    ).effective_resistance.as_fraction() == Fraction(2, 3)


def test_path_graph_three_vertices_gives_two() -> None:
    net = _net(3, _edge(0, 1, "1", "1"), _edge(1, 2, "1", "1"))
    req = EffectiveResistanceRequest(network=net, terminal_a=0, terminal_b=2)
    assert compute_effective_resistance(
        req
    ).effective_resistance.as_fraction() == Fraction(2)


def test_high_conductance_gives_low_resistance() -> None:
    """Single edge with conductance 3 -> resistance 1/3."""
    net = _net(2, _edge(0, 1, "3", "1"))
    req = EffectiveResistanceRequest(network=net, terminal_a=0, terminal_b=1)
    assert compute_effective_resistance(
        req
    ).effective_resistance.as_fraction() == Fraction(1, 3)


def test_four_cycle_square_unit_resistances_gives_expected_values() -> None:
    """C4 with unit resistances: R(adjacent) = 3/4, R(opposite) = 1."""
    net = _net(
        4,
        _edge(0, 1, "1", "1"),
        _edge(1, 2, "1", "1"),
        _edge(2, 3, "1", "1"),
        _edge(0, 3, "1", "1"),
    )
    req_adj = EffectiveResistanceRequest(network=net, terminal_a=0, terminal_b=1)
    assert compute_effective_resistance(
        req_adj
    ).effective_resistance.as_fraction() == Fraction(3, 4)
    req_opp = EffectiveResistanceRequest(network=net, terminal_a=0, terminal_b=2)
    assert compute_effective_resistance(
        req_opp
    ).effective_resistance.as_fraction() == Fraction(1)


def test_rational_conductances_give_exact_rational_resistance() -> None:
    """Single edge with conductance 2/3 -> resistance 3/2."""
    net = _net(2, _edge(0, 1, "2", "3"))
    req = EffectiveResistanceRequest(network=net, terminal_a=0, terminal_b=1)
    assert compute_effective_resistance(
        req
    ).effective_resistance.as_fraction() == Fraction(3, 2)


# ------------------------------------------------------------------ node potentials


def test_node_potentials_path_graph() -> None:
    net = _net(3, _edge(0, 1, "1", "1"), _edge(1, 2, "1", "1"))
    req = NodePotentialRequest(network=net, source=0, sink=2)
    result = compute_node_potentials(req)
    assert len(result.potentials) == 3
    assert result.potentials[0].potential.as_fraction() == Fraction(2)
    assert result.potentials[1].potential.as_fraction() == Fraction(1)
    assert result.potentials[2].potential.as_fraction() == Fraction(0)


def test_node_potentials_sink_is_gauge_zero() -> None:
    net = _net(2, _edge(0, 1, "1", "1"))
    req = NodePotentialRequest(network=net, source=0, sink=1)
    result = compute_node_potentials(req)
    assert result.potentials[1].potential.as_fraction() == Fraction(0)


def test_node_potentials_satisfy_kirchhoff_current() -> None:
    """For a path 0-1 with unit conductance, injecting 1A at 0, extracting at 1:
    V0 - V1 = 1 (resistance), V1 = 0 (gauge), so V0 = 1."""
    net = _net(2, _edge(0, 1, "1", "1"))
    req = NodePotentialRequest(network=net, source=0, sink=1)
    result = compute_node_potentials(req)
    assert result.potentials[0].potential.as_fraction() == Fraction(1)
    assert result.potentials[1].potential.as_fraction() == Fraction(0)


def test_flint_solves_a_path_above_the_previous_vertex_ceiling() -> None:
    vertex_count = 200
    net = _net(
        vertex_count,
        *(_edge(node, node + 1, "1", "1") for node in range(vertex_count - 1)),
    )

    resistance = compute_effective_resistance(
        EffectiveResistanceRequest(
            network=net, terminal_a=0, terminal_b=vertex_count - 1
        )
    )
    potentials = compute_node_potentials(
        NodePotentialRequest(network=net, source=0, sink=vertex_count - 1)
    )

    assert resistance.effective_resistance.as_fraction() == vertex_count - 1
    assert tuple(
        value.potential.as_fraction() for value in potentials.potentials
    ) == tuple(Fraction(vertex_count - 1 - node) for node in range(vertex_count))


def test_flint_solves_the_wide_carrier_unit_path() -> None:
    vertex_count = 256
    net = _net(
        vertex_count,
        *(_edge(node, node + 1, "1", "1") for node in range(vertex_count - 1)),
    )

    resistance = compute_effective_resistance(
        EffectiveResistanceRequest(
            network=net, terminal_a=0, terminal_b=vertex_count - 1
        )
    )
    potentials = compute_node_potentials(
        NodePotentialRequest(network=net, source=0, sink=vertex_count - 1)
    )

    assert resistance.effective_resistance.as_fraction() == vertex_count - 1
    assert potentials.potentials[-1].potential.as_fraction() == Fraction(0)
    assert potentials.potentials[0].potential.as_fraction() == vertex_count - 1


def test_solve_work_rejects_accumulated_star_height_on_both_ops() -> None:
    """Hub incidence of 215 distinct 50-digit dens exceeds the solve-work bound."""

    net = _star_of_distinct_fifty_digit_dens()
    with pytest.raises(
        OperationDomainValidationError, match="solve-work bound"
    ) as resistance:
        compute_effective_resistance(
            EffectiveResistanceRequest(network=net, terminal_a=0, terminal_b=1)
        )
    with pytest.raises(
        OperationDomainValidationError, match="solve-work bound"
    ) as potentials:
        compute_node_potentials(NodePotentialRequest(network=net, source=0, sink=1))
    assert resistance.value.errors()[0]["type"] == "electrical_network.solve_work_bound"
    assert potentials.value.errors()[0]["type"] == "electrical_network.solve_work_bound"


def test_laplacian_keeps_its_separate_materialized_matrix_ceiling() -> None:
    with pytest.raises(ValidationError):
        LaplacianNetwork(
            vertex_count=MAX_LAPLACIAN_VERTICES + 1,
            edges=(_edge(0, 1, "1", "1"),),
        )


def test_laplacian_request_schema_exposes_vertex_ceiling() -> None:
    laplacian_schema = LaplacianRequest.model_json_schema()
    network = laplacian_schema["$defs"]["LaplacianNetwork"]
    assert network["properties"]["vertex_count"]["maximum"] == MAX_LAPLACIAN_VERTICES
    solve_schema = ConductanceNetwork.model_json_schema()
    assert solve_schema["properties"]["vertex_count"]["maximum"] == 256


# ------------------------------------------------------------------ Laplacian


def test_laplacian_single_edge() -> None:
    net = _laplacian_net(2, _edge(0, 1, "1", "1"))
    req = LaplacianRequest(network=net)
    result = compute_laplacian(req)
    assert result.vertex_count == 2
    matrix: dict[tuple[int, int], Fraction] = {}
    for entry in result.entries:
        matrix[(entry.row, entry.col)] = entry.value.as_fraction()
    assert matrix[(0, 0)] == Fraction(1)
    assert matrix[(1, 1)] == Fraction(1)
    assert matrix[(0, 1)] == Fraction(-1)
    assert matrix[(1, 0)] == Fraction(-1)


def test_laplacian_triangle_diagonal_sums_conductances() -> None:
    net = _laplacian_net(
        3,
        _edge(0, 1, "1", "1"),
        _edge(1, 2, "1", "1"),
        _edge(0, 2, "2", "1"),
    )
    req = LaplacianRequest(network=net)
    result = compute_laplacian(req)
    matrix = {(e.row, e.col): e.value.as_fraction() for e in result.entries}
    assert matrix[(0, 0)] == Fraction(3)  # 1 + 2
    assert matrix[(1, 1)] == Fraction(2)  # 1 + 1
    assert matrix[(2, 2)] == Fraction(3)  # 1 + 2
    assert matrix[(0, 1)] == Fraction(-1)
    assert matrix[(1, 0)] == Fraction(-1)
    assert matrix[(0, 2)] == Fraction(-2)
    assert matrix[(2, 0)] == Fraction(-2)
    assert matrix[(1, 2)] == Fraction(-1)
    assert matrix[(2, 1)] == Fraction(-1)


def test_laplacian_rows_sum_to_zero() -> None:
    net = _laplacian_net(
        4,
        _edge(0, 1, "1", "1"),
        _edge(1, 2, "3", "2"),
        _edge(2, 3, "5", "3"),
        _edge(0, 3, "7", "4"),
    )
    req = LaplacianRequest(network=net)
    result = compute_laplacian(req)
    matrix = {(e.row, e.col): e.value.as_fraction() for e in result.entries}
    for row in range(4):
        assert sum(matrix[(row, col)] for col in range(4)) == Fraction(0)


def test_laplacian_accepts_disconnected_network() -> None:
    """The Laplacian is well-defined without connectivity."""
    net = _laplacian_net(4, _edge(0, 1, "1", "1"), _edge(2, 3, "1", "1"))
    result = compute_laplacian(LaplacianRequest(network=net))
    assert result.vertex_count == 4
    assert len(result.entries) == 16


# ------------------------------------------------------------------ contract validation


def test_contract_rejects_nonpositive_conductance() -> None:
    edge = ConductanceEdge(source=0, target=1, conductance=C(num="0", den="1"))
    with pytest.raises(OperationDomainValidationError) as error:
        compute_laplacian(LaplacianRequest(network=_laplacian_net(2, edge)))
    assert (
        error.value.errors()[0]["type"] == "electrical_network.conductance_not_positive"
    )


def test_contract_rejects_self_loop() -> None:
    edge = ConductanceEdge(source=0, target=0, conductance=C(num="1", den="1"))
    with pytest.raises(OperationDomainValidationError) as error:
        compute_laplacian(LaplacianRequest(network=_laplacian_net(2, edge)))
    assert (
        error.value.errors()[0]["type"]
        == "electrical_network.edge_endpoints_not_distinct"
    )


def test_contract_rejects_duplicate_edges() -> None:
    net = _laplacian_net(3, _edge(0, 1, "1", "1"), _edge(1, 0, "2", "1"))
    with pytest.raises(OperationDomainValidationError) as error:
        compute_laplacian(LaplacianRequest(network=net))
    assert error.value.errors()[0]["type"] == "electrical_network.duplicate_edges"


def test_contract_rejects_nonzero_denominator() -> None:
    with pytest.raises(ValidationError) as error:
        C(num="1", den="0")
    assert error.value.errors()[0]["type"] == "canonical_rational.zero_denominator"


def test_contract_rejects_same_terminals() -> None:
    net = _net(2, _edge(0, 1, "1", "1"))
    req = EffectiveResistanceRequest(network=net, terminal_a=0, terminal_b=0)
    with pytest.raises(OperationDomainValidationError) as error:
        compute_effective_resistance(req)
    assert (
        error.value.errors()[0]["type"] == "electrical_network.terminals_not_distinct"
    )


def test_contract_rejects_vertex_out_of_range() -> None:
    net = _laplacian_net(2, _edge(0, 5, "1", "1"))
    with pytest.raises(OperationDomainValidationError) as error:
        compute_laplacian(LaplacianRequest(network=net))
    assert (
        error.value.errors()[0]["type"] == "electrical_network.edge_vertex_out_of_range"
    )


# ------------------------------------------------------------------ review root-cause fixes


def test_contract_rejects_disconnected_effective_resistance() -> None:
    """Deleting one Laplacian row/column still leaves a singular component."""
    net = _net(4, _edge(0, 1, "1", "1"), _edge(2, 3, "1", "1"))
    req = EffectiveResistanceRequest(network=net, terminal_a=0, terminal_b=1)
    with pytest.raises(OperationDomainValidationError) as error:
        compute_effective_resistance(req)
    assert error.value.errors()[0]["type"] == "electrical_network.network_not_connected"


def test_contract_rejects_disconnected_node_potentials() -> None:
    net = _net(4, _edge(0, 1, "1", "1"), _edge(2, 3, "1", "1"))
    req = NodePotentialRequest(network=net, source=0, sink=1)
    with pytest.raises(OperationDomainValidationError) as error:
        compute_node_potentials(req)
    assert error.value.errors()[0]["type"] == "electrical_network.network_not_connected"


def test_contract_rejects_isolated_vertex() -> None:
    net = _net(4, _edge(1, 2, "1", "1"))
    req = EffectiveResistanceRequest(network=net, terminal_a=1, terminal_b=2)
    with pytest.raises(OperationDomainValidationError) as error:
        compute_effective_resistance(req)
    assert error.value.errors()[0]["type"] == "electrical_network.network_not_connected"


def test_contract_rejects_oversized_conductance() -> None:
    edge = ConductanceEdge(
        source=0,
        target=1,
        conductance=C(num="9" * 51, den="1"),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compute_laplacian(LaplacianRequest(network=_laplacian_net(2, edge)))
    assert (
        error.value.errors()[0]["type"]
        == "electrical_network.conductance_exceeds_digit_bound"
    )


def test_contract_accepts_boundary_conductance() -> None:
    """A 50-digit conductance is the declared maximum and must be accepted."""
    numerator = "9" * 50
    net = _net(2, _edge(0, 1, numerator, "1"))
    req = EffectiveResistanceRequest(network=net, terminal_a=0, terminal_b=1)
    result = compute_effective_resistance(req)
    assert result.effective_resistance.as_fraction() == Fraction(1, int(numerator))
