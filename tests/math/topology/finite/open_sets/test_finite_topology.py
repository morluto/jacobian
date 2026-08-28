"""Correctness and contract tests for finite topological spaces."""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.finite.open_sets import (
    FiniteTopology,
    PointMap,
    closure,
    connected_components,
    continuity,
    interior,
    is_t0,
    minimal_open_neighborhoods,
    specialization_preorder,
)
from jacobian.math.topology.finite.open_sets._models import (
    BeatPointsRequest,
    ConnectedComponentsRequest,
    ContinuityRequest,
    SpecializationPreorderRequest,
)
from jacobian.math.topology.finite.open_sets._tools import (
    TOOLS,
    compute_beat_points,
    compute_connected_components,
    compute_continuity,
    compute_specialization_preorder,
)


def _sierpinski() -> FiniteTopology:
    return FiniteTopology(point_count=2, open_sets=((), (1,), (0, 1)))


def _discrete(size: int) -> FiniteTopology:
    return FiniteTopology(
        point_count=size,
        open_sets=tuple(
            tuple(point for point in range(size) if mask & (1 << point))
            for mask in range(1 << size)
        ),
    )


def _indiscrete(size: int) -> FiniteTopology:
    return FiniteTopology(point_count=size, open_sets=((), tuple(range(size))))


def test_public_surface_keeps_closure_and_interior_native_only() -> None:
    assert tuple(tool.operation_id for tool in TOOLS) == (
        "topology.specialization_preorder.compute",
        "topology.connected_components.compute",
        "topology.is_continuous.compute",
        "topology.beat_points.compute",
    )
    space = _sierpinski()
    assert closure(space, (0,)) == frozenset({0})
    assert closure(space, (1,)) == frozenset({0, 1})
    assert interior(space, (0,)) == frozenset()
    assert interior(space, (1,)) == frozenset({1})


def test_topology_axioms_are_validated_at_the_value_boundary() -> None:
    with pytest.raises(ValidationError) as exc_info:
        FiniteTopology(
            point_count=3,
            open_sets=((), (0,), (1,), (0, 1, 2)),
        )
    assert (
        exc_info.value.errors()[0]["type"] == "finite_topology.not_closed_under_unions"
    )
    with pytest.raises(ValidationError) as exc_info:
        FiniteTopology(
            point_count=3,
            open_sets=((), (0, 1), (1, 2), (0, 1, 2)),
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "finite_topology.not_closed_under_intersections"
    )
    with pytest.raises(ValidationError) as exc_info:
        FiniteTopology(point_count=2, open_sets=((), (1, 0)))
    assert (
        exc_info.value.errors()[0]["type"] == "finite_topology.open_set_not_canonical"
    )
    with pytest.raises(ValidationError) as exc_info:
        FiniteTopology(point_count=1, open_sets=((), (0,), (0,)))
    assert (
        exc_info.value.errors()[0]["type"] == "finite_topology.open_sets_not_distinct"
    )


def test_carriers_are_structural_while_wire_requests_own_operation_bounds() -> None:
    """A native consumer may use a larger finite carrier than the MCP tools."""

    topology = _indiscrete(33)
    assert topology.point_count == 33
    assert len(specialization_preorder(topology)) == 33

    value_schema = FiniteTopology.model_json_schema()
    assert "maximum" not in value_schema["properties"]["point_count"]
    assert "maxItems" not in value_schema["properties"]["open_sets"]

    with pytest.raises(OperationDomainValidationError, match="at most 32 points"):
        compute_specialization_preorder(
            SpecializationPreorderRequest(topology=topology)
        )

    request_schema = SpecializationPreorderRequest.model_json_schema()
    topology_schema = request_schema["properties"]["topology"]
    assert topology_schema["properties"]["point_count"]["maximum"] == 32
    assert topology_schema["properties"]["open_sets"]["maxItems"] == 1_024

    map_schema = PointMap.model_json_schema()
    assert "maximum" not in map_schema["properties"]["domain_point_count"]
    assert "maximum" not in map_schema["properties"]["codomain_point_count"]
    assert "maxItems" not in map_schema["properties"]["values"]

    continuity_schema = ContinuityRequest.model_json_schema()
    point_map_schema = continuity_schema["properties"]["point_map"]
    assert point_map_schema["properties"]["domain_point_count"]["maximum"] == 32
    assert point_map_schema["properties"]["codomain_point_count"]["maximum"] == 32
    assert point_map_schema["properties"]["values"]["maxItems"] == 32


def test_specialization_orientation_is_explicit_and_bound() -> None:
    result = compute_specialization_preorder(
        SpecializationPreorderRequest(topology=_sierpinski())
    )
    assert result.relation == ((True, True), (False, True))
    assert result.orientation == "RELATION_X_Y_MEANS_X_IN_CLOSURE_OF_SINGLETON_Y"
    assert result.topology == _sierpinski()


def test_minimal_neighborhoods_and_components() -> None:
    assert minimal_open_neighborhoods(_sierpinski()) == (
        frozenset({0, 1}),
        frozenset({1}),
    )
    connected = compute_connected_components(
        ConnectedComponentsRequest(topology=_sierpinski())
    )
    assert connected.components == ((0, 1),)
    assert connected.component_count == 1
    discrete = compute_connected_components(
        ConnectedComponentsRequest(topology=_discrete(3))
    )
    assert discrete.components == ((0,), (1,), (2,))


def test_continuity_returns_an_exact_counterexample() -> None:
    point_map = PointMap(domain_point_count=2, codomain_point_count=2, values=(0, 1))
    result = compute_continuity(
        ContinuityRequest(
            domain=_indiscrete(2),
            codomain=_sierpinski(),
            point_map=point_map,
        )
    )
    assert result.is_continuous is False
    assert result.violating_open_set == (1,)
    assert result.violating_preimage == (1,)

    identity = compute_continuity(
        ContinuityRequest(
            domain=_sierpinski(), codomain=_sierpinski(), point_map=point_map
        )
    )
    assert identity.is_continuous is True
    assert identity.violating_open_set is None
    assert identity.violating_preimage is None


def test_continuity_request_binds_map_carrier_sizes() -> None:
    request = ContinuityRequest(
        domain=_sierpinski(),
        codomain=_sierpinski(),
        point_map=PointMap(domain_point_count=1, codomain_point_count=2, values=(0,)),
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_continuity(request)
    assert exc_info.value.errors()[0]["type"] == "finite_topology.map_carrier_mismatch"


def test_beat_points_use_strict_t0_order_and_return_witnesses() -> None:
    result = compute_beat_points(BeatPointsRequest(topology=_sierpinski()))
    assert tuple((entry.point, entry.witness) for entry in result.down_beat_points) == (
        (1, 0),
    )
    assert tuple((entry.point, entry.witness) for entry in result.up_beat_points) == (
        (0, 1),
    )


def test_non_t0_beat_point_request_fails_closed() -> None:
    assert is_t0(_indiscrete(2)) is False
    with pytest.raises(OperationDomainValidationError, match="T0"):
        compute_beat_points(BeatPointsRequest(topology=_indiscrete(2)))


def _all_topologies(size: int) -> tuple[FiniteTopology, ...]:
    subsets = tuple(
        frozenset(point for point in range(size) if mask & (1 << point))
        for mask in range(1 << size)
    )
    required = {frozenset(), frozenset(range(size))}
    results = []
    for family_mask in range(1 << len(subsets)):
        family = {
            subset for index, subset in enumerate(subsets) if family_mask & (1 << index)
        }
        if not required <= family:
            continue
        if any(left | right not in family for left in family for right in family):
            continue
        if any(left & right not in family for left in family for right in family):
            continue
        results.append(
            FiniteTopology(
                point_count=size,
                open_sets=tuple(tuple(sorted(open_set)) for open_set in family),
            )
        )
    return tuple(results)


def test_all_topologies_through_three_points_match_set_theoretic_oracles() -> None:
    for size in range(1, 4):
        carrier = frozenset(range(size))
        for topology in _all_topologies(size):
            opens = {frozenset(open_set) for open_set in topology.open_sets}
            closed = {carrier - open_set for open_set in opens}
            relation = specialization_preorder(topology)
            expected_relation = tuple(
                tuple(
                    all(upper in open_set for open_set in opens if lower in open_set)
                    for upper in range(size)
                )
                for lower in range(size)
            )
            assert relation == expected_relation
            for subset in (
                frozenset(points)
                for length in range(size + 1)
                for points in itertools.combinations(range(size), length)
            ):
                expected_closure = frozenset.intersection(
                    *(closed_set for closed_set in closed if subset <= closed_set)
                )
                expected_interior = frozenset().union(
                    *(open_set for open_set in opens if open_set <= subset)
                )
                assert closure(topology, subset) == expected_closure
                assert interior(topology, subset) == expected_interior


def test_all_two_point_maps_match_open_preimage_oracle() -> None:
    spaces = _all_topologies(2)
    for domain in spaces:
        domain_opens = {frozenset(open_set) for open_set in domain.open_sets}
        for codomain in spaces:
            for values in itertools.product(range(2), repeat=2):
                point_map = PointMap(
                    domain_point_count=2,
                    codomain_point_count=2,
                    values=values,
                )
                expected = all(
                    frozenset(
                        point
                        for point, target in enumerate(values)
                        if target in open_set
                    )
                    in domain_opens
                    for open_set in codomain.open_sets
                )
                assert continuity(domain, codomain, point_map).is_continuous is expected
                assert connected_components(domain)
