from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.geometry import geometry_operations


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(tmp_path / "state", geometry_operations()) as services:
        yield services


ZERO = {"num": "0", "den": "1"}
ONE = {"num": "1", "den": "1"}
TWO = {"num": "2", "den": "1"}
P0 = {"x": ZERO, "y": ZERO}
PX = {"x": TWO, "y": ZERO}
PY = {"x": ZERO, "y": TWO}
PXY = {"x": TWO, "y": TWO}
LARGE_CANONICAL_INTEGER = "1" + ("0" * 4_999) + "1"


def _point(x: int, y: int) -> dict[str, dict[str, str]]:
    return {
        "x": {"num": str(x), "den": "1"},
        "y": {"num": str(y), "den": "1"},
    }


def _segment(
    start: dict[str, object],
    end: dict[str, object],
) -> dict[str, object]:
    return {"start": start, "end": end}


def test_segment_midpoint_example_is_directly_invocable(domain_services) -> None:
    descriptor = next(
        descriptor
        for descriptor in domain_services.core.operations.snapshot().operations
        if descriptor.operation_id == "geometry.segment.compute.midpoint"
    )
    example = descriptor.examples[0]

    result = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id=descriptor.operation_id,
            input=example.input,
        )
    )

    assert example.input == {
        "first": {"x": ZERO, "y": ZERO},
        "second": {"x": ONE, "y": ZERO},
    }
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {
        "point": {
            "x": {"num": "1", "den": "2"},
            "y": ZERO,
        }
    }


def test_geometry_operations_have_distinct_ids() -> None:
    ids = [operation.operation_id for operation in geometry_operations()]

    assert ids, "expected geometry operations"
    assert len(ids) == len(set(ids))


def test_geometry_exact_outputs_are_inline(domain_services) -> None:

    distance = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.points.compute.squared_distance",
            input={"first": P0, "second": PXY},
        )
    )
    circle = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.triangle.compute.circumcircle",
            input={"first": P0, "second": PX, "third": PY},
        )
    )

    assert distance.output["result"] == {"value": {"num": "8", "den": "1"}}
    assert circle.output["result"] == {
        "center": {"x": ONE, "y": ONE},
        "radius_squared": {"num": "2", "den": "1"},
    }
    assert distance.artifact_uris == ()
    assert circle.artifact_uris == ()


def test_squared_distance_accepts_contract_sized_coordinates(domain_services) -> None:
    coordinate = {"num": LARGE_CANONICAL_INTEGER, "den": "1"}
    point = {"x": coordinate, "y": ZERO}

    result = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.points.compute.squared_distance",
            input={"first": point, "second": point},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"value": ZERO}


def test_convex_hull_returns_segment_endpoints_for_two_points(
    domain_services,
) -> None:

    result = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.points.compute.convex_hull",
            input={"points": [PXY, P0]},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"points": [P0, PXY]}


def test_convex_hull_returns_extreme_endpoints_for_collinear_points(
    domain_services,
) -> None:
    middle = {"x": ONE, "y": ONE}

    result = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.points.compute.convex_hull",
            input={"points": [middle, PXY, P0]},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"points": [P0, PXY]}


def test_degenerate_geometry_fails_before_artifact_writes(domain_services) -> None:

    invalid_line = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.lines.compute.intersection",
            input={
                "first_line": {"first": P0, "second": P0},
                "second_line": {"first": P0, "second": PX},
            },
        )
    )
    collinear_circle = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.triangle.compute.circumcircle",
            input={
                "first": P0,
                "second": {"x": ONE, "y": ONE},
                "third": PXY,
            },
        )
    )

    assert invalid_line.execution.status is ExecutionStatus.ERROR
    assert invalid_line.diagnostics[0].code == "INVALID_GEOMETRY_REQUEST"
    assert collinear_circle.execution.status is ExecutionStatus.ERROR
    assert collinear_circle.diagnostics[0].code == "GEOMETRY_OPERATION_NOT_APPLICABLE"
    assert invalid_line.artifact_uris == ()
    assert collinear_circle.artifact_uris == ()


def test_closed_segment_intersection_preserves_degenerate_classification(
    domain_services,
) -> None:
    cases = (
        (
            _segment(_point(0, 0), _point(2, 2)),
            _segment(_point(0, 2), _point(2, 0)),
            {
                "status": "POINT",
                "point": _point(1, 1),
                "contact_kind": "PROPER",
                "overlap": None,
            },
        ),
        (
            _segment(_point(0, 0), _point(2, 0)),
            _segment(_point(2, 0), _point(2, 2)),
            {
                "status": "POINT",
                "point": _point(2, 0),
                "contact_kind": "ENDPOINT_TOUCH",
                "overlap": None,
            },
        ),
        (
            _segment(_point(0, 0), _point(3, 0)),
            _segment(_point(1, 0), _point(2, 0)),
            {
                "status": "OVERLAP",
                "point": None,
                "contact_kind": None,
                "overlap": _segment(_point(1, 0), _point(2, 0)),
            },
        ),
        (
            _segment(_point(0, 0), _point(0, 0)),
            _segment(_point(-1, 0), _point(1, 0)),
            {
                "status": "POINT",
                "point": _point(0, 0),
                "contact_kind": "DEGENERATE_TOUCH",
                "overlap": None,
            },
        ),
        (
            _segment(_point(0, 0), _point(1, 0)),
            _segment(_point(2, 0), _point(3, 0)),
            {
                "status": "DISJOINT",
                "point": None,
                "contact_kind": None,
                "overlap": None,
            },
        ),
    )
    for first, second, expected in cases:
        result = domain_services.core.operations.invoke(
            OperationRequest(
                operation_id="geometry.segments.intersection.compute",
                input={"first": first, "second": second},
            )
        )

        assert result.execution.status is ExecutionStatus.COMPLETED, expected
        assert result.output["result"] == expected
        assert result.artifact_uris == ()


def test_simple_polygon_decision_exposes_first_exact_violation(
    domain_services,
) -> None:
    cases = (
        (
            [_point(0, 0), _point(2, 0), _point(2, 2), _point(0, 2)],
            True,
            None,
        ),
        (
            [_point(0, 0), _point(2, 2), _point(0, 2), _point(2, 0)],
            False,
            "POINT",
        ),
        (
            [_point(0, 0), _point(3, 0), _point(1, 0), _point(1, 2)],
            False,
            "OVERLAP",
        ),
    )
    for points, is_simple, witness_status in cases:
        result = domain_services.core.operations.invoke(
            OperationRequest(
                operation_id="geometry.polygon.simple.decide",
                input={"points": points},
            )
        )

        assert result.execution.status is ExecutionStatus.COMPLETED, points
        assert result.output["result"]["is_simple"] is is_simple, points
        witness = result.output["result"]["witness"]
        assert (None if witness is None else witness["intersection"]["status"]) == (
            witness_status
        ), points


def test_simple_polygon_point_classification_is_exact_and_boundary_aware(
    domain_services,
) -> None:
    cases = (
        (_point(1, 1), "INSIDE"),
        (_point(2, 1), "BOUNDARY"),
        (_point(3, 1), "OUTSIDE"),
    )
    polygon = [_point(0, 0), _point(2, 0), _point(2, 2), _point(0, 2)]
    for point, classification in cases:
        forward = domain_services.core.operations.invoke(
            OperationRequest(
                operation_id="geometry.polygon.point.classify",
                input={"polygon": {"points": polygon}, "point": point},
            )
        )
        reverse = domain_services.core.operations.invoke(
            OperationRequest(
                operation_id="geometry.polygon.point.classify",
                input={
                    "polygon": {"points": list(reversed(polygon))},
                    "point": point,
                },
            )
        )

        assert forward.output["result"]["classification"] == classification, point
        assert reverse.output["result"] == forward.output["result"], point


def test_point_classification_rejects_non_simple_polygon_before_writes(
    domain_services,
) -> None:
    result = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.polygon.point.classify",
            input={
                "polygon": {
                    "points": [
                        _point(0, 0),
                        _point(2, 2),
                        _point(0, 2),
                        _point(2, 0),
                    ]
                },
                "point": _point(1, 1),
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_GEOMETRY_REQUEST"
    assert result.artifact_uris == ()


def test_polygon_ring_rejects_repeated_closure_or_zero_edge_before_writes(
    domain_services,
) -> None:
    cases = (
        [_point(0, 0), _point(2, 0), _point(0, 2), _point(0, 0)],
        [_point(0, 0), _point(2, 0), _point(2, 0), _point(0, 2)],
    )
    for points in cases:
        result = domain_services.core.operations.invoke(
            OperationRequest(
                operation_id="geometry.polygon.simple.decide",
                input={"points": points},
            )
        )

        assert result.execution.status is ExecutionStatus.ERROR, points
        assert result.diagnostics[0].code == "INVALID_GEOMETRY_REQUEST", points
        assert result.artifact_uris == (), points
