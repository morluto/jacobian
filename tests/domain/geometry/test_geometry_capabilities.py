from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.geometry import (
    LinePairRequest,
    PointLineRequest,
    PointPairRequest,
    PointQuadrupleRequest,
    PointSetRequest,
    PointTripleRequest,
    PolygonRequest,
    SegmentIntersectionRequest,
    SimplePolygonPointRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.geometry import build_geometry_bundle


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(tmp_path / "state", build_geometry_bundle()) as services:
        yield services


ZERO = {"num": "0", "den": "1"}
ONE = {"num": "1", "den": "1"}
TWO = {"num": "2", "den": "1"}
P0 = {"x": ZERO, "y": ZERO}
PX = {"x": TWO, "y": ZERO}
PY = {"x": ZERO, "y": TWO}
PXY = {"x": TWO, "y": TWO}


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
        for descriptor in domain_services.core.capabilities.catalog().capabilities
        if descriptor.capability_id == "geometry.segment.compute.midpoint"
    )
    example = descriptor.invocation_examples[0]

    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=descriptor.capability_id,
            mode=example.mode,
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


def test_geometry_capabilities_are_distinct_and_every_contract_completes(
    domain_services,
) -> None:
    line_x = {"first": P0, "second": PX}
    line_y = {"first": P0, "second": PY}
    payloads = {
        PointPairRequest: {"first": P0, "second": PXY},
        PointTripleRequest: {"first": P0, "second": PX, "third": PY},
        PointQuadrupleRequest: {
            "first": P0,
            "second": PX,
            "third": PY,
            "fourth": PXY,
        },
        LinePairRequest: {"first_line": line_x, "second_line": line_y},
        PointLineRequest: {"point": PXY, "line": line_x},
        PolygonRequest: {"points": [P0, PX, PY]},
        PointSetRequest: {"points": [P0, PX, PY, PXY]},
        SegmentIntersectionRequest: {
            "first": _segment(P0, PXY),
            "second": _segment(PX, PY),
        },
        SimplePolygonPointRequest: {
            "polygon": {"points": [P0, PX, PXY, PY]},
            "point": {"x": ONE, "y": ONE},
        },
    }
    ids = [
        operation.capability_id for operation in build_geometry_bundle().capabilities
    ]

    assert ids, "expected geometry capabilities"
    assert len(ids) == len(set(ids))
    for operation in build_geometry_bundle().capabilities:
        result = domain_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=operation.capability_id,
                input=payloads[operation.request_model],
            )
        )
        assert result.execution.status is ExecutionStatus.COMPLETED, (
            operation.capability_id,
            result.diagnostics,
        )
        assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
        assert len(result.artifact_uris) == 2


def test_geometry_exact_outputs_are_inline_and_materialized(domain_services) -> None:

    distance = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.points.compute.squared_distance",
            input={"first": P0, "second": PXY},
        )
    )
    circle = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.triangle.compute.circumcircle",
            input={"first": P0, "second": PX, "third": PY},
        )
    )

    assert distance.output["result"] == {"value": {"num": "8", "den": "1"}}
    assert circle.output["result"] == {
        "center": {"x": ONE, "y": ONE},
        "radius_squared": {"num": "2", "den": "1"},
    }
    assert (
        domain_services.core.store.get(distance.output["result_uri"]).payload
        == distance.output["result"]
    )


def test_convex_hull_returns_segment_endpoints_for_two_points(
    domain_services,
) -> None:

    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.points.compute.convex_hull",
            input={"points": [PXY, P0]},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"points": [P0, PXY]}


def test_convex_hull_returns_extreme_endpoints_for_collinear_points(
    domain_services,
) -> None:
    middle = {"x": ONE, "y": ONE}

    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.points.compute.convex_hull",
            input={"points": [middle, PXY, P0]},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"points": [P0, PXY]}


def test_degenerate_geometry_fails_before_artifact_writes(domain_services) -> None:

    invalid_line = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.lines.compute.intersection",
            input={
                "first_line": {"first": P0, "second": P0},
                "second_line": {"first": P0, "second": PX},
            },
        )
    )
    collinear_circle = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.triangle.compute.circumcircle",
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


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    (
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
    ),
)
def test_closed_segment_intersection_preserves_degenerate_classification(
    domain_services,
    first: dict[str, object],
    second: dict[str, object],
    expected: dict[str, object],
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.segments.intersection.compute",
            input={"first": first, "second": second},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == expected
    assert len(result.artifact_uris) == 2


@pytest.mark.parametrize(
    ("points", "is_simple", "witness_status"),
    (
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
    ),
)
def test_simple_polygon_decision_exposes_first_exact_violation(
    domain_services,
    points: list[dict[str, object]],
    is_simple: bool,
    witness_status: str | None,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.polygon.simple.decide",
            input={"points": points},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["is_simple"] is is_simple
    witness = result.output["result"]["witness"]
    assert (None if witness is None else witness["intersection"]["status"]) == (
        witness_status
    )


@pytest.mark.parametrize(
    ("point", "classification"),
    (
        (_point(1, 1), "INSIDE"),
        (_point(2, 1), "BOUNDARY"),
        (_point(3, 1), "OUTSIDE"),
    ),
)
def test_simple_polygon_point_classification_is_exact_and_boundary_aware(
    domain_services,
    point: dict[str, object],
    classification: str,
) -> None:
    polygon = [_point(0, 0), _point(2, 0), _point(2, 2), _point(0, 2)]
    forward = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.polygon.point.classify",
            input={"polygon": {"points": polygon}, "point": point},
        )
    )
    reverse = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.polygon.point.classify",
            input={"polygon": {"points": list(reversed(polygon))}, "point": point},
        )
    )

    assert forward.output["result"]["classification"] == classification
    assert reverse.output["result"] == forward.output["result"]


def test_point_classification_rejects_non_simple_polygon_before_writes(
    domain_services,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.polygon.point.classify",
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


@pytest.mark.parametrize(
    "points",
    (
        [_point(0, 0), _point(2, 0), _point(0, 2), _point(0, 0)],
        [_point(0, 0), _point(2, 0), _point(2, 0), _point(0, 2)],
    ),
)
def test_polygon_ring_rejects_repeated_closure_or_zero_edge_before_writes(
    domain_services,
    points: list[dict[str, object]],
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.polygon.simple.decide",
            input={"points": points},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_GEOMETRY_REQUEST"
    assert result.artifact_uris == ()
