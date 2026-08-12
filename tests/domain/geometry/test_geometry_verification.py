from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.checker_operations import derive_verification_capability_id
from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.geometry import build_geometry_bundle
from jacobian.runtime.config import CheckerAuthorityMode

ZERO = {"num": "0", "den": "1"}
ONE = {"num": "1", "den": "1"}
TWO = {"num": "2", "den": "1"}
P0 = {"x": ZERO, "y": ZERO}
PX = {"x": TWO, "y": ZERO}
PY = {"x": ZERO, "y": TWO}
PXY = {"x": TWO, "y": TWO}


@pytest.fixture
def geometry_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install geometry and its exact checkers without the full portfolio."""

    with open_exact_domain_services(
        tmp_path / "state",
        build_geometry_bundle(),
    ) as services:
        yield services


def test_geometry_checker_availability_does_not_grant_authority(
    tmp_path: Path,
) -> None:
    with open_exact_domain_services(
        tmp_path / "state",
        build_geometry_bundle(),
        checker_authority=CheckerAuthorityMode.NONE,
    ) as services:
        catalog_ids = {
            item.capability_id
            for item in services.core.capabilities.catalog().capabilities
        }
        assert any(
            item.endswith(".compute") or ".decide" in item for item in catalog_ids
        )
        assert not any(item.endswith(".verify") for item in catalog_ids)
        assert not services.installation.authorizes_bundled_checkers


def _weighted_square() -> dict[str, object]:
    return {
        "polygon": {"points": [P0, PX, PXY, PY]},
        "diagonal_weights": [
            {"first": 0, "second": 2, "weight": ONE},
            {"first": 1, "second": 3, "weight": TWO},
        ],
        "objective": "NON_HULL_DIAGONAL_WEIGHT_SUM",
    }


def test_minimum_weight_triangulation_charges_one_diagonal_once(
    geometry_services: DomainTestServices,
) -> None:
    payload = _weighted_square()
    computed = geometry_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.polygon.triangulation.minimum_weight.compute",
            input=payload,
        )
    )

    assert computed.output["result"]["optimum"] == ONE
    assert computed.output["result"]["diagonals"] == [
        {"first": 0, "second": 2, "weight": ONE}
    ]
    assert computed.output["result"]["triangles"] == [
        {"vertices": [0, 1, 2]},
        {"vertices": [0, 2, 3]},
    ]

    verified = geometry_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.polygon.triangulation.minimum_weight.verify",
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.verification_record_uri is not None


def test_triangulation_checker_rejects_double_counted_cost(
    geometry_services: DomainTestServices,
) -> None:
    payload = _weighted_square()
    computed = geometry_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.polygon.triangulation.minimum_weight.compute",
            input=payload,
        )
    )
    forged = dict(computed.output["result"])
    forged["optimum"] = TWO

    rejected = geometry_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.polygon.triangulation.minimum_weight.verify",
            input={"input": payload, "candidate": forged},
        )
    )
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"


_GEOMETRY_CASES = (
    (
        "geometry.points.compute.squared_distance",
        {"first": P0, "second": PXY},
    ),
    (
        "geometry.segment.compute.midpoint",
        {"first": P0, "second": PXY},
    ),
    (
        "geometry.points.compute.convex_hull",
        {"points": [PXY, P0, PY, PX, {"x": ONE, "y": ONE}]},
    ),
    (
        "geometry.segments.intersection.compute",
        {
            "first": {"start": P0, "end": PXY},
            "second": {"start": PX, "end": PY},
        },
    ),
    (
        "geometry.polygon.simple.decide",
        {"points": [P0, PXY, PY, PX]},
    ),
    (
        "geometry.polygon.point.classify",
        {
            "polygon": {"points": [P0, PX, PXY, PY]},
            "point": {"x": ONE, "y": ONE},
        },
    ),
    (
        "geometry.triangle.compute.orientation",
        {"first": P0, "second": PX, "third": PY},
    ),
    (
        "geometry.triangle.compute.centroid",
        {"first": P0, "second": PX, "third": PY},
    ),
)


def test_selected_geometry_results_verify_through_public_dispatch(
    geometry_services: DomainTestServices,
) -> None:
    for operation_id, payload in _GEOMETRY_CASES:
        computed = geometry_services.core.capabilities.invoke(
            CapabilityRequest(capability_id=operation_id, input=payload)
        )
        verified = geometry_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=derive_verification_capability_id(operation_id),
                input={"input": payload, "candidate": computed.output["result"]},
            )
        )

        assert verified.execution.status is ExecutionStatus.COMPLETED
        assert verified.output["status"] == "VERIFIED"
        assert verified.output["operation_id"] == operation_id
        assert verified.verification_record_uri is not None


def test_mutated_geometry_candidate_is_rejected_without_false_conclusion(
    geometry_services: DomainTestServices,
) -> None:
    geometry_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.points.compute.squared_distance",
            input={"first": P0, "second": PXY},
        )
    )
    rejected = geometry_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.points.squared_distance.verify",
            input={
                "input": {"first": P0, "second": PXY},
                "candidate": {"value": {"num": "7", "den": "1"}},
            },
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.verification_record_uri is None


def test_schema_valid_false_simple_polygon_decision_is_rejected(
    geometry_services: DomainTestServices,
) -> None:
    geometry_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.polygon.simple.decide",
            input={"points": [P0, PXY, PY, PX]},
        )
    )
    rejected = geometry_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.polygon.simple.verify",
            input={
                "input": {"points": [P0, PXY, PY, PX]},
                "candidate": {
                    "vertex_count": 4,
                    "is_simple": True,
                    "checked_edge_pairs": 6,
                    "witness": None,
                },
            },
        )
    )

    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.verification_record_uri is None
