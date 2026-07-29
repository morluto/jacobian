from __future__ import annotations

from pathlib import Path

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.capabilities import CapabilityService
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.geometry import GEOMETRY_BUNDLE
from jacobian.geometry_verification import install_geometry_checker
from jacobian.memory import ResearchMemory
from jacobian.operation_installation import OperationInstaller
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry
from jacobian.store import ArtifactStore
from jacobian.verification import VerificationService

ZERO = {"num": "0", "den": "1"}
ONE = {"num": "1", "den": "1"}
TWO = {"num": "2", "den": "1"}
P0 = {"x": ZERO, "y": ZERO}
PX = {"x": TWO, "y": ZERO}
PY = {"x": ZERO, "y": TWO}
PXY = {"x": TWO, "y": TWO}


class _GeometryRuntime:
    def __init__(self, root: Path) -> None:
        self.store = ArtifactStore(root)
        self.schemas = SchemaRegistry(self.store)
        self.artifacts = ArtifactService(self.store, self.schemas)
        self.checkers = CheckerRegistry(self.store.db_path)
        self.verification = VerificationService(self.store, self.checkers)
        self.capabilities = CapabilityService(
            self.store, ResearchMemory(self.store, self.schemas)
        )
        self.geometry = OperationInstaller(
            self.store, self.schemas, self.artifacts
        ).install(GEOMETRY_BUNDLE)
        for adapter in self.geometry.adapters:
            self.capabilities.register(adapter)


def _runtime_with_geometry_checker(root: Path) -> _GeometryRuntime:
    runtime = _GeometryRuntime(root)
    adapter, _installation = install_geometry_checker(
        runtime.store,
        runtime.schemas,
        runtime.artifacts,
        runtime.geometry,
        runtime.verification,
        runtime.checkers,
        authorize_checker=True,
    )
    assert adapter is not None
    runtime.capabilities.register(adapter)
    return runtime


def test_geometry_checker_availability_does_not_grant_authority(
    tmp_path: Path,
) -> None:
    runtime = _GeometryRuntime(tmp_path)

    adapter, installation = install_geometry_checker(
        runtime.store,
        runtime.schemas,
        runtime.artifacts,
        runtime.geometry,
        runtime.verification,
        runtime.checkers,
        authorize_checker=False,
    )

    assert adapter is None
    assert installation.checker_id is None


@pytest.mark.parametrize(
    ("operation_id", "payload"),
    [
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
    ],
)
def test_selected_geometry_results_verify_through_public_dispatch(
    tmp_path: Path,
    operation_id: str,
    payload: dict[str, object],
) -> None:
    runtime = _runtime_with_geometry_checker(tmp_path)
    computed = runtime.capabilities.invoke(
        CapabilityRequest(capability_id=operation_id, input=payload)
    )

    verified = runtime.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED_RESULT"
    assert verified.output["operation_id"] == operation_id
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.assurance.verification_record_uri is not None


def test_mutated_geometry_candidate_is_rejected_without_false_conclusion(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_geometry_checker(tmp_path)
    computed = runtime.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.points.compute.squared_distance",
            input={"first": P0, "second": PXY},
        )
    )
    geometry = runtime.geometry
    mutated = runtime.artifacts.put(
        schema_uri=geometry.result_schema_uris[
            "geometry.points.compute.squared_distance"
        ],
        semantics_uri=geometry.semantics_uri,
        payload={"value": {"num": "7", "den": "1"}},
        parents=(computed.output["input_uri"],),
        summary="adversarial wrong squared distance",
    )

    rejected = runtime.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": mutated.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert rejected.assurance.verification_record_uri is None


def test_schema_valid_false_simple_polygon_decision_is_rejected(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_geometry_checker(tmp_path)
    computed = runtime.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.polygon.simple.decide",
            input={"points": [P0, PXY, PY, PX]},
        )
    )
    mutated = runtime.artifacts.put(
        schema_uri=runtime.geometry.result_schema_uris[
            "geometry.polygon.simple.decide"
        ],
        semantics_uri=runtime.geometry.semantics_uri,
        payload={
            "vertex_count": 4,
            "is_simple": True,
            "checked_edge_pairs": 6,
            "witness": None,
        },
        parents=(computed.output["input_uri"],),
        summary="adversarial false simple-polygon decision",
    )

    rejected = runtime.capabilities.invoke(
        CapabilityRequest(
            capability_id="geometry.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": mutated.artifact_uri},
        )
    )

    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert rejected.assurance.verification_record_uri is None
