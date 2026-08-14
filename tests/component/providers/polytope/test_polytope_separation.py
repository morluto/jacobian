"""Z3-backed polytope separation and checker integration contracts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.rationals import rational_payload as _q
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.canonical import canonicalize_json
from jacobian.checker_authorization import install_polytope_checkers
from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.polytope import PolytopeSeparateRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.operation_projection import project_operation_result
from jacobian.polytope_operations import PolytopeSeparationAdapter


@dataclass(frozen=True, slots=True)
class PolytopeTestServices(DomainTestServices):
    witness_checker_id: str
    certificate_checker_id: str


@pytest.fixture
def polytope_services(tmp_path: Path) -> Iterator[PolytopeTestServices]:
    with open_domain_services(
        tmp_path / "state",
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        polytope = services.polytope
        with atomic_installation(services.core):
            checkers = install_polytope_checkers(
                services.core.checkers,
                claim_schema_uri=polytope.claim_schema_uri,
                semantics_uri=polytope.semantics_uri,
                point_schema_uri=polytope.point_schema_uri,
            )
        assert checkers.witness_checker_id is not None
        assert checkers.certificate_checker_id is not None
        yield PolytopeTestServices(
            core=services.core,
            verification=services.verification,
            polytope=services.polytope,
            installation=services.installation,
            witness_checker_id=checkers.witness_checker_id,
            certificate_checker_id=checkers.certificate_checker_id,
        )


def _simplex(
    runtime: PolytopeTestServices,
    point: tuple[tuple[int, int], ...],
) -> tuple[str, str]:
    point_artifact = runtime.core.artifacts.put(
        schema_uri=runtime.polytope.point_schema_uri,
        semantics_uri=runtime.polytope.semantics_uri,
        payload={
            "point_schema_version": "1",
            "coordinates": [_q(*value) for value in point],
        },
    )
    generators = runtime.core.artifacts.put(
        schema_uri=runtime.polytope.generator_set_schema_uri,
        semantics_uri=runtime.polytope.semantics_uri,
        payload={
            "generator_set_schema_version": "1",
            "dimension": 3,
            "generators": [
                {"values": [_q(0), _q(0), _q(0)]},
                {"values": [_q(1), _q(0), _q(0)]},
                {"values": [_q(0), _q(1), _q(0)]},
                {"values": [_q(0), _q(0), _q(1)]},
            ],
        },
    )
    return point_artifact.artifact_uri, generators.artifact_uri


def test_backend_failure_keeps_provider_detail_local(
    polytope_services: PolytopeTestServices,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import z3

    point_uri, generators_uri = _simplex(
        polytope_services,
        ((1, 4), (1, 4), (1, 4)),
    )

    def fail_backend(*_args: object, **_kwargs: object) -> None:
        raise z3.Z3Exception("provider=solver internal-id=secret")

    monkeypatch.setattr(polytope_services.polytope, "_convex_weights", fail_backend)
    result = polytope_services.polytope.separate(
        PolytopeSeparateRequest(
            point_uri=point_uri,
            generator_set_uri=generators_uri,
        )
    )
    adapter = PolytopeSeparationAdapter(polytope_services.polytope)
    projected = project_operation_result(
        adapter.invoke(
            adapter.prepare(
                OperationRequest(
                    operation_id="polytope.separate",
                    input={
                        "point_uri": point_uri,
                        "generator_set_uri": generators_uri,
                    },
                )
            )
        )
    )

    assert result.execution.status.value == "ERROR"
    assert projected.execution.status is ExecutionStatus.ERROR
    assert projected.output == {}
    assert projected.artifact_uris == (point_uri, generators_uri)
    assert result.execution.detail == (
        "The exact polytope check failed. Retry with a smaller input; "
        "if it fails again, inspect the local Jacobian log."
    )
    assert "solver" not in result.execution.detail
    assert "internal-id" not in result.execution.detail
    assert "internal-id=secret" in caplog.text


def test_exact_membership_witness_is_independently_replayed(
    polytope_services: PolytopeTestServices,
) -> None:
    point_uri, generators_uri = _simplex(
        polytope_services,
        ((1, 4), (1, 4), (1, 4)),
    )

    proposed = polytope_services.polytope.separate(
        PolytopeSeparateRequest(
            point_uri=point_uri,
            generator_set_uri=generators_uri,
        )
    )

    assert proposed.status.value == "MEMBER"
    assert proposed.witness_uri is not None
    assert proposed.certificate_uri is None
    verified = polytope_services.verification.verify_witness(
        claim_uri=proposed.claim_uri or "",
        candidate_uri=proposed.effective_point_uri or "",
        witness_uri=proposed.witness_uri,
        checker_id=polytope_services.witness_checker_id,
    )
    assert verified.conclusion.value == "TRUE"
    assert verified.verification_record_uri is not None
    assert verified.verification_record_uri is not None


def test_exact_separator_is_generated_then_independently_checked(
    polytope_services: PolytopeTestServices,
) -> None:
    point_uri, generators_uri = _simplex(
        polytope_services,
        ((1, 2), (1, 2), (1, 2)),
    )

    proposed = polytope_services.polytope.separate(
        PolytopeSeparateRequest(
            point_uri=point_uri,
            generator_set_uri=generators_uri,
        )
    )

    assert proposed.status.value == "SEPARATED"
    assert proposed.certificate_uri is not None
    assert proposed.witness_uri is None
    certificate = polytope_services.core.store.get(proposed.certificate_uri).payload
    payload = certificate["payload"]
    coefficients = [
        int(value["num"]) // int(value["den"]) for value in payload["coefficients"]
    ]
    rhs = int(payload["rhs"]["num"]) // int(payload["rhs"]["den"])
    assert math.gcd(*[abs(value) for value in (*coefficients, rhs)]) == 1
    assert payload["margin"] == _q(1, 2)

    verified = polytope_services.verification.verify_certificate(
        certificate_uri=proposed.certificate_uri,
    )
    assert verified.conclusion.value == "TRUE"
    assert verified.verification_record_uri is not None


def test_separator_payload_tampering_fails_closed(
    polytope_services: PolytopeTestServices,
) -> None:
    point_uri, generators_uri = _simplex(
        polytope_services,
        ((1, 2), (1, 2), (1, 2)),
    )
    proposed = polytope_services.polytope.separate(
        PolytopeSeparateRequest(
            point_uri=point_uri,
            generator_set_uri=generators_uri,
        )
    )
    assert proposed.certificate_uri is not None
    original = polytope_services.core.store.get(proposed.certificate_uri)
    tampered = dict(original.payload)
    tampered_payload = dict(tampered["payload"])
    tampered_payload["coefficients"] = [_q(0), _q(0), _q(1)]
    tampered["payload"] = tampered_payload
    tampered["payload_digest"] = (
        "sha256:" + hashlib.sha256(canonicalize_json(tampered_payload)).hexdigest()
    )
    stored = polytope_services.core.store.put(
        schema_uri=original.manifest.schema_uri,
        semantics_uri=original.manifest.semantics_uri,
        payload=tampered,
        parents=original.manifest.parents,
        summary="adversarial separator tampering",
    )

    result = polytope_services.verification.verify_certificate(
        certificate_uri=stored.artifact_uri
    )

    assert result.input.status.value == "REJECTED"
    assert result.conclusion.value == "UNKNOWN"
    assert result.verification_record_uri is None
    assert result.verification_record_uri is None


def test_projection_is_explicit_and_bound_to_derived_artifacts(
    polytope_services: PolytopeTestServices,
) -> None:
    point = polytope_services.core.artifacts.put(
        schema_uri=polytope_services.polytope.point_schema_uri,
        semantics_uri=polytope_services.polytope.semantics_uri,
        payload={
            "point_schema_version": "1",
            "coordinates": [_q(99), _q(1, 2), _q(1, 2), _q(1, 2)],
        },
    )
    generators = polytope_services.core.artifacts.put(
        schema_uri=polytope_services.polytope.generator_set_schema_uri,
        semantics_uri=polytope_services.polytope.semantics_uri,
        payload={
            "generator_set_schema_version": "1",
            "dimension": 4,
            "generators": [
                {"values": [_q(0), _q(0), _q(0), _q(0)]},
                {"values": [_q(7), _q(1), _q(0), _q(0)]},
                {"values": [_q(8), _q(0), _q(1), _q(0)]},
                {"values": [_q(9), _q(0), _q(0), _q(1)]},
            ],
        },
    )

    proposed = polytope_services.polytope.separate(
        PolytopeSeparateRequest(
            point_uri=point.artifact_uri,
            generator_set_uri=generators.artifact_uri,
            projection=(1, 2, 3),
        )
    )

    assert proposed.status.value == "SEPARATED"
    assert proposed.effective_point_uri != point.artifact_uri
    assert proposed.effective_generator_set_uri != generators.artifact_uri
    projected_point = polytope_services.core.store.get(
        proposed.effective_point_uri or ""
    )
    projected_generators = polytope_services.core.store.get(
        proposed.effective_generator_set_uri or ""
    )
    assert projected_point.manifest.parents == (point.artifact_uri,)
    assert projected_generators.manifest.parents == (generators.artifact_uri,)
