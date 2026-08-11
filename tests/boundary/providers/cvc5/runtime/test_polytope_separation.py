from __future__ import annotations

import hashlib
import math

import pytest
from tests.support.rationals import rational_payload as _q

from jacobian.canonical import canonicalize_json
from jacobian.contracts.polytope import PolytopeSeparateRequest
from jacobian.runtime.model import JacobianRuntime


def _simplex(
    runtime: JacobianRuntime,
    point: tuple[tuple[int, int], ...],
) -> tuple[str, str]:
    point_artifact = runtime.core.artifacts.put(
        schema_uri=runtime.services.polytope.point_schema_uri,
        semantics_uri=runtime.services.polytope.semantics_uri,
        payload={
            "point_schema_version": "1",
            "coordinates": [_q(*value) for value in point],
        },
    )
    generators = runtime.core.artifacts.put(
        schema_uri=runtime.services.polytope.generator_set_schema_uri,
        semantics_uri=runtime.services.polytope.semantics_uri,
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
    authorized_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import z3

    point_uri, generators_uri = _simplex(
        authorized_complete_runtime,
        ((1, 4), (1, 4), (1, 4)),
    )

    def fail_backend(*_args: object, **_kwargs: object) -> None:
        raise z3.Z3Exception("provider=solver internal-id=secret")

    monkeypatch.setattr(
        authorized_complete_runtime.services.polytope, "_convex_weights", fail_backend
    )
    result = authorized_complete_runtime.services.polytope.separate(
        PolytopeSeparateRequest(
            point_uri=point_uri,
            generator_set_uri=generators_uri,
        )
    )

    assert result.result.execution.status.value == "ERROR"
    assert result.result.execution.detail == (
        "The exact polytope check failed. Retry with a smaller input; "
        "if it fails again, inspect the local Jacobian log."
    )
    assert "solver" not in result.result.execution.detail
    assert "internal-id" not in result.result.execution.detail
    assert "internal-id=secret" in caplog.text


def test_exact_membership_witness_is_independently_replayed(
    authorized_complete_runtime,
) -> None:
    assert authorized_complete_runtime.portfolio.polytope_checkers is not None
    point_uri, generators_uri = _simplex(
        authorized_complete_runtime,
        ((1, 4), (1, 4), (1, 4)),
    )

    proposed = authorized_complete_runtime.services.polytope.separate(
        PolytopeSeparateRequest(
            point_uri=point_uri,
            generator_set_uri=generators_uri,
        )
    )

    assert proposed.status.value == "MEMBER"
    assert proposed.witness_uri is not None
    assert proposed.certificate_uri is None
    assert proposed.result.assurance.verification.value == "UNVERIFIED"
    verified = authorized_complete_runtime.services.verification.verify_witness(
        claim_uri=proposed.claim_uri or "",
        candidate_uri=proposed.effective_point_uri or "",
        witness_uri=proposed.witness_uri,
        checker_id=authorized_complete_runtime.portfolio.polytope_checkers.witness_checker_id,
    )
    assert verified.conclusion.value == "TRUE"
    assert verified.assurance.arithmetic.value == "EXACT_RATIONAL"
    assert verified.assurance.verification.value == "VERIFIED"
    assert verified.verification_record_uri is not None


def test_exact_separator_is_generated_then_independently_checked(
    authorized_complete_runtime,
) -> None:
    point_uri, generators_uri = _simplex(
        authorized_complete_runtime,
        ((1, 2), (1, 2), (1, 2)),
    )

    proposed = authorized_complete_runtime.services.polytope.separate(
        PolytopeSeparateRequest(
            point_uri=point_uri,
            generator_set_uri=generators_uri,
        )
    )

    assert proposed.status.value == "SEPARATED"
    assert proposed.certificate_uri is not None
    assert proposed.witness_uri is None
    assert proposed.result.assurance.verification.value == "UNVERIFIED"
    certificate = authorized_complete_runtime.core.store.get(
        proposed.certificate_uri
    ).payload
    payload = certificate["payload"]
    coefficients = [
        int(value["num"]) // int(value["den"]) for value in payload["coefficients"]
    ]
    rhs = int(payload["rhs"]["num"]) // int(payload["rhs"]["den"])
    assert math.gcd(*[abs(value) for value in (*coefficients, rhs)]) == 1
    assert payload["margin"] == _q(1, 2)

    verified = authorized_complete_runtime.services.verification.verify_certificate(
        certificate_uri=proposed.certificate_uri,
    )
    assert verified.conclusion.value == "TRUE"
    assert verified.assurance.arithmetic.value == "EXACT_RATIONAL"
    assert verified.assurance.coverage.value == "EXHAUSTIVE"
    assert verified.assurance.verification.value == "VERIFIED"


def test_separator_payload_tampering_fails_closed(authorized_complete_runtime) -> None:
    point_uri, generators_uri = _simplex(
        authorized_complete_runtime,
        ((1, 2), (1, 2), (1, 2)),
    )
    proposed = authorized_complete_runtime.services.polytope.separate(
        PolytopeSeparateRequest(
            point_uri=point_uri,
            generator_set_uri=generators_uri,
        )
    )
    assert proposed.certificate_uri is not None
    original = authorized_complete_runtime.core.store.get(proposed.certificate_uri)
    tampered = dict(original.payload)
    tampered_payload = dict(tampered["payload"])
    tampered_payload["coefficients"] = [_q(0), _q(0), _q(1)]
    tampered["payload"] = tampered_payload
    tampered["payload_digest"] = (
        "sha256:" + hashlib.sha256(canonicalize_json(tampered_payload)).hexdigest()
    )
    stored = authorized_complete_runtime.core.store.put(
        schema_uri=original.manifest.schema_uri,
        semantics_uri=original.manifest.semantics_uri,
        payload=tampered,
        parents=original.manifest.parents,
        summary="adversarial separator tampering",
    )

    result = authorized_complete_runtime.services.verification.verify_certificate(
        certificate_uri=stored.artifact_uri
    )

    assert result.input.status.value == "REJECTED"
    assert result.conclusion.value == "UNKNOWN"
    assert result.assurance.verification.value == "UNVERIFIED"
    assert result.verification_record_uri is None


def test_projection_is_explicit_and_bound_to_derived_artifacts(
    authorized_complete_runtime,
) -> None:
    point = authorized_complete_runtime.core.artifacts.put(
        schema_uri=authorized_complete_runtime.services.polytope.point_schema_uri,
        semantics_uri=authorized_complete_runtime.services.polytope.semantics_uri,
        payload={
            "point_schema_version": "1",
            "coordinates": [_q(99), _q(1, 2), _q(1, 2), _q(1, 2)],
        },
    )
    generators = authorized_complete_runtime.core.artifacts.put(
        schema_uri=authorized_complete_runtime.services.polytope.generator_set_schema_uri,
        semantics_uri=authorized_complete_runtime.services.polytope.semantics_uri,
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

    proposed = authorized_complete_runtime.services.polytope.separate(
        PolytopeSeparateRequest(
            point_uri=point.artifact_uri,
            generator_set_uri=generators.artifact_uri,
            projection=(1, 2, 3),
        )
    )

    assert proposed.status.value == "SEPARATED"
    assert proposed.effective_point_uri != point.artifact_uri
    assert proposed.effective_generator_set_uri != generators.artifact_uri
    projected_point = authorized_complete_runtime.core.store.get(
        proposed.effective_point_uri or ""
    )
    projected_generators = authorized_complete_runtime.core.store.get(
        proposed.effective_generator_set_uri or ""
    )
    assert projected_point.manifest.parents == (point.artifact_uri,)
    assert projected_generators.manifest.parents == (generators.artifact_uri,)
