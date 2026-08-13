from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tests.support.nullstellensatz import (
    load_chart_certificates,
    open_nullstellensatz_services,
)
from tests.support.services import DomainTestServices

from jacobian.capability_errors import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.nullstellensatz import NullstellensatzCertificateBundle
from jacobian.contracts.results import (
    Conclusion,
    Execution,
    ExecutionStatus,
    InputStatus,
    InputValidation,
    VerificationResult,
)
from jacobian.domains.polynomial_nullstellensatz.core import (
    MATERIALIZE_CAPABILITY_ID,
    VERIFY_CAPABILITY_ID,
    _failure_details,
)
from jacobian.runtime.config import CheckerAuthorityMode


def test_failure_details_are_bounded_and_summarize_validation_errors() -> None:
    long_details = _failure_details(ValueError("x" * 2048))
    assert long_details["exception_type"] == "ValueError"
    assert len(long_details["reason"]) == 512

    with pytest.raises(ValidationError) as caught:
        NullstellensatzCertificateBundle.model_validate({})
    validation_details = _failure_details(caught.value)
    assert validation_details["exception_type"] == "ValidationError"
    assert int(validation_details["validation_error_count"]) > 0
    assert validation_details["reason"] == (
        "validation_error: "
        f"{validation_details['validation_error_count']} invalid field(s)"
    )


def test_invalid_request_uri_values_are_summarized_without_echoing(
    tmp_path: Path,
) -> None:
    oversized_uri = "secret" * 200_000
    with open_nullstellensatz_services(
        tmp_path,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        adapter = services.core.capabilities._adapters[VERIFY_CAPABILITY_ID]
        with pytest.raises(CapabilityInvocationError) as caught:
            adapter.prepare(
                CapabilityRequest(
                    capability_id=VERIFY_CAPABILITY_ID,
                    input={
                        "system_uri": oversized_uri,
                        "certificate_bundle_uri": oversized_uri,
                    },
                )
            )

    diagnostic = caught.value.diagnostic
    assert diagnostic.details["system_uri"] == (f"string(length={len(oversized_uri)})")
    assert diagnostic.details["certificate_bundle_uri"] == (
        f"string(length={len(oversized_uri)})"
    )
    assert "secret" not in str(diagnostic.details)


def _invoke(
    services: DomainTestServices,
    capability_id: str,
    payload: dict[str, Any],
) -> CapabilityResult:
    return services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=capability_id,
            input=payload,
        )
    )


def _persist_certificate(
    services: DomainTestServices,
    materialized: CapabilityResult,
    *,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    system_uri = materialized.output["system_uri"]
    system_artifact = services.core.store.get(system_uri)
    verify_descriptor = services.core.capabilities._adapters[
        VERIFY_CAPABILITY_ID
    ].descriptor
    bundle = NullstellensatzCertificateBundle(
        system_uri=system_uri,
        system_digest=system_artifact.manifest.object_digest,
        producer_version="4.4.1p5",
        producer_digest="sha256:" + "4" * 64,
        charts=load_chart_certificates(),
    )
    payload = bundle.model_dump(mode="json")
    if mutate is not None:
        mutate(payload)
    stored = services.core.artifacts.put(
        schema_uri=verify_descriptor.accepted_artifact_types[1],
        semantics_uri=system_artifact.manifest.semantics_uri,
        payload=payload,
        parents=(system_uri,),
        summary="test Nullstellensatz certificate bundle",
        producer_write=True,
    )
    return str(stored.artifact_uri)


def test_authorized_checker_verifies_complete_bundle(tmp_path: Path) -> None:
    with open_nullstellensatz_services(
        tmp_path,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        materialized = _invoke(
            services,
            MATERIALIZE_CAPABILITY_ID,
            {},
        )
        certificate_uri = _persist_certificate(services, materialized)

        result = _invoke(
            services,
            VERIFY_CAPABILITY_ID,
            {
                "system_uri": materialized.output["system_uri"],
                "certificate_bundle_uri": certificate_uri,
            },
        )

        assert result.output["claim"] == "SYSTEM_INFEASIBLE"
        assert result.output["conclusion"] == "TRUE"
        assert result.output["verification_record_uri"] in result.artifact_uris
        assert result.verification_record_uri is not None


def test_unavailable_checker_never_false_certifies(tmp_path: Path) -> None:
    with open_nullstellensatz_services(
        tmp_path,
        checker_authority=CheckerAuthorityMode.NONE,
    ) as services:
        materialized = _invoke(
            services,
            MATERIALIZE_CAPABILITY_ID,
            {},
        )
        certificate_uri = _persist_certificate(services, materialized)

        result = _invoke(
            services,
            VERIFY_CAPABILITY_ID,
            {
                "system_uri": materialized.output["system_uri"],
                "certificate_bundle_uri": certificate_uri,
            },
        )

        assert result.output["conclusion"] == "UNKNOWN"
        assert result.output["verification_record_uri"] is None


def test_mutated_certificate_cannot_return_verified(tmp_path: Path) -> None:
    with open_nullstellensatz_services(
        tmp_path,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        materialized = _invoke(
            services,
            MATERIALIZE_CAPABILITY_ID,
            {},
        )

        def remove_term(payload: dict[str, Any]) -> None:
            multipliers = payload["charts"][0]["multipliers"]
            next(
                item["multiplier"]["terms"]
                for item in multipliers
                if len(item["multiplier"]["terms"]) >= 2
            ).pop()

        certificate_uri = _persist_certificate(
            services,
            materialized,
            mutate=remove_term,
        )
        result = _invoke(
            services,
            VERIFY_CAPABILITY_ID,
            {
                "system_uri": materialized.output["system_uri"],
                "certificate_bundle_uri": certificate_uri,
            },
        )

        assert result.output["conclusion"] == "UNKNOWN"
        assert result.output["verification_record_uri"] is None


def test_stale_artifact_binding_is_rejected_before_checker(tmp_path: Path) -> None:
    with open_nullstellensatz_services(
        tmp_path,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        materialized = _invoke(
            services,
            MATERIALIZE_CAPABILITY_ID,
            {},
        )
        certificate_uri = _persist_certificate(services, materialized)
        wrong_schema = services.core.capabilities._adapters[
            VERIFY_CAPABILITY_ID
        ].descriptor.accepted_artifact_types[0]
        reordered_payload = dict(
            services.core.store.get(materialized.output["system_uri"]).payload
        )
        reordered_payload["charts"] = list(reversed(reordered_payload["charts"]))
        wrong_system = services.core.artifacts.put(
            schema_uri=wrong_schema,
            semantics_uri=services.core.store.get(
                materialized.output["system_uri"]
            ).manifest.semantics_uri,
            payload=reordered_payload,
            summary="reordered system with no certificate parent binding",
            producer_write=True,
        )

        result = _invoke(
            services,
            VERIFY_CAPABILITY_ID,
            {
                "system_uri": wrong_system.artifact_uri,
                "certificate_bundle_uri": certificate_uri,
            },
        )

        assert result.execution.status.value == "ERROR"
        assert result.diagnostics[0].code == (
            "INVALID_NULLSTELLENSATZ_VERIFICATION_REQUEST"
        )
        assert result.diagnostics[0].actual_type is None
        assert result.diagnostics[0].details == {
            "exception_type": "ValueError",
            "reason": "system artifact differs from the frozen degree slice",
            "system_uri": wrong_system.artifact_uri,
            "certificate_bundle_uri": certificate_uri,
        }
        assert "cannot establish infeasibility" in (result.diagnostics[0].hint or "")
        assert "stop" not in (result.diagnostics[0].hint or "")


def test_wrong_bundle_schema_reports_actionable_artifact_diagnostics(
    tmp_path: Path,
) -> None:
    with open_nullstellensatz_services(
        tmp_path,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        materialized = _invoke(
            services,
            MATERIALIZE_CAPABILITY_ID,
            {},
        )
        system_uri = materialized.output["system_uri"]

        result = _invoke(
            services,
            VERIFY_CAPABILITY_ID,
            {
                "system_uri": system_uri,
                "certificate_bundle_uri": system_uri,
            },
        )

        diagnostic = result.diagnostics[0]
        assert result.execution.status is ExecutionStatus.ERROR
        assert diagnostic.code == "INVALID_NULLSTELLENSATZ_VERIFICATION_REQUEST"
        assert diagnostic.actual_type == (
            services.core.store.get(system_uri).manifest.schema_uri
        )
        assert diagnostic.details == {
            "exception_type": "ValueError",
            "reason": "certificate_bundle_uri has the wrong schema",
            "system_uri": system_uri,
            "certificate_bundle_uri": system_uri,
        }
        assert diagnostic.expected is not None
        assert (
            services.core.capabilities._adapters[
                VERIFY_CAPABILITY_ID
            ].descriptor.accepted_artifact_types[1]
            in diagnostic.expected
        )


def test_checker_timeout_never_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with open_nullstellensatz_services(
        tmp_path,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        materialized = _invoke(
            services,
            MATERIALIZE_CAPABILITY_ID,
            {},
        )
        certificate_uri = _persist_certificate(services, materialized)
        monkeypatch.setattr(
            services.application.verification,
            "verify_certificate",
            lambda **_kwargs: VerificationResult(
                execution=Execution(status=ExecutionStatus.TIMEOUT),
                input=InputValidation(status=InputStatus.ACCEPTED),
                conclusion=Conclusion.UNKNOWN,
            ),
        )

        result = _invoke(
            services,
            VERIFY_CAPABILITY_ID,
            {
                "system_uri": materialized.output["system_uri"],
                "certificate_bundle_uri": certificate_uri,
            },
        )

        assert result.execution.status is ExecutionStatus.TIMEOUT
        assert result.output == {}
        assert result.verification_record_uri is None
