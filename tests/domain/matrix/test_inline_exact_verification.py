from __future__ import annotations

from typing import Any

import pytest
from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.checkers import CheckerDecision
from jacobian.contracts.exact_domain_verification import InlineExactVerificationRecord
from jacobian.contracts.results import Arithmetic, Conclusion, Coverage, Method


def _matrix() -> dict[str, object]:
    def rational(value: int) -> dict[str, str]:
        return {"num": str(value), "den": "1"}

    return {
        "matrix_schema_version": "1",
        "domain": "QQ",
        "entries": [[rational(1), rational(2)], [rational(2), rational(4)]],
    }


def test_inline_exact_replay_persists_only_its_bound_record(
    matrix_services: DomainTestServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordinary values must not be materialized merely for checker replay."""

    runtime = matrix_services
    computed = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.rank.compute", input={"matrix": _matrix()}
        )
    )

    def unexpected_artifact_put(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("inline exact replay must not materialize an artifact")

    monkeypatch.setattr(runtime.core.artifacts, "put", unexpected_artifact_put)
    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.rank.verify",
            input={
                "input": {"matrix": _matrix()},
                "candidate": computed.output["result"],
            },
        )
    )

    assert verified.output["status"] == "VERIFIED"
    assert verified.output["input_uri"] is None
    assert verified.output["result_uri"] is None
    assert verified.output["witness_uri"] is None
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert len(verified.artifact_uris) == 2
    record = runtime.core.store.get(verified.output["verification_record_uri"])
    parsed = InlineExactVerificationRecord.model_validate(record.payload)
    assert record.manifest.parents == (parsed.semantics_uri,)
    assert parsed.semantics_uri in verified.artifact_uris
    assert parsed.decision.accepted is True


def test_inline_exact_validation_does_not_echo_a_rejected_candidate(
    matrix_services: DomainTestServices,
) -> None:
    marker = "private_inline_candidate_marker"
    checked = matrix_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.rank.verify",
            input={
                "input": {"matrix": _matrix()},
                "candidate": {"rank": marker, "pivot_columns": []},
            },
        )
    )

    assert checked.execution.status == "ERROR"
    assert checked.diagnostics[0].code == "INVALID_REQUEST"
    assert marker not in checked.model_dump_json()


def test_inline_exact_rejects_bounded_accepted_checker_decisions(
    matrix_services: DomainTestServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inline values have no scope artifact for bounded verification."""

    runtime = matrix_services

    def accept_with_bounded_coverage(**_kwargs: object) -> CheckerDecision:
        return CheckerDecision(
            accepted=True,
            conclusion=Conclusion.TRUE,
            arithmetic=Arithmetic.EXACT_RATIONAL,
            method=Method.BOUNDED_SEARCH,
            coverage=Coverage.BOUNDED,
        )

    monkeypatch.setattr(
        runtime.application.verification._checker_executor,
        "execute",
        accept_with_bounded_coverage,
    )

    computed = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.rank.compute", input={"matrix": _matrix()}
        )
    )
    checked = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.rank.verify",
            input={
                "input": {"matrix": _matrix()},
                "candidate": computed.output["result"],
            },
        )
    )

    assert checked.execution.status == "COMPLETED"
    assert checked.output["status"] == "REJECTED"
    assert checked.output["verification_record_uri"] is None
    assert "cannot bind a bounded scope" in checked.output["detail"]
