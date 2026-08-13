from __future__ import annotations

from typing import Any

import pytest
from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.checkers import CheckerDecision
from jacobian.contracts.exact_domain_verification import InlineExactVerificationRecord
from jacobian.contracts.matrix_operations import MatrixRankResult
from jacobian.contracts.results import Arithmetic, Conclusion, Coverage, Method
from jacobian.math.matrices.values import SmithNormalForm


def _matrix() -> dict[str, object]:
    def rational(value: int) -> dict[str, str]:
        return {"num": str(value), "den": "1"}

    return {
        "matrix_schema_version": "1",
        "domain": "QQ",
        "entries": [[rational(1), rational(2)], [rational(2), rational(4)]],
    }


def _integer_matrix() -> dict[str, object]:
    return {"entries": [["2", "4"], ["6", "8"]]}


def _wrong_smith_result() -> SmithNormalForm:
    return SmithNormalForm.model_validate(
        {
            "normal_form": {"entries": [["1", "0"], ["0", "8"]]},
            "rank": 2,
            "invariant_factors": ["1", "8"],
        }
    )


def test_smith_checker_consumes_the_producers_typed_candidate_reference(
    matrix_services: DomainTestServices,
) -> None:
    runtime = matrix_services
    computed = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.compute",
            input={"matrix": _integer_matrix()},
        )
    )

    value_ref = computed.output["value_refs"]["smith_form"]
    descriptors = {
        item.capability_id: item
        for item in runtime.core.capabilities.catalog().capabilities
    }
    assert [
        port.model_dump(mode="json")
        for port in descriptors["matrix.normal_form.smith.compute"].output_ports
    ] == [{"name": "smith_form", "value_type": "SmithNormalForm"}]
    assert [
        port.model_dump(mode="json")
        for port in descriptors["matrix.normal_form.smith.verify"].input_ports
    ] == [{"name": "candidate", "value_type": "SmithNormalForm"}]

    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.verify",
            input={"input": {"matrix": _integer_matrix()}},
            inputs={"candidate": value_ref},
        )
    )

    assert verified.output["status"] == "VERIFIED"
    assert verified.output["conclusion"] == "TRUE"
    assert verified.verification_record_uri is not None


def test_candidate_reference_does_not_transfer_producer_authority(
    matrix_services: DomainTestServices,
) -> None:
    runtime = matrix_services
    value_ref = runtime.core.values.put(
        _wrong_smith_result(),
        operation_id="untrusted.example.compute",
        operation_version="1",
        output_port="candidate",
    )

    rejected = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.verify",
            input={"input": {"matrix": _integer_matrix()}},
            inputs={"candidate": value_ref},
        )
    )

    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.verification_record_uri is None


def test_smith_checker_rejects_a_reference_with_the_wrong_value_type(
    matrix_services: DomainTestServices,
) -> None:
    runtime = matrix_services
    value_ref = runtime.core.values.put(
        MatrixRankResult(rank=2, pivot_columns=(0, 1)),
        operation_id="matrix.compute.rank",
        operation_version="1",
        output_port="rank",
    )

    failed = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.verify",
            input={"input": {"matrix": _integer_matrix()}},
            inputs={"candidate": value_ref},
        )
    )

    assert failed.execution.status == "ERROR"
    assert failed.diagnostics[0].code == "INVALID_EXACT_DOMAIN_INPUT"
    assert failed.verification_record_uri is None


@pytest.mark.parametrize(
    ("input_payload", "inputs"),
    (
        (
            {
                "input": {"matrix": _integer_matrix()},
                "candidate": _wrong_smith_result().model_dump(mode="json"),
            },
            {"candidate": "value://AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},
        ),
        (
            {"input": {"matrix": _integer_matrix()}},
            {"unknown": "value://AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},
        ),
        (
            {"input": {"matrix": _integer_matrix()}},
            {"candidate": "value://AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},
        ),
    ),
)
def test_smith_candidate_reference_binding_fails_closed(
    matrix_services: DomainTestServices,
    input_payload: dict[str, object],
    inputs: dict[str, str],
) -> None:
    if "candidate" in input_payload and "candidate" in inputs:
        inputs = {
            "candidate": matrix_services.core.values.put(
                _wrong_smith_result(),
                operation_id="untrusted.example.compute",
                operation_version="1",
                output_port="candidate",
            )
        }
    failed = matrix_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.verify",
            input=input_payload,
            inputs=inputs,
        )
    )

    assert failed.execution.status == "ERROR"
    assert failed.diagnostics[0].code == "INVALID_EXACT_DOMAIN_INPUT"
    assert failed.verification_record_uri is None


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
    registration = runtime.core.checkers.require_active(parsed.checker_id)
    assert parsed.record_schema_version == "4"
    assert parsed.checker_manifest == registration.implementation
    assert (
        parsed.implementation_digest == parsed.checker_manifest.implementation_digest()
    )


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
    assert checked.diagnostics[0].code == "INVALID_EXACT_DOMAIN_INPUT"
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
