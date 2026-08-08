from __future__ import annotations

from typing import Any

import pytest

from jacobian.contracts.capabilities import CapabilityMode, CapabilityRequest
from jacobian.contracts.exact_domain_verification import InlineExactVerificationRecord
from jacobian.runtime.model import JacobianRuntime


def _matrix() -> dict[str, object]:
    def rational(value: int) -> dict[str, str]:
        return {"num": str(value), "den": "1"}

    return {
        "matrix_schema_version": "1",
        "domain": "QQ",
        "entries": [[rational(1), rational(2)], [rational(2), rational(4)]],
    }


def test_inline_exact_replay_persists_only_its_bound_record(
    authorized_complete_runtime: JacobianRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordinary values must not be materialized merely for checker replay."""

    runtime = authorized_complete_runtime
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
            mode=CapabilityMode.VERIFY,
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
    assert verified.scope is not None
    assert parsed.bindings.claim_digest == verified.scope.parameters["claim_digest"]
    assert (
        parsed.bindings.candidate_digest
        == verified.scope.parameters["candidate_digest"]
    )
    assert (
        parsed.bindings.semantics_digest
        == verified.scope.parameters["semantics_digest"]
    )
    assert parsed.operation_id == verified.scope.parameters["operation_id"]
    assert parsed.checker_id == verified.scope.parameters["checker_id"]
    assert parsed.witness_format == verified.scope.parameters["witness_format"]
