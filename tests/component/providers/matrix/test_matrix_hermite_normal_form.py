from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.operations import invoke_operation

from jacobian.contracts.matrices import IntegerMatrix
from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.domains.matrix_lattice import matrix_operations
from jacobian.domains.matrix_lattice.hnf import _parse_hnf_worker_result
from jacobian.operation_declarations import DurablePublication


def _matrix(entries: list[list[int]]) -> dict[str, object]:
    return {
        "matrix": {
            "matrix_schema_version": "1",
            "domain": "ZZ",
            "entries": [[str(value) for value in row] for row in entries],
        }
    }


@pytest.fixture
def hnf_services(tmp_path: Path):
    with open_exact_domain_services(
        tmp_path,
        matrix_operations(),
    ) as services:
        yield services


def test_hnf_is_domain_owned_and_explicitly_durable() -> None:
    bundle = matrix_operations()
    operation = next(
        operation
        for operation in bundle
        if operation.operation_id == "matrix.normal_form.hermite.materialize"
    )
    assert isinstance(operation.publication, DurablePublication)
    assert operation.publication.resource_reason
    assert "python-flint" in operation.tags
    assert not hasattr(operation, "provider_binding")


def test_python_flint_hnf_produces_a_durable_certificate(hnf_services) -> None:
    result = invoke_operation(
        hnf_services,
        "matrix.normal_form.hermite.materialize",
        _matrix([[0, 2, 4], [0, 6, 8]]),
    )
    assert result.execution.status.value == "COMPLETED"
    assert result.output["result_uri"].startswith("artifact://sha256/")
    payload = hnf_services.core.store.get(result.output["result_uri"]).payload
    assert payload["normal_form"]["entries"] == [["0", "2", "0"], ["0", "0", "4"]]
    assert payload["transformation"]["entries"] == [["-2", "1"], ["3", "-1"]]


def test_hnf_worker_envelope_requires_identity_and_source_shapes() -> None:
    source = IntegerMatrix.model_validate(
        {
            "entries": [["1", "2", "3"], ["4", "5", "6"]],
        }
    )
    valid = {
        "protocol": "jacobian.matrix-lattice-hnf-worker/v1",
        "status": "NORMAL_FORM_PRODUCED",
        "result": {
            "normal_form": {"entries": [["1", "0", "0"], ["0", "1", "0"]]},
            "transformation": {"entries": [["1", "0"], ["0", "1"]]},
        },
    }
    result = _parse_hnf_worker_result(valid, source)
    assert result.normal_form.entries == (("1", "0", "0"), ("0", "1", "0"))

    for invalid in (
        {key: value for key, value in valid.items() if key != "protocol"},
        {**valid, "status": "ERROR"},
        {
            **valid,
            "result": {
                **valid["result"],
                "normal_form": {"entries": [["1", "0"], ["0", "1"]]},
            },
        },
    ):
        with pytest.raises((TypeError, ValueError)):
            _parse_hnf_worker_result(invalid, source)


def test_hnf_checker_replays_the_retained_certificate(hnf_services) -> None:
    computed = invoke_operation(
        hnf_services,
        "matrix.normal_form.hermite.materialize",
        _matrix([[1, 2], [3, 4]]),
    )
    verified = hnf_services.core.operations.invoke(
        OperationRequest(
            operation_id="matrix.normal_form.hermite.verify",
            input={"result_uri": computed.output["result_uri"]},
        )
    )
    assert verified.execution.status.value == "COMPLETED"
    assert verified.output["status"] == "VERIFIED"
    assert verified.verification_record_uri is not None


def test_legacy_matrices_package_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("jacobian.matrices")
