from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from tests.support.capabilities import invoke_capability
from tests.support.services import open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.matrices import IntegerMatrix
from jacobian.domains.matrix_lattice import build_matrix_bundle
from jacobian.domains.matrix_lattice.hnf import _parse_hnf_worker_result
from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.operation_installation import OperationInstaller
from jacobian.runtime import CheckerAuthorityMode


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
    bundle = build_matrix_bundle()
    with open_domain_services(
        tmp_path, checker_authority=CheckerAuthorityMode.NONE
    ) as services:
        installed = OperationInstaller(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
        ).install(bundle)
        for adapter in installed.adapters:
            services.installation.register_capability(adapter)
        adapters, _ = install_exact_domain_verification(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
            services.installation.verification,
            services.core.checkers,
            bundles={"matrix": (bundle, installed)},
            authorize=True,
        )
        for adapter in adapters:
            services.installation.register_capability(adapter)
        yield services


def test_hnf_is_domain_owned_and_explicitly_durable() -> None:
    bundle = build_matrix_bundle()
    operation = next(
        operation
        for operation in bundle.capabilities
        if operation.capability_id == "matrix.normal_form.hermite.materialize"
    )
    assert operation.resource_reason
    assert operation.provider_runtime is not None


def test_python_flint_hnf_produces_a_durable_certificate(hnf_services) -> None:
    result = invoke_capability(
        hnf_services,
        "matrix.normal_form.hermite.materialize",
        _matrix([[0, 2, 4], [0, 6, 8]]),
    )
    assert result.execution.status.value == "COMPLETED"
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.output["backend_version"] == "0.9.0"
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
        "backend_version": "0.9.0",
        "flint_library_version": "3.6.0",
        "normal_form": [["1", "0", "0"], ["0", "1", "0"]],
        "transformation": [["1", "0"], ["0", "1"]],
    }
    result = _parse_hnf_worker_result(valid, source)
    assert result.normal_form.entries == (("1", "0", "0"), ("0", "1", "0"))

    for invalid in (
        {key: value for key, value in valid.items() if key != "protocol"},
        {**valid, "backend_version": "0.8.0"},
        {
            **valid,
            "normal_form": [["1", "0"], ["0", "1"]],
        },
    ):
        with pytest.raises((TypeError, ValueError)):
            _parse_hnf_worker_result(invalid, source)


def test_hnf_checker_replays_the_retained_certificate(hnf_services) -> None:
    computed = invoke_capability(
        hnf_services,
        "matrix.normal_form.hermite.materialize",
        _matrix([[1, 2], [3, 4]]),
    )
    verified = hnf_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.hermite.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )
    assert verified.execution.status.value == "COMPLETED"
    assert verified.output["status"] == "VERIFIED"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_legacy_matrices_package_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("jacobian.matrices")
