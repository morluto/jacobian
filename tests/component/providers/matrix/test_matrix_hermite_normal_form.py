from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.support.capabilities import invoke_capability as _invoke
from tests.support.services import open_domain_services

import jacobian.providers.flint_runtime as flint_runtime
from jacobian.bounded_process import BoundedProcessResult
from jacobian.canonical import canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityInstallTier,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.matrices.flint_hnf import install_python_flint_hnf_capability
from jacobian.matrices.normal_form import (
    install_matrix_normal_form_checker,
)
from jacobian.providers.flint_runtime import python_flint_hnf_provider_runtime
from jacobian.runtime import CheckerAuthorityMode
from jacobian.runtime.services import CoreServices


def _matrix(entries: list[list[int | str]]) -> dict[str, Any]:
    return {"entries": [[str(value) for value in row] for row in entries]}


@dataclass(frozen=True, slots=True)
class _HnfRuntime:
    core: CoreServices
    provider_runtime: CapabilityProviderRuntime


@contextmanager
def _open_hnf_runtime(
    root: Path,
    *,
    install_checker: bool,
) -> Iterator[_HnfRuntime]:
    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if install_checker
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        runtime = python_flint_hnf_provider_runtime()
        producer = install_python_flint_hnf_capability(
            services.core.matrix_normal_forms,
            runtime,
        )
        services.installation.register_capability(producer)
        if install_checker:
            adapter, _installation = install_matrix_normal_form_checker(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.core.matrix_normal_forms,
                services.installation.verification,
                services.core.checkers,
                authorize_checker=True,
            )
            assert adapter is not None
            services.installation.register_capability(adapter)
        yield _HnfRuntime(core=services.core, provider_runtime=runtime)


@pytest.fixture
def hnf_services(tmp_path: Path) -> Iterator[_HnfRuntime]:
    with _open_hnf_runtime(tmp_path, install_checker=False) as services:
        yield services


@pytest.fixture
def hnf_checker_services(tmp_path: Path) -> Iterator[_HnfRuntime]:
    with _open_hnf_runtime(tmp_path, install_checker=True) as services:
        yield services


def test_python_flint_produces_bound_rectangular_row_hnf(
    hnf_services: _HnfRuntime,
) -> None:
    runtime = hnf_services
    result = _invoke(
        runtime,
        "matrix.normal_form.hermite",
        {
            "matrix": _matrix([[0, 2, 4], [0, 6, 8]]),
            "resource_budget": {"wall_seconds": 5},
        },
        mode=CapabilityMode.EXPLORE,
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "NORMAL_FORM_PRODUCED"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification"] == "UNVERIFIED"
    assert result.output["flint_library_version"] == "3.6.0"
    assert result.output["normal_form"]["entries"] == [
        ["0", "2", "0"],
        ["0", "0", "4"],
    ]
    assert result.output["transformation"]["entries"] == [
        ["-2", "1"],
        ["3", "-1"],
    ]
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert (
        result.relationships[0].relation_id
        == "matrix.relation.row-hermite-normal-form-of"
    )

    resolved = runtime.core.matrix_normal_forms.resolve_hermite_normal_form(
        result.output["normal_form_uri"]
    )
    assert resolved.candidate.source.matrix_artifact_uri == result.output["matrix_uri"]
    assert result.output["matrix_uri"] in resolved.artifact.manifest.parents


def test_hnf_runtime_has_a_distinct_exact_operation_profile() -> None:
    runtime = python_flint_hnf_provider_runtime()

    assert runtime.availability is CapabilityProviderAvailability.AVAILABLE
    assert runtime.version == "0.9.0"
    assert runtime.install_tier is CapabilityInstallTier.T1
    assert runtime.digest is not None and runtime.digest.startswith("sha256:")
    assert runtime.configuration == {
        "distribution": "python-flint",
        "domain": "ZZ",
        "operation": "fmpz_mat.hnf(transform=True)",
        "flint_library_version": "3.6.0",
        "maximum_rows": 32,
        "maximum_columns": 32,
        "normal_form_convention": "FLINT_ROW_HNF",
        "relation": "H=U*A",
    }


def test_hnf_runtime_rejects_a_different_linked_flint_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = python_flint_hnf_provider_runtime()
    assert available.availability is CapabilityProviderAvailability.AVAILABLE
    monkeypatch.setattr(
        flint_runtime,
        "python_distribution_provider_runtime",
        lambda *_args, **_kwargs: available,
    )
    monkeypatch.setattr(
        flint_runtime.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__FLINT_VERSION__="3.5.0"),
    )

    rejected = python_flint_hnf_provider_runtime(refresh=True)

    assert rejected.availability is CapabilityProviderAvailability.UNAVAILABLE
    assert rejected.version is None
    assert rejected.digest is None
    assert "linked FLINT library" in rejected.diagnostic


@pytest.mark.parametrize(
    "entries",
    [
        [["01"]],
        [["1"], ["2", "3"]],
        [["9" * 257]],
    ],
    ids=("noncanonical_integer", "ragged_rows", "digit_limit"),
)
def test_hnf_rejects_inputs_outside_the_exact_matrix_contract(
    hnf_services: _HnfRuntime,
    entries: list[list[str]],
) -> None:
    runtime = hnf_services

    result = _invoke(
        runtime,
        "matrix.normal_form.hermite",
        {"matrix": {"entries": entries}},
        mode=CapabilityMode.EXPLORE,
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["error"]["stage"] in {
        "capability_input_validation",
        "input_validation",
    }


def test_independent_checker_verifies_full_hnf_relation(
    hnf_checker_services: _HnfRuntime,
) -> None:
    runtime = hnf_checker_services
    computed = _invoke(
        runtime,
        "matrix.normal_form.hermite",
        {"matrix": _matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])},
        mode=CapabilityMode.EXPLORE,
    )
    verified = _invoke(
        runtime,
        "matrix.normal_form.hermite.verify",
        {"normal_form_uri": computed.output["normal_form_uri"]},
        mode=CapabilityMode.VERIFY,
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED_HERMITE_NORMAL_FORM"
    assert verified.output["conclusion"] == "TRUE"
    assert verified.output["verification_record_uri"].startswith("artifact://sha256/")
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


@pytest.mark.parametrize(
    ("source", "normal_form", "transformation"),
    [
        (
            [[2, 4], [6, 8]],
            [[2, 1], [0, 4]],
            [[-2, 1], [3, -1]],
        ),
        (
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[2, 0], [0, 2]],
        ),
        (
            [[0, 1], [1, 0]],
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ),
    ],
    ids=("broken_relation", "nonunimodular_transform", "not_row_hnf"),
)
def test_checker_rejects_each_independent_hnf_obligation(
    hnf_checker_services: _HnfRuntime,
    source: list[list[int]],
    normal_form: list[list[int]],
    transformation: list[list[int]],
) -> None:
    runtime = hnf_checker_services
    matrix_uri = runtime.core.matrix_normal_forms.put_matrix(
        _matrix(source)
    ).artifact_uri
    candidate = runtime.core.matrix_normal_forms.put_hermite_normal_form(
        matrix_uri=matrix_uri,
        normal_form=[[str(value) for value in row] for row in normal_form],
        transformation=[[str(value) for value in row] for row in transformation],
        producer=runtime.provider_runtime,
        resource_budget={"wall_seconds": 5},
    )

    rejected = _invoke(
        runtime,
        "matrix.normal_form.hermite.verify",
        {"normal_form_uri": candidate.artifact_uri},
        mode=CapabilityMode.VERIFY,
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is not CapabilityAssuranceLevel.VERIFIED


def test_python_flint_hnf_timeout_is_operational(
    hnf_services: _HnfRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = hnf_services
    monkeypatch.setattr(
        "jacobian.matrices.flint_hnf.run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )

    result = _invoke(
        runtime,
        "matrix.normal_form.hermite",
        {"matrix": _matrix([[1]])},
        mode=CapabilityMode.EXPLORE,
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output["status"] == "NO_NORMAL_FORM_PRODUCED"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["normal_form_uri"] is None
    assert result.assurance.level is not CapabilityAssuranceLevel.VERIFIED


def test_hnf_worker_gets_only_fixed_environment_and_budget(
    hnf_services: _HnfRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = hnf_services
    monkeypatch.setenv("JACOBIAN_HNF_SECRET", "must-not-propagate")
    observed: dict[str, Any] = {}

    def fake_worker(*_args: Any, **kwargs: Any) -> BoundedProcessResult:
        observed.update(kwargs)
        stdout = (
            canonicalize_json(
                {
                    "protocol": "jacobian.flint-hnf-worker/v1",
                    "status": "NORMAL_FORM_PRODUCED",
                    "backend_version": "0.9.0",
                    "flint_library_version": "3.6.0",
                    "normal_form": [["1"]],
                    "transformation": [["1"]],
                }
            )
            + b"\n"
        )
        return BoundedProcessResult(
            returncode=0,
            stdout=stdout,
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        )

    monkeypatch.setattr("jacobian.matrices.flint_hnf.run_bounded_process", fake_worker)
    result = _invoke(
        runtime,
        "matrix.normal_form.hermite",
        {
            "matrix": _matrix([[1]]),
            "resource_budget": {"wall_seconds": 7},
        },
        mode=CapabilityMode.EXPLORE,
    )

    assert result.output["status"] == "NORMAL_FORM_PRODUCED"
    assert observed["timeout_seconds"] == 7.0
    assert observed["environment"] == {
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "PYTHONHASHSEED": "0",
    }
    assert "JACOBIAN_HNF_SECRET" in os.environ
    assert "JACOBIAN_HNF_SECRET" not in observed["environment"]


def test_hnf_output_is_discarded_if_runtime_identity_changes(
    hnf_services: _HnfRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = hnf_services
    original = runtime.provider_runtime
    changed = original.model_copy(update={"digest": "sha256:" + "f" * 64})
    observations = iter((original, changed))
    monkeypatch.setattr(
        "jacobian.matrices.flint_hnf.python_flint_hnf_provider_runtime",
        lambda **_kwargs: next(observations),
    )
    monkeypatch.setattr(
        "jacobian.matrices.flint_hnf.run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=0,
            stdout=canonicalize_json(
                {
                    "protocol": "jacobian.flint-hnf-worker/v1",
                    "status": "NORMAL_FORM_PRODUCED",
                    "backend_version": "0.9.0",
                    "flint_library_version": "3.6.0",
                    "normal_form": [["1"]],
                    "transformation": [["1"]],
                }
            )
            + b"\n",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        ),
    )

    result = _invoke(
        runtime,
        "matrix.normal_form.hermite",
        {"matrix": _matrix([[1]])},
        mode=CapabilityMode.EXPLORE,
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["status"] == "NO_NORMAL_FORM_PRODUCED"
    assert result.output["normal_form_uri"] is None
    assert "changed during execution" in result.output["detail"]


def test_invalid_worker_protocol_retains_no_hnf_evidence(
    hnf_services: _HnfRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = hnf_services
    monkeypatch.setattr(
        "jacobian.matrices.flint_hnf.run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=0,
            stdout=b'{"status":"NORMAL_FORM_PRODUCED"}\n',
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        ),
    )

    result = _invoke(
        runtime,
        "matrix.normal_form.hermite",
        {"matrix": _matrix([[1]])},
        mode=CapabilityMode.EXPLORE,
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["status"] == "NO_NORMAL_FORM_PRODUCED"
    assert result.output["normal_form_uri"] is None
    assert result.output["conclusion"] == "UNKNOWN"


def test_hnf_checker_timeout_is_operational(
    hnf_checker_services: _HnfRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = hnf_checker_services
    computed = _invoke(
        runtime,
        "matrix.normal_form.hermite",
        {"matrix": _matrix([[1]])},
        mode=CapabilityMode.EXPLORE,
    )
    monkeypatch.setattr(
        "jacobian.verification.service.run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )

    result = _invoke(
        runtime,
        "matrix.normal_form.hermite.verify",
        {"normal_form_uri": computed.output["normal_form_uri"]},
        mode=CapabilityMode.VERIFY,
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output["status"] == "TIMEOUT"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification_record_uri"] is None
    assert result.assurance.level is not CapabilityAssuranceLevel.VERIFIED
