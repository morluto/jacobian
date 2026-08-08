from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.implementation import checker_source_digest, package_source_digest
from jacobian.plugin_execution import _plugin_failure_detail
from jacobian.verification._helpers import _checker_failure_detail


@pytest.mark.parametrize(
    ("module", "entrypoint", "public_detail"),
    [
        (
            "jacobian.plugin_worker",
            "tests.support.process_entrypoints:echo",
            "The plugin changed after it was registered. Reload Jacobian to "
            "register the current plugin version, then retry.",
        ),
        (
            "jacobian.checker_worker",
            "tests.component.checkers._fixture_checkers:check_fixture_value",
            "The checker changed after authorization. Authorize the current "
            "checker version, then retry.",
        ),
    ],
)
def test_source_changes_cross_worker_boundary_as_typed_codes(
    module: str,
    entrypoint: str,
    public_detail: str,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            entrypoint,
            "sha256:" + "0" * 64,
        ],
        input=b"{}",
        capture_output=True,
        check=False,
        timeout=10,
    )
    response = loads_strict_json(completed.stdout)

    assert completed.returncode == 1
    assert response["error_code"] == "SOURCE_CHANGED"
    if module == "jacobian.plugin_worker":
        assert _plugin_failure_detail(response) == public_detail
    else:
        assert _checker_failure_detail(response) == public_detail


@pytest.mark.parametrize(
    ("module", "entrypoint"),
    [
        (
            "jacobian.plugin_worker",
            "tests.support.process_entrypoints:imitate_source_change",
        ),
        (
            "jacobian.checker_worker",
            "tests.component.checkers._fixture_checkers:imitate_source_change",
        ),
    ],
)
def test_worker_code_cannot_self_report_a_source_change(
    module: str,
    entrypoint: str,
) -> None:
    source_digest = (
        checker_source_digest(entrypoint)
        if module == "jacobian.checker_worker"
        else package_source_digest(entrypoint)
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            entrypoint,
            source_digest,
        ],
        input=b"{}",
        capture_output=True,
        check=False,
        timeout=10,
    )
    response = loads_strict_json(completed.stdout)

    assert completed.returncode == 1
    assert response["error_code"] == "EXECUTION_FAILED"


def test_checker_worker_classifies_malformed_provider_runtime() -> None:
    entrypoint = "tests.component.checkers._fixture_checkers:check_fixture_value"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jacobian.checker_worker",
            entrypoint,
            checker_source_digest(entrypoint),
            "{malformed",
        ],
        input=b"{}",
        capture_output=True,
        check=False,
        timeout=10,
    )
    response = loads_strict_json(completed.stdout)

    assert completed.returncode == 1
    assert response == {"error_code": "MALFORMED_RUNTIME"}


def test_rational_linear_worker_classifies_non_string_protocol_as_invalid_input() -> (
    None
):
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "jacobian.domains.rational_linear.worker",
        ],
        input=canonicalize_json({"protocol": [], "system": {}}),
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 2
    assert completed.stderr == b""
    assert loads_strict_json(completed.stdout) == {
        "protocol": "jacobian.rational-linear-solution-worker/v1",
        "status": "ERROR",
        "error": "TypeError",
    }


def _lean_runtime_encoded(
    lean: Path,
    lake: Path | None,
    lake_digest: str | None,
) -> str:
    lean_digest = "sha256:" + hashlib.sha256(lean.read_bytes()).hexdigest()
    configuration: dict[str, object] = {"executable": str(lean)}
    if lake is not None:
        configuration["lake_executable"] = str(lake)
        configuration["lake_digest"] = lake_digest
    runtime = CapabilityProviderRuntime(
        provider="jacobian.lean4",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="4.31.0",
        digest=lean_digest,
        digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
        platform="linux_x86_64",
        install_tier=CapabilityInstallTier.T3,
        license_id="Apache-2.0",
        configuration=configuration,
    )
    return canonicalize_json(runtime.model_dump(mode="json")).decode("utf-8")


def test_measure_runtime_binds_the_lake_launcher_into_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.checker_worker import _measure_runtime

    lean = tmp_path / "lean"
    lean.write_bytes(b"pinned-lean")
    lake = tmp_path / "lake"
    lake.write_bytes(b"pinned-lake")
    lake_digest = "sha256:" + hashlib.sha256(lake.read_bytes()).hexdigest()
    encoded = _lean_runtime_encoded(lean, lake, lake_digest)
    for name in (
        "JACOBIAN_CHECKER_EXECUTABLE",
        "JACOBIAN_CHECKER_RUNTIME_DIGEST",
        "JACOBIAN_CHECKER_LAKE_DIGEST",
    ):
        monkeypatch.delenv(name, raising=False)

    _measure_runtime(encoded)

    assert os.environ["JACOBIAN_CHECKER_EXECUTABLE"] == str(lean)
    assert os.environ["JACOBIAN_CHECKER_LAKE_DIGEST"] == lake_digest


def test_measure_runtime_rejects_a_swapped_lake_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.checker_worker import _measure_runtime

    lean = tmp_path / "lean"
    lean.write_bytes(b"pinned-lean")
    lake = tmp_path / "lake"
    lake.write_bytes(b"pinned-lake")
    encoded = _lean_runtime_encoded(
        lean,
        lake,
        "sha256:" + hashlib.sha256(b"pinned-lake").hexdigest(),
    )
    # Swap the on-disk launcher after the runtime identity was measured.
    lake.write_bytes(b"replaced-lake")
    for name in (
        "JACOBIAN_CHECKER_EXECUTABLE",
        "JACOBIAN_CHECKER_RUNTIME_DIGEST",
        "JACOBIAN_CHECKER_LAKE_DIGEST",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="lake launcher digest changed"):
        _measure_runtime(encoded)

    # A rejected binding must not leak a stale digest into the environment.
    assert "JACOBIAN_CHECKER_LAKE_DIGEST" not in os.environ


def test_measure_runtime_clears_a_stale_lake_binding_when_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.checker_worker import _measure_runtime

    lean = tmp_path / "lean"
    lean.write_bytes(b"pinned-lean")
    monkeypatch.setenv("JACOBIAN_CHECKER_LAKE_DIGEST", "sha256:" + "0" * 64)
    encoded = _lean_runtime_encoded(lean, None, None)

    _measure_runtime(encoded)

    assert "JACOBIAN_CHECKER_LAKE_DIGEST" not in os.environ
