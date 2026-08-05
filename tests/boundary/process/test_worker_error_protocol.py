from __future__ import annotations

import subprocess
import sys

import pytest

from jacobian.canonical import loads_strict_json
from jacobian.implementation import checker_source_digest, package_source_digest
from jacobian.plugin_execution import _plugin_failure_detail
from jacobian.verification._helpers import _checker_failure_detail


@pytest.mark.parametrize(
    ("module", "entrypoint", "public_detail"),
    [
        (
            "jacobian.plugin_worker",
            "tests.support.plugin_entrypoints:echo",
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
            "tests.support.plugin_entrypoints:imitate_source_change",
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
