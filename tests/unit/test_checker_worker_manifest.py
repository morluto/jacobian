from __future__ import annotations

import sys
from pathlib import Path

import pytest

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.checker_identity import build_checker_manifest
from jacobian.process_policy import (
    ProcessRequest,
    ProcessTermination,
    execute_process,
)
from jacobian.worker_environment import worker_environment


def _install_decision_checker(tmp_path: Path) -> str:
    package = tmp_path / "decision_checker_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "checker.py").write_text(
        "def run(request):\n"
        "    accepted = request.get('accept') is True\n"
        "    return {\n"
        "        'accepted': accepted,\n"
        "        'conclusion': 'TRUE' if accepted else 'UNKNOWN',\n"
        "        'arithmetic': 'EXACT_INTEGER',\n"
        "        'method': 'DIRECT_WITNESS',\n"
        "        'coverage': 'NOT_APPLICABLE',\n"
        "        'detail': 'fixture decision',\n"
        "    }\n",
        encoding="utf-8",
    )
    return "decision_checker_fixture.checker:run"


def _run_checker_worker(
    manifest_json: str, tmp_path: Path, request: object | None = None
):
    return execute_process(
        ProcessRequest(
            executable=sys.executable,
            arguments=("-m", "jacobian.checker_worker", manifest_json),
            environment=worker_environment(
                extra_variables=("COVERAGE_FILE", "COVERAGE_PROCESS_CONFIG"),
                overrides={"PYTHONPATH": str(tmp_path)},
            ),
            cwd=str(tmp_path),
            timeout_seconds=10.0,
            stdin_bytes=canonicalize_json({} if request is None else request),
            stdout_limit_bytes=16_384,
            stderr_limit_bytes=16_384,
        ),
    )


@pytest.mark.parametrize("accepted", [True, False])
def test_worker_returns_manifest_bound_checker_decisions(
    tmp_path: Path, accepted: bool
) -> None:
    entrypoint = _install_decision_checker(tmp_path)
    sys.path.insert(0, str(tmp_path))
    try:
        manifest = build_checker_manifest(
            entrypoint,
            provider_runtime=None,
            passive_contract_uris=(),
        )
        completed = _run_checker_worker(
            canonicalize_json(manifest.model_dump(mode="json")).decode("utf-8"),
            tmp_path,
            {"accept": accepted},
        )
    finally:
        sys.path.remove(str(tmp_path))

    assert completed.termination is ProcessTermination.EXITED
    assert completed.returncode == 0
    response = loads_strict_json(completed.stdout)
    assert response["decision"]["accepted"] is accepted
    assert response["decision"]["conclusion"] == ("TRUE" if accepted else "UNKNOWN")


def test_worker_rejects_a_malformed_manifest(tmp_path: Path) -> None:
    completed = _run_checker_worker("{}", tmp_path)

    assert completed.termination is ProcessTermination.EXITED
    assert completed.returncode == 1
    assert loads_strict_json(completed.stdout) == {"error_code": "MALFORMED_RUNTIME"}


def test_worker_rejects_dynamic_first_party_imports(tmp_path: Path) -> None:
    package = tmp_path / "dynamic_checker_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "checker.py").write_text(
        "import importlib\n\n"
        "def run(_request):\n"
        "    return importlib.import_module('dynamic_checker_fixture.helper').run()\n",
        encoding="utf-8",
    )
    (package / "helper.py").write_text(
        "def run():\n    return {}\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        manifest = build_checker_manifest(
            "dynamic_checker_fixture.checker:run",
            provider_runtime=None,
            passive_contract_uris=(),
        )
        completed = _run_checker_worker(
            canonicalize_json(manifest.model_dump(mode="json")).decode("utf-8"),
            tmp_path,
        )
    finally:
        sys.path.remove(str(tmp_path))

    assert completed.termination is ProcessTermination.EXITED
    assert completed.returncode == 1
    assert loads_strict_json(completed.stdout) == {"error_code": "UNDECLARED_IMPORT"}


def test_worker_rejects_dynamic_third_party_imports(tmp_path: Path) -> None:
    package = tmp_path / "dynamic_dependency_checker"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "checker.py").write_text(
        "def run(_request):\n"
        "    module = __builtins__['__import__']('sympy')\n"
        "    return {'accepted': bool(module)}\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        manifest = build_checker_manifest(
            "dynamic_dependency_checker.checker:run",
            provider_runtime=None,
            passive_contract_uris=(),
        )
        assert all(
            item.distribution.lower() != "sympy"
            for item in manifest.python_distributions
        )
        completed = _run_checker_worker(
            canonicalize_json(manifest.model_dump(mode="json")).decode("utf-8"),
            tmp_path,
        )
    finally:
        sys.path.remove(str(tmp_path))

    assert completed.termination is ProcessTermination.EXITED
    assert completed.returncode == 1
    assert loads_strict_json(completed.stdout) == {"error_code": "UNDECLARED_IMPORT"}
