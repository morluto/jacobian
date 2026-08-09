from __future__ import annotations

import hashlib
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

import jacobian_checkers.lean4 as lean4_checker
from jacobian.process_policy import ProcessResult, ProcessTermination
from jacobian_checkers.lean4 import (
    LEAN_COMMIT,
    LEAN_VERSION,
    check_kernel_certificate,
)


def _request() -> dict[str, Any]:
    bindings = {
        "claim_digest": "sha256:" + "a" * 64,
        "semantics_digest": "sha256:" + "b" * 64,
        "candidate_digest": "sha256:" + "c" * 64,
        "scope_digest": None,
        "encoding_digest": None,
    }
    statement = "True"
    proof = "by trivial"
    return {
        "request_version": "1",
        "claim": {
            "payload": {
                "statement": statement,
                "environment": "CORE",
                "allowed_axioms": [],
            }
        },
        "candidate": {
            "payload": {
                "statement": statement,
                "proof": proof,
                "environment": "CORE",
            }
        },
        "certificate": {
            "payload": {
                "certificate_type": "lean4.kernel",
                "format_version": "1",
                "bindings": deepcopy(bindings),
                "payload": {
                    "statement": statement,
                    "proof": proof,
                    "environment": "CORE",
                    "allowed_axioms": [],
                    "declaration_name": "jacobian_theorem",
                    "lean_version": LEAN_VERSION,
                    "lean_commit": LEAN_COMMIT,
                    "import_name": None,
                    "mathlib_commit": None,
                },
            }
        },
        "expected_bindings": bindings,
    }


def test_lean_checker_uses_worker_authorized_executable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "lean"
    executable.write_bytes(b"lean-runtime")
    monkeypatch.setenv("JACOBIAN_CHECKER_EXECUTABLE", str(executable))
    monkeypatch.setenv(
        "JACOBIAN_CHECKER_RUNTIME_DIGEST",
        "sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(lean4_checker.shutil, "which", lambda _name: None)

    assert lean4_checker._lean_command("lean") == (str(executable),)


def test_lean_checker_binds_authorized_lake_launcher(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "lean"
    executable.write_bytes(b"lean-runtime")
    lake = tmp_path / "lake"
    lake.write_bytes(b"lake-runtime")
    monkeypatch.setenv("JACOBIAN_CHECKER_EXECUTABLE", str(executable))
    monkeypatch.setenv(
        "JACOBIAN_CHECKER_RUNTIME_DIGEST",
        "sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv(
        "JACOBIAN_CHECKER_LAKE_DIGEST",
        "sha256:" + hashlib.sha256(lake.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(lean4_checker.shutil, "which", lambda _name: None)

    assert lean4_checker._lean_command("lake") == (str(lake),)


def test_lean_checker_rejects_replaced_authorized_lake_launcher(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "lean"
    executable.write_bytes(b"lean-runtime")
    lake = tmp_path / "lake"
    lake.write_bytes(b"authorized-lake")
    monkeypatch.setenv("JACOBIAN_CHECKER_EXECUTABLE", str(executable))
    monkeypatch.setenv(
        "JACOBIAN_CHECKER_RUNTIME_DIGEST",
        "sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv(
        "JACOBIAN_CHECKER_LAKE_DIGEST",
        "sha256:" + hashlib.sha256(lake.read_bytes()).hexdigest(),
    )
    lake.write_bytes(b"replaced-lake")

    with pytest.raises(RuntimeError, match="lake launcher digest changed"):
        lean4_checker._lean_command("lake")


def test_lean_checker_refuses_unbound_lake_in_authorized_mode(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "lean"
    executable.write_bytes(b"lean-runtime")
    # A sibling lake is present on disk but no digest authorizes it.
    (tmp_path / "lake").write_bytes(b"hostile-lake")
    monkeypatch.setenv("JACOBIAN_CHECKER_EXECUTABLE", str(executable))
    monkeypatch.setenv(
        "JACOBIAN_CHECKER_RUNTIME_DIGEST",
        "sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest(),
    )
    monkeypatch.delenv("JACOBIAN_CHECKER_LAKE_DIGEST", raising=False)
    monkeypatch.setattr(lean4_checker.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="lake launcher is not authorized"):
        lean4_checker._lean_command("lake")


def test_lake_launcher_path_resolves_the_toolchain_sibling(
    tmp_path,
) -> None:
    lean = tmp_path / "lean"
    lean.write_bytes(b"lean")
    (tmp_path / "lake").write_bytes(b"lake")

    assert lean4_checker.lake_launcher_path(lean) == tmp_path / "lake"


def test_lake_launcher_path_rejects_a_symlinked_sibling(
    tmp_path,
) -> None:
    lean = tmp_path / "lean"
    lean.write_bytes(b"lean")
    target = tmp_path / "lake-target"
    target.write_bytes(b"lake")
    (tmp_path / "lake").symlink_to(target)

    assert lean4_checker.lake_launcher_path(lean) is None


def test_mathlib_checker_gives_lake_a_bounded_path_with_git(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = tmp_path / "toolchain" / "bin"
    toolchain.mkdir(parents=True)
    lake = toolchain / "lake"
    lake.write_bytes(b"lake")
    system_bin = tmp_path / "system" / "bin"
    system_bin.mkdir(parents=True)
    git = system_bin / "git"
    git.write_bytes(b"git")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    requests = []

    monkeypatch.delenv("PATH", raising=False)
    monkeypatch.setattr(lean4_checker, "_mathlib_runtime", lambda: runtime)
    monkeypatch.setattr(
        lean4_checker,
        "_lean_command",
        lambda name: (str(lake),) if name == "lake" else (str(toolchain / name),),
    )
    monkeypatch.setattr(lean4_checker, "_validate_lean", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        lean4_checker.shutil,
        "which",
        lambda name: str(git) if name == "git" else None,
    )

    def execute(request):
        requests.append(request)
        return ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=0,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        )

    monkeypatch.setattr(lean4_checker, "execute_process", execute)

    lean4_checker._run_lean("#check True", environment_name="MATHLIB")

    assert len(requests) == 1
    assert requests[0].environment["PATH"] == (
        f"{toolchain.resolve()}:{system_bin.resolve()}"
    )


def test_lean_checker_rejects_replaced_authorized_executable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "lean"
    executable.write_bytes(b"authorized")
    monkeypatch.setenv("JACOBIAN_CHECKER_EXECUTABLE", str(executable))
    monkeypatch.setenv(
        "JACOBIAN_CHECKER_RUNTIME_DIGEST",
        "sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest(),
    )
    executable.write_bytes(b"replaced")

    with pytest.raises(RuntimeError, match="digest changed"):
        lean4_checker._lean_command("lean")


def test_lean_checker_accepts_exact_bound_core_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian_checkers.lean4._run_lean",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="'jacobian_theorem' does not depend on any axioms",
            stderr="",
        ),
    )

    decision = check_kernel_certificate(_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["method"] == "CHECKED_CERTIFICATE"


@pytest.mark.parametrize("mutation", ["bindings", "format", "forbidden_proof"])
def test_lean_checker_rejects_unbound_or_unsupported_certificates(
    mutation: str,
) -> None:
    request = _request()
    certificate = request["certificate"]["payload"]
    if mutation == "bindings":
        certificate["bindings"]["candidate_digest"] = "sha256:" + "9" * 64
    elif mutation == "format":
        certificate["format_version"] = "2"
    else:
        certificate["payload"]["proof"] = "by sorry"
        request["candidate"]["payload"]["proof"] = "by sorry"

    decision = check_kernel_certificate(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_lean_checker_timeout_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian_checkers.lean4._validate_lean", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "jacobian_checkers.lean4._lean_command", lambda _name: ("/usr/bin/lean",)
    )
    monkeypatch.setattr(
        "jacobian_checkers.lean4.execute_process",
        lambda _request: ProcessResult(
            termination=ProcessTermination.TIMED_OUT,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )

    decision = check_kernel_certificate(_request())

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
