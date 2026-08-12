from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from jacobian.process_policy import ProcessRequest, ProcessResult, ProcessTermination
from jacobian_checkers.lean4 import (
    LEAN_TOOLCHAIN,
    _elan_home,
    _lean_command,
    _lean_rejection,
    _LeanSetupError,
    _resolve_elan_toolchain_executable,
    _run_lean,
    _source,
    _validate_lean,
)


def test_lean_rejection_keeps_repair_context_without_local_details() -> None:
    detail = _lean_rejection(
        "/tmp/jacobian-lean-secret/Main.lean:7:12: error: unexpected token "
        "provider=hidden /private/toolchain/cache"
    )

    assert detail == (
        "Lean rejected the proof at line 7, column 12: unexpected token "
        "<local-path>. Correct the proof body and retry."
    )
    assert "jacobian-lean-secret" not in detail
    assert "provider=hidden" not in detail
    assert "/private" not in detail


def test_lean_rejection_has_a_generic_recovery_for_unknown_output() -> None:
    assert _lean_rejection("unstructured local compiler output") == (
        "Lean rejected the proof. Check the statement and proof body, then retry."
    )


def test_lean_rejection_accepts_named_lean_diagnostics() -> None:
    detail = _lean_rejection(
        "<stdin>:6:8: error(lean.unknownIdentifier): Unknown identifier "
        "`not_a_mathlib_theorem`"
    )

    assert detail == (
        "Lean rejected the proof at line 6, column 8: Unknown identifier "
        "`not_a_mathlib_theorem`. Correct the proof body and retry."
    )


@pytest.mark.parametrize(
    "diagnostic",
    [
        'Main.lean:7:12: error: failed at "/tmp/private path/file.lean"',
        "Main.lean:7:12: error: failed at /tmp/private path/file.lean",
        r"Main.lean:7:12: error: failed at C:\Users\Alice Smith\x.lean",
        r"Main.lean:7:12: error: failed at \\server\private\file.lean",
        "Main.lean:7:12: error: failed at ~/private/file.lean",
        "Main.lean:7:12: error: provider: secret unexpected token",
        "Main.lean:7:12: error: internal_id = secret unexpected token",
    ],
)
def test_lean_rejection_removes_independent_local_diagnostics(
    diagnostic: str,
) -> None:
    detail = _lean_rejection(diagnostic)

    assert "private" not in detail
    assert "secret" not in detail
    assert "provider" not in detail
    assert "internal_id" not in detail


def test_elan_command_selects_the_pinned_toolchain_without_a_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian_checkers.lean4.shutil.which",
        lambda name: "/opt/elan/bin/elan" if name == "elan" else None,
    )

    assert _lean_command("lean") == (
        "/opt/elan/bin/elan",
        "run",
        LEAN_TOOLCHAIN,
        "lean",
    )


def test_authorized_checker_resolves_lake_beside_the_lean_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain_bin = tmp_path / "toolchains" / "lean4" / "bin"
    toolchain_bin.mkdir(parents=True)
    lean = toolchain_bin / "lean"
    lake = toolchain_bin / "lake"
    lean_bytes = b"lean-bin"
    lake_bytes = b"lake-bin"
    lean.write_bytes(lean_bytes)
    lake.write_bytes(lake_bytes)
    lean_path = str(lean.resolve())
    digest = "sha256:" + hashlib.sha256(lean_bytes).hexdigest()
    lake_digest = "sha256:" + hashlib.sha256(lake_bytes).hexdigest()
    monkeypatch.setenv("JACOBIAN_CHECKER_EXECUTABLE", lean_path)
    monkeypatch.setenv("JACOBIAN_CHECKER_RUNTIME_DIGEST", digest)
    monkeypatch.setenv("JACOBIAN_CHECKER_LAKE_DIGEST", lake_digest)

    assert _lean_command("lean") == (lean_path,)
    assert _lean_command("lake") == (str(lake.resolve()),)


def test_elan_toolchain_resolution_uses_print_prefix_not_proxy_which(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain_bin = tmp_path / "toolchains" / "leanprover--lean4---v4.31.0" / "bin"
    toolchain_bin.mkdir(parents=True)
    lean = toolchain_bin / "lean"
    lean.write_bytes(b"toolchain-lean")
    elan_proxy = tmp_path / "elan" / "bin" / "lean"
    elan_proxy.parent.mkdir(parents=True)
    elan_proxy.write_bytes(b"elan-proxy")
    seen: list[tuple[str, ...]] = []

    def completed(request: ProcessRequest, **_kwargs: object) -> ProcessResult:
        seen.append((request.executable, *request.arguments))
        return ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=0,
            stdout=f"{toolchain_bin.parent}\n".encode(),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        )

    monkeypatch.setattr("jacobian_checkers.lean4.execute_process", completed)
    monkeypatch.setattr(
        "jacobian_checkers.lean4.shutil.which",
        lambda name: str(elan_proxy) if name == "elan" else None,
    )

    resolved = _resolve_elan_toolchain_executable(
        ("/opt/elan/bin/elan", "run", LEAN_TOOLCHAIN, "lean")
    )

    assert resolved == lean
    assert seen == [
        ("/opt/elan/bin/elan", "run", LEAN_TOOLCHAIN, "lean", "--print-prefix")
    ]


def test_elan_toolchain_resolution_rejects_elan_proxy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    elan_proxy = tmp_path / "elan"
    elan_proxy.write_bytes(b"elan-proxy")

    def completed(request: ProcessRequest, **_kwargs: object) -> ProcessResult:
        return ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=0,
            stdout=f"{tmp_path}\n".encode(),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        )

    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "lean").write_bytes(b"elan-proxy")
    # Make bin/lean the same file as the elan proxy for samefile().
    (tmp_path / "bin" / "lean").unlink()
    (tmp_path / "bin" / "lean").hardlink_to(elan_proxy)

    monkeypatch.setattr("jacobian_checkers.lean4.execute_process", completed)
    monkeypatch.setattr(
        "jacobian_checkers.lean4.shutil.which",
        lambda name: str(elan_proxy) if name == "elan" else None,
    )

    with pytest.raises(_LeanSetupError, match="elan proxy"):
        _resolve_elan_toolchain_executable(
            ("/opt/elan/bin/elan", "run", LEAN_TOOLCHAIN, "lean")
        )


def test_system_elan_uses_the_original_user_toolchain_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ELAN_HOME", raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    assert _elan_home(("/usr/bin/elan", "run", LEAN_TOOLCHAIN, "lean")) == (
        str(home / ".elan")
    )


def test_mathlib_validates_lean_before_the_digest_bound_lake_compiler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated: list[tuple[tuple[str, ...], Path | None]] = []
    monkeypatch.setattr(
        "jacobian_checkers.lean4._mathlib_runtime",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "jacobian_checkers.lean4._lean_command",
        lambda name: (f"/usr/bin/{name}",),
    )
    monkeypatch.setattr(
        "jacobian_checkers.lean4._validate_lean",
        lambda command, *, cwd=None: validated.append((command, cwd)),
    )
    monkeypatch.setattr(
        "jacobian_checkers.lean4._mathlib_process_path",
        lambda _command: "/usr/bin",
    )
    monkeypatch.setattr(
        "jacobian_checkers.lean4._mathlib_git_config",
        lambda _runtime: {},
    )
    monkeypatch.setattr(
        "jacobian_checkers.lean4.execute_process",
        lambda *_args, **_kwargs: ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=0,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )

    _run_lean("", environment_name="MATHLIB")

    assert validated == [(("/usr/bin/lean",), tmp_path)]


def test_source_accepts_let_expressions_and_inline_by_terms() -> None:
    inline = _source("True", "by trivial", None)
    witness = _source("let n : Nat := 2; n + n = 4", "rfl", None)

    assert "theorem jacobian_theorem : (True) := by trivial" in inline
    assert ":= by\n  by trivial" not in inline
    assert (
        "theorem jacobian_theorem : (let n : Nat := 2; n + n = 4) := by\n  rfl"
    ) in witness


def test_missing_pinned_toolchain_names_the_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian_checkers.lean4.execute_process",
        lambda *_args, **_kwargs: ProcessResult(
            termination=ProcessTermination.START_FAILED,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )

    with pytest.raises(_LeanSetupError) as raised:
        _validate_lean(("/opt/elan/bin/elan", "run", LEAN_TOOLCHAIN, "lean"))

    assert str(raised.value) == (
        "TOOLCHAIN_PROBE: The pinned Lean 4.31.0 toolchain is unavailable. "
        "Install it with "
        "`elan toolchain install leanprover/lean4:v4.31.0`, then retry."
    )


def test_toolchain_probe_allows_cold_start_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts: list[float] = []
    outputs = iter((b"4.31.0", b"68218e876d2a38b1985b8590fff244a83c321783"))

    def completed(request: ProcessRequest, **_kwargs: object) -> ProcessResult:
        timeouts.append(request.timeout_seconds)
        return ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=0,
            stdout=next(outputs),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        )

    monkeypatch.setattr("jacobian_checkers.lean4.execute_process", completed)

    _validate_lean(("/opt/elan/bin/elan", "run", LEAN_TOOLCHAIN, "lean"))

    assert timeouts == [15.0, 15.0]
