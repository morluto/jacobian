from __future__ import annotations

import os
from pathlib import Path

import pytest
from benchmarks.tooling.codex_binary import resolve_codex_binary


def _write_executable(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o755)
    return path


def test_resolve_codex_binary_accepts_native_linux_executable(tmp_path: Path) -> None:
    binary = _write_executable(tmp_path / "codex", b"\x7fELFfixture")

    assert resolve_codex_binary(binary) == binary.resolve()


def test_resolve_codex_binary_finds_native_payload_for_npm_launcher(
    tmp_path: Path,
) -> None:
    package = tmp_path / "lib/node_modules/@openai/codex"
    launcher = _write_executable(package / "bin/codex.js", b"#!/usr/bin/env node\n")
    native = _write_executable(
        package
        / "node_modules/@openai/codex-linux-x64"
        / "vendor/x86_64-unknown-linux-musl/bin/codex",
        b"\x7fELFfixture",
    )

    assert resolve_codex_binary(launcher) == native.resolve()


def test_resolve_codex_binary_rejects_launcher_without_native_payload(
    tmp_path: Path,
) -> None:
    launcher = _write_executable(
        tmp_path / "lib/node_modules/@openai/codex/bin/codex.js",
        b"#!/usr/bin/env node\n",
    )

    with pytest.raises(ValueError, match="Linux standalone Codex binary"):
        resolve_codex_binary(launcher)


def test_resolve_codex_binary_requires_executable_file(tmp_path: Path) -> None:
    binary = tmp_path / "codex"
    binary.write_bytes(b"\x7fELFfixture")
    binary.chmod(0o644)

    with pytest.raises(ValueError, match="executable"):
        resolve_codex_binary(binary)
    assert not os.access(binary, os.X_OK)
