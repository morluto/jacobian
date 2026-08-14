"""Contracts for source bootstrap identity and environment pins."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _load_tool(name: str) -> ModuleType:
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bootstrap_checkout(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "tools").mkdir(parents=True)
    (checkout / "npm" / "bin").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "setup-agent", checkout / "scripts" / "setup-agent")
    shutil.copy2(ROOT / "tools" / "setup_lean.py", checkout / "tools" / "setup_lean.py")
    (checkout / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    (checkout / "uv.lock").touch()
    (checkout / ".uv-version").write_text("0.0.0\n")
    (checkout / ".gitignore").write_text(".state/\n.venv-custom/\n")
    (checkout / "npm" / "bin" / "jacobian.cjs").touch()
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv_log = tmp_path / "uv.log"
    node_log = tmp_path / "node.log"
    (fake_bin / "uv").write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >>"$UV_LOG"
if [[ "$1" == "--version" ]]; then
  echo "uv 0.0.0"
elif [[ "$1" == "python" && "$2" == "find" ]]; then
  [[ " $* " == *" --no-python-downloads "* ]] || exit 92
  echo "$BOOTSTRAP_PYTHON"
else
  exit 91
fi
""",
        encoding="utf-8",
    )
    (fake_bin / "node").write_text(
        """#!/usr/bin/env bash
set -eu
if [[ "$1" == "-p" ]]; then
  echo 18
  exit 0
fi
printf '%s\\n' "$@" >"$NODE_LOG"
""",
        encoding="utf-8",
    )
    for executable in (fake_bin / "uv", fake_bin / "node"):
        executable.chmod(0o755)
    environment = os.environ | {
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        "UV_LOG": str(uv_log),
        "NODE_LOG": str(node_log),
        "BOOTSTRAP_PYTHON": sys.executable,
        "UV_PROJECT_ENVIRONMENT": str(checkout / ".venv-custom"),
        "ELAN_HOME": str(tmp_path / "elan"),
        "JACOBIAN_LEAN_RUNTIME": str(tmp_path / "lean"),
    }
    return checkout, environment


def _run_bootstrap(
    checkout: Path,
    environment: dict[str, str],
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(checkout / "scripts" / "setup-agent"), *arguments],
        cwd=checkout,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_version_identity_uses_uv_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doctor = _load_tool("source_agent_doctor")
    from benchmarks.tooling.command_runner import (
        ToolCommandResult,
        ToolCommandStatus,
    )

    def fake_run_operator_command(
        command: str,
        arguments: tuple[str, ...] = (),
        **_kwargs: object,
    ) -> object:
        return ToolCommandResult(
            status=ToolCommandStatus.EXITED,
            exit_code=0,
            stdout=b"0.7.0a0\n",
            stderr=b"",
        )

    monkeypatch.setattr(doctor, "run_operator_command", fake_run_operator_command)
    assert doctor._repository_version(ROOT) == "0.7.0a0"


def test_active_uv_surfaces_share_the_repository_pin() -> None:
    pinned = (ROOT / ".uv-version").read_text().strip()
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert f"ghcr.io/astral-sh/uv:{pinned}-" in dockerfile
    assert "RUN uv sync --locked --no-dev\n" in dockerfile
    assert "--extra" not in dockerfile
    setup_files = [
        ROOT / ".github" / "actions" / "setup-python-tests" / "action.yml",
        ROOT / ".github" / "actions" / "setup-lean" / "action.yml",
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / ".github" / "workflows" / "release.yml",
        ROOT / ".github" / "workflows" / "release-please.yml",
    ]
    for path in setup_files:
        text = path.read_text()
        assert text.count("astral-sh/setup-uv@") == text.count(f'version: "{pinned}"')


def test_every_bootstrap_profile_audits_z3_and_networkx() -> None:
    doctor = _load_tool("source_agent_doctor")
    for providers in doctor._PROFILE_PROVIDERS.values():
        assert "z3" in providers
        assert "networkx" in providers


def test_source_bootstrap_uses_locked_uv_sync_and_optional_lean_setup() -> None:
    script = (ROOT / "scripts" / "setup-agent").read_text(encoding="utf-8")

    assert "uv sync --locked" in script
    assert "tools/setup_lean.py" in script
    assert "development_profiles.py" not in script
    assert "lake update" not in script
    assert "lake build" not in script


def test_bootstrap_dry_run_forwards_resolved_environment_without_downloads(
    tmp_path: Path,
) -> None:
    checkout, environment = _bootstrap_checkout(tmp_path)
    state_dir = checkout / ".state"

    completed = _run_bootstrap(
        checkout,
        environment,
        "--client",
        "codex",
        "--state-dir",
        str(state_dir),
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    assert "python find --no-python-downloads" in (tmp_path / "uv.log").read_text(
        encoding="utf-8"
    )
    arguments = (tmp_path / "node.log").read_text(encoding="utf-8").splitlines()
    assert arguments[1:] == [
        "setup",
        "--source",
        str(checkout),
        "--state-dir",
        str(state_dir),
        "--uv-bin",
        str(tmp_path / "bin" / "uv"),
        "--profile",
        "core",
        "--provider-path",
        arguments[arguments.index("--provider-path") + 1],
        "--project-environment",
        str(checkout / ".venv-custom"),
        "--elan-home",
        str(tmp_path / "elan"),
        "--lean-runtime",
        str(tmp_path / "lean"),
        "--client",
        "codex",
        "--dry-run",
    ]
    assert not state_dir.exists()


def test_bootstrap_rejects_invalid_client_and_dirty_checkout_before_uv(
    tmp_path: Path,
) -> None:
    checkout, environment = _bootstrap_checkout(tmp_path)

    invalid_client = _run_bootstrap(
        checkout,
        environment,
        "--client",
        "unknown",
        "--dry-run",
    )
    assert invalid_client.returncode == 2
    assert "unknown MCP client" in invalid_client.stderr
    assert not (tmp_path / "uv.log").exists()

    (checkout / "dirty.txt").write_text("dirty\n")
    dirty_checkout = _run_bootstrap(
        checkout,
        environment,
        "--client",
        "codex",
        "--dry-run",
    )
    assert dirty_checkout.returncode == 2
    assert "requires a clean checkout" in dirty_checkout.stderr
    assert not (tmp_path / "uv.log").exists()


@pytest.mark.skipif(
    shutil.which("git") is None,
    reason="requires git to initialize a test checkout",
)
def test_nonexistent_checkout_directory_uses_git_directory_ignore_semantics(
    tmp_path: Path,
) -> None:
    checkout, environment = _bootstrap_checkout(tmp_path)
    completed = _run_bootstrap(
        checkout,
        environment,
        "--client",
        "codex",
        "--state-dir",
        str(checkout / "not-ignored"),
        "--dry-run",
    )

    assert completed.returncode == 2
    assert "inside the checkout must be ignored by Git" in completed.stderr
