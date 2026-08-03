from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
INSTALLER = REPOSITORY_ROOT / "deploy" / "install.sh"
SERVICE_STATE = REPOSITORY_ROOT / "deploy" / "lib" / "service_state.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="deploy installer tests require bash",
)


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(INSTALLER), *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_installer_is_valid_bash() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(INSTALLER)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_installer_help_exposes_three_deployment_modes() -> None:
    completed = _run("--help")

    assert completed.returncode == 0
    assert "--mode local" in completed.stdout
    assert "--mode domain" in completed.stdout
    assert "--mode tailscale" in completed.stdout


def test_domain_dry_run_reports_connector_without_requiring_root() -> None:
    completed = _run(
        "--mode",
        "domain",
        "--domain",
        "math.example.org",
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    assert "mode:        domain" in completed.stdout
    assert "connector:   https://math.example.org/mcp" in completed.stdout
    assert "python:      /opt/jacobian/python" in completed.stdout
    assert "caddy:       enabled" in completed.stdout
    assert "funnel:      disabled" in completed.stdout


def test_dry_run_never_echoes_supplied_credentials(tmp_path: Path) -> None:
    sentinel = "sentinel-secret-that-must-not-be-logged"
    credentials = tmp_path / "tokens.json"
    credentials.write_text(
        '{"tokens":{"' + sentinel + '":{"tenant_id":"tenant-a"}}}',
        encoding="utf-8",
    )

    completed = _run("--auth-tokens-file", str(credentials), "--dry-run")

    assert completed.returncode == 0, completed.stderr
    assert sentinel not in completed.stdout
    assert sentinel not in completed.stderr


def test_domain_mode_requires_a_valid_fqdn() -> None:
    missing = _run("--mode", "domain", "--dry-run")
    invalid = _run(
        "--mode",
        "domain",
        "--domain",
        "https://math.example.org/path",
        "--dry-run",
    )

    assert missing.returncode != 0
    assert "--mode domain requires --domain" in missing.stderr
    assert invalid.returncode != 0
    assert "fully qualified DNS name" in invalid.stderr


def test_public_anonymous_mode_requires_double_confirmation() -> None:
    rejected = _run(
        "--mode",
        "domain",
        "--domain",
        "math.example.org",
        "--allow-anonymous",
        "--dry-run",
    )
    accepted = _run(
        "--mode",
        "domain",
        "--domain",
        "math.example.org",
        "--allow-anonymous",
        "--confirm-public-anonymous",
        "--dry-run",
    )

    assert rejected.returncode != 0
    assert "--confirm-public-anonymous" in rejected.stderr
    assert accepted.returncode == 0, accepted.stderr
    assert "auth:        anonymous shared tenant jacobian-test" in accepted.stdout


def test_local_mode_rejects_an_unusable_domain_option() -> None:
    completed = _run(
        "--mode",
        "local",
        "--domain",
        "math.example.org",
        "--dry-run",
    )

    assert completed.returncode != 0
    assert "--domain is only valid with --mode domain" in completed.stderr


def test_release_environment_is_built_at_its_final_path() -> None:
    source = INSTALLER.read_text(encoding="utf-8")
    release_block = source[
        source.index('log "installing immutable release') : source.index(
            'log "installing authentication configuration"'
        )
    ]

    assert 'cd "${RELEASE_DIR}"' in release_block
    assert 'UV_PYTHON_INSTALL_DIR="${PYTHON_INSTALL_ROOT}"' in release_block
    assert "--managed-python" in release_block
    assert "--link-mode copy" in release_block
    assert 'mv "${RELEASE_CANDIDATE}" "${RELEASE_DIR}"' not in release_block
    assert '"${FLOCK_BIN}" --nonblock 9' in release_block


def test_release_runtime_is_checked_before_current_symlink_is_changed() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    validation = source.index('validate_release_runtime "${RELEASE_DIR}"')
    revision_marker = source.index(
        'printf \'%s\\n\' "${REVISION}" >"${RELEASE_DIR}/.git-revision"'
    )
    current_link = source.index('ln -sfn "${RELEASE_DIR}" "${CURRENT_LINK}.new"')

    assert validation < revision_marker < current_link
    assert '"${RUNUSER_BIN}" --user jacobian -- "${entrypoint}" --version' in source


def test_activation_arms_rollback_before_switching_current() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    armed = source.index("ROLLBACK_ARMED=1")
    current_link = source.index('ln -sfn "${RELEASE_DIR}" "${CURRENT_LINK}.new"')
    smoke = source.index('log "running the read-only deployment smoke"')
    accepted = source.index("DEPLOYMENT_ACCEPTED=1")

    assert armed < current_link < smoke < accepted
    assert 'return "${original_status}"' in source
    assert 'exit "${status}"' in source
    assert "rollback encountered additional failures" in source


def test_generated_token_is_written_only_to_the_restricted_file() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600" in source
    assert "print(token)" not in source
    assert "print(next(grant.token" not in source
    assert "JACOBIAN_MCP_AUTH_TOKENS_FILE" in source
    assert "generated bearer token: ${" not in source
    assert "retrieve it explicitly with privileged access" in source


def test_rollback_restores_prior_service_activity_and_enablement(
    tmp_path: Path,
) -> None:
    state = tmp_path / "systemd-state"
    snapshots = tmp_path / "snapshots"
    state.mkdir()
    snapshots.mkdir()
    fake_systemctl = tmp_path / "systemctl"
    fake_systemctl.write_text(
        """#!/usr/bin/env bash
set -eu
action="$1"
case "$action" in
  is-enabled) test -f "$FAKE_SYSTEMD_STATE/$3.enabled" ;;
  is-active) test -f "$FAKE_SYSTEMD_STATE/$3.active" ;;
  enable) : >"$FAKE_SYSTEMD_STATE/$2.enabled" ;;
  disable) rm -f -- "$FAKE_SYSTEMD_STATE/$2.enabled" ;;
  restart) : >"$FAKE_SYSTEMD_STATE/$2.active" ;;
  stop) rm -f -- "$FAKE_SYSTEMD_STATE/$2.active" ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    fake_systemctl.chmod(0o755)
    (state / "jacobian-mcp.service.enabled").touch()
    (state / "jacobian-mcp.service.active").touch()
    (state / "jacobian-caddy.service.enabled").touch()
    environment = os.environ | {"FAKE_SYSTEMD_STATE": str(state)}
    units = (
        "jacobian-mcp.service",
        "jacobian-caddy.service",
        "jacobian-funnel.service",
    )

    for unit in units:
        subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; snapshot_systemd_service_state "$2" "$3" "$4"',
                "service-state-test",
                str(SERVICE_STATE),
                str(fake_systemctl),
                str(snapshots),
                unit,
            ],
            check=True,
            env=environment,
        )
        (state / f"{unit}.enabled").touch()
        (state / f"{unit}.active").touch()

    for unit in units:
        subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; restore_systemd_service_state "$2" "$3" "$4"',
                "service-state-test",
                str(SERVICE_STATE),
                str(fake_systemctl),
                str(snapshots),
                unit,
            ],
            check=True,
            env=environment,
        )

    assert (state / "jacobian-mcp.service.enabled").is_file()
    assert (state / "jacobian-mcp.service.active").is_file()
    assert (state / "jacobian-caddy.service.enabled").is_file()
    assert not (state / "jacobian-caddy.service.active").exists()
    assert not (state / "jacobian-funnel.service.enabled").exists()
    assert not (state / "jacobian-funnel.service.active").exists()
