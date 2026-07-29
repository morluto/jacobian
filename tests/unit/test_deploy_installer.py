from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPOSITORY_ROOT / "deploy" / "install.sh"


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
