from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS = ROOT / "tests" / "fixtures" / "providers"
_CURL_OUTPUT = re.compile(r"\bcurl\s+-fsSL\s+\S+\s+-o\s+(\S+)")
# Split Dockerfile RUN chains into individual commands so that ordering can be
# checked across logical commands, not just physical lines.
_COMMAND_SPLIT = re.compile(r"\s*(?P<operator>&&|;|\|\|)\s*")
_PIPELINE_SPLIT = re.compile(r"\s*(?<!\|)\|(?!\|)\s*")


def _logical_commands(contents: str) -> list[str]:
    """Flatten continuation lines and split RUN chains into logical commands."""

    return [command for _, command in _logical_command_sequence(contents)]


def _logical_command_sequence(contents: str) -> list[tuple[str | None, str]]:
    """Return logical commands with the shell operator that precedes each one."""

    # Join backslash-continuation lines first.
    joined = re.sub(r"\\\s*\n\s*", " ", contents)
    commands: list[tuple[str | None, str]] = []
    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        start = 0
        operator: str | None = None
        for match in _COMMAND_SPLIT.finditer(stripped):
            cmd = stripped[start : match.start()].strip()
            cmd = cmd.strip()
            if cmd:
                commands.append((operator, cmd))
            operator = match.group("operator")
            start = match.end()
        cmd = stripped[start:].strip()
        if cmd:
            commands.append((operator, cmd))
    return commands


def _target_used_unsafely(contents: str, target: str, verify_text: str) -> bool:
    """Return True if ``target`` is extracted/executed before its verification.

    A reference counts as a "use" only when the logical command containing it
    is neither the ``curl ... -o <target>`` download nor the ``sha256sum -c -``
    verification. Download-then-verify within one RUN block is safe because no
    extraction or execution happens between them.
    """

    commands = _logical_command_sequence(contents)
    downloaded = False
    verified = False
    for operator, cmd in commands:
        # A Dockerfile RUN boundary is fail-closed like a successful `&&`
        # continuation: a failed preceding RUN prevents the next one from
        # executing. Semicolons and `||` do not establish that guarantee.
        if operator not in (None, "&&"):
            verified = False
        if target not in cmd:
            continue
        if re.search(r"\bcurl\s+-fsSL\s+\S+\s+-o\s+" + re.escape(target), cmd):
            downloaded = True
            verified = False
            continue  # the download itself; it invalidates prior verification
        if "sha256sum -c -" in cmd:
            pipeline = _PIPELINE_SPLIT.split(cmd)
            # POSIX sh uses the last pipeline stage's status, so a trailing
            # command such as tee can otherwise mask checksum failure.
            checksum_is_pipeline_status = "sha256sum -c -" in pipeline[-1]
            if (
                verify_text in cmd
                and downloaded
                and operator in (None, "&&")
                and checksum_is_pipeline_status
            ):
                verified = True
            continue  # a different target's verification, or a stale file
        if not verified:
            return True  # extraction/execution before the latest verification
    return False


def test_provider_downloads_are_sha256_verified_before_use() -> None:
    """Each downloaded target must be SHA-256 verified before any extraction
    or execution of that target, not merely verified somewhere in the file."""

    for dockerfile in sorted(PROVIDERS.glob("*/environment/Dockerfile")):
        contents = dockerfile.read_text(encoding="utf-8")
        downloads = _CURL_OUTPUT.findall(contents)
        for target in downloads:
            verification = f"{target} | sha256sum -c -"
            assert verification in contents, (
                f"{dockerfile.relative_to(ROOT)} downloads {target} without "
                "an explicit SHA-256 verification"
            )
            assert not _target_used_unsafely(contents, target, verification), (
                f"{dockerfile.relative_to(ROOT)} uses {target} before its "
                "SHA-256 verification"
            )


def test_checksum_must_follow_the_latest_download() -> None:
    target = "/opt/provider/pkg.tgz"
    verification = f"{target} | sha256sum -c -"
    stale_verification = (
        f"printf '%s  %s\\n' deadbeef {target} | sha256sum -c - && "
        f"curl -fsSL https://example.test/pkg.tgz -o {target} && "
        f"tar -xzf {target}"
    )

    assert _target_used_unsafely(stale_verification, target, verification)


def test_download_verify_use_sequence_is_accepted() -> None:
    target = "/opt/provider/pkg.tgz"
    verification = f"{target} | sha256sum -c -"
    safe_sequence = (
        f"curl -fsSL https://example.test/pkg.tgz -o {target} && "
        f"printf '%s  %s\\n' deadbeef {target} | sha256sum -c - && "
        f"tar -xzf {target}"
    )

    assert not _target_used_unsafely(safe_sequence, target, verification)


def test_checksum_pipeline_status_must_gate_use() -> None:
    target = "/opt/provider/pkg.tgz"
    verification = f"{target} | sha256sum -c -"
    unsafe_sequence = (
        f"curl -fsSL https://example.test/pkg.tgz -o {target} && "
        f"printf '%s  %s\\n' deadbeef {target} | sha256sum -c - | "
        f"tee /tmp/check.log && tar -xzf {target}"
    )

    assert _target_used_unsafely(unsafe_sequence, target, verification)


@pytest.mark.parametrize(
    "unsafe_sequence",
    [
        (
            "curl -fsSL https://example.test/pkg.tgz -o /opt/provider/pkg.tgz; "
            "printf '%s  %s\\n' deadbeef /opt/provider/pkg.tgz | sha256sum -c -; "
            "tar -xzf /opt/provider/pkg.tgz"
        ),
        (
            "curl -fsSL https://example.test/pkg.tgz -o /opt/provider/pkg.tgz && "
            "printf '%s  %s\\n' deadbeef /opt/provider/pkg.tgz | sha256sum -c - || "
            "true && tar -xzf /opt/provider/pkg.tgz"
        ),
    ],
)
def test_checksum_failure_paths_cannot_reach_use(unsafe_sequence: str) -> None:
    target = "/opt/provider/pkg.tgz"
    verification = f"{target} | sha256sum -c -"

    assert _target_used_unsafely(unsafe_sequence, target, verification)
