from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS = ROOT / "benchmarks" / "datasets" / "provider-feasibility-v1"
_CURL_OUTPUT = re.compile(r"\bcurl\s+-fsSL\s+\S+\s+-o\s+(\S+)")
# Split Dockerfile RUN chains into individual commands so that ordering can be
# checked across logical commands, not just physical lines.
_COMMAND_SPLIT = re.compile(r"\s*(?:&&|;|\|\|)\s*")


def _logical_commands(contents: str) -> list[str]:
    """Flatten continuation lines and split RUN chains into logical commands."""

    # Join backslash-continuation lines first.
    joined = re.sub(r"\\\s*\n\s*", " ", contents)
    commands: list[str] = []
    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for cmd in _COMMAND_SPLIT.split(stripped):
            cmd = cmd.strip()
            if cmd:
                commands.append(cmd)
    return commands


def _target_used_unsafely(contents: str, target: str, verify_text: str) -> bool:
    """Return True if ``target`` is extracted/executed before its verification.

    A reference counts as a "use" only when the logical command containing it
    is neither the ``curl ... -o <target>`` download nor the ``sha256sum -c -``
    verification. Download-then-verify within one RUN block is safe because no
    extraction or execution happens between them.
    """

    commands = _logical_commands(contents)
    for cmd in commands:
        if target not in cmd:
            continue
        if "sha256sum -c -" in cmd:
            if verify_text in cmd:
                return False  # reached the verification: no unsafe use found
            continue  # a different target's verification; skip
        if re.search(r"\bcurl\s+-fsSL\s+\S+\s+-o\s+" + re.escape(target), cmd):
            continue  # the download itself
        return True  # extraction/execution before verification
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


def test_lean_separate_verifier_receives_both_bound_inputs() -> None:
    task = PROVIDERS / "lean-repl"
    assert (task / "tests" / "input.json").read_bytes() == (
        task / "environment" / "input.json"
    ).read_bytes()
    assert '"/app/input.json"' in (task / "task.toml").read_text(encoding="utf-8")
    assert "COPY expected.json input.json " in (
        task / "tests" / "Dockerfile"
    ).read_text(encoding="utf-8")
