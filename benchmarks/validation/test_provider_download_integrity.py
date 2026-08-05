from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS = ROOT / "benchmarks" / "datasets" / "provider-feasibility-v1"
_CURL_OUTPUT = re.compile(r"\bcurl\s+-fsSL\s+\S+\s+-o\s+(\S+)")


def test_provider_downloads_are_sha256_verified_before_use() -> None:
    for dockerfile in sorted(PROVIDERS.glob("*/environment/Dockerfile")):
        contents = dockerfile.read_text(encoding="utf-8")
        downloads = _CURL_OUTPUT.findall(contents)
        for target in downloads:
            verification = f"{target} | sha256sum -c -"
            assert verification in contents, (
                f"{dockerfile.relative_to(ROOT)} downloads {target} without "
                "an explicit SHA-256 verification"
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
