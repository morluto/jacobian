"""Executable ownership checks for Harbor verifier support copies."""

from __future__ import annotations

from pathlib import Path

from benchmarks.tooling.harbor_suite import load_registry

ROOT = Path(__file__).parents[3]


def test_harbor_verifier_support_copies_are_identical_and_registry_owned() -> None:
    source = (ROOT / "benchmarks" / "tooling" / "verifier_support.py").read_bytes()
    targets = sorted(
        path
        for suite in load_registry()
        for path in suite.path.glob("*/tests/verifier_support.py")
    )
    expected = {
        ref.path / "tests" / "verifier_support.py"
        for suite in load_registry()
        for ref in suite.tasks
    }
    assert set(targets) == expected
    assert all(target.read_bytes() == source for target in targets)
