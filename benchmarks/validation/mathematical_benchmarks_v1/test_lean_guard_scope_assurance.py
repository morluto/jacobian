from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
TASK = (
    ROOT / "benchmarks/datasets/mathematical-benchmarks-v1/lean-guard-scope-assurance"
)


def _module():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "lean_guard_scope_assurance_verifier", TASK / "tests/verifier.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _allzero_bad_guard_case() -> dict[str, object]:
    return {
        "id": "allzero_bad_lt_guard",
        "ofnat_zero_equals_one": True,
        "lt_is_universal": True,
    }


def test_accepts_negated_guard_strength_with_positive_zero_divisor_finding() -> None:
    module = _module()
    item = {
        "id": "allzero_bad_lt_guard",
        "findings": ["DIVISION_BY_ZERO"],
        "reason": (
            "The universal order guard does not supply a sound nonzero divisor "
            "fact, so the divisor remains semantically zero."
        ),
    }

    assert module._result_item_ok(item, _allzero_bad_guard_case())


def test_rejects_reason_that_denies_division_by_zero() -> None:
    module = _module()
    item = {
        "id": "allzero_bad_lt_guard",
        "findings": ["DIVISION_BY_ZERO"],
        "reason": "There is not division by zero because the divisor is safe.",
    }

    assert not module._result_item_ok(item, _allzero_bad_guard_case())
