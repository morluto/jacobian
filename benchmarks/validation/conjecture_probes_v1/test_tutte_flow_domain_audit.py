from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/tutte-flow-domain-audit"
FLAWED = [3, 2, 1, 4, 0, 1, 2, 4, 2, 1, 2, 1, 1, 2, 4]
REPAIR = [2, 1, 4, 3, 1, 1, 3, 3, 3, 1, 2, 1, 2, 1, 4]


def _module():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "tutte_flow_verifier", TASK / "tests/verifier.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_submission_is_bounded_before_read(monkeypatch):
    module = _module()

    class UnreadablePath:
        def __init__(self, _value):
            pass

        def read_text(self):
            raise AssertionError("oversized submission must not be read")

    def reject_oversized(_path, *, max_bytes):
        assert max_bytes == module.MAX_SUBMISSION_BYTES
        return False

    monkeypatch.setattr(module, "Path", UnreadablePath)
    monkeypatch.setattr(module, "is_regular_bounded_file", reject_oversized)
    assert module._raw() is None


def _result(module):
    return {
        "flawed_flow": FLAWED,
        "flawed_balances": module._balances(FLAWED),
        "zero_edge_index": 4,
        "repair_flow": REPAIR,
        "repair_balances": module._balances(REPAIR),
    }


def test_oracle_mathematics():
    module = _module()
    assert module.mathematics(_result(module))


def test_rejects_zero_flow_shortcut():
    module = _module()
    result = _result(module)
    result["flawed_flow"] = [0] * 15
    result["flawed_balances"] = [0] * 10
    assert not module.mathematics(result)


def test_rejects_zero_in_repair():
    module = _module()
    result = _result(module)
    result["repair_flow"] = FLAWED
    result["repair_balances"] = [0] * 10
    assert not module.mathematics(result)


def test_accepts_scalar_multiple():
    module = _module()
    result = _result(module)
    scaled = [(2 * x) % 5 for x in REPAIR]
    result["repair_flow"] = scaled
    result["repair_balances"] = module._balances(scaled)
    assert module.mathematics(result)
