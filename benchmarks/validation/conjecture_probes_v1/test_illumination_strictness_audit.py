from __future__ import annotations

import importlib.util
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/illumination-strictness-audit"
VERTICES = list(product((-1, 1), repeat=3))


def _module():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "illumination_verifier", TASK / "tests/verifier.py"
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
    flawed = [[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]]
    repair = [[-x for x in v] for v in VERTICES]
    pairs = [
        {"vertex_index": i, "direction_index": j}
        for i, v in enumerate(VERTICES)
        for j, d in enumerate(flawed)
        if module._weak(v, d) and not module._strict(v, d)
    ]
    return {
        "flawed_directions": flawed,
        "weak_false_positive_pairs": pairs,
        "repair_directions": repair,
        "vertex_to_direction": list(range(8)),
    }


def test_oracle_mathematics():
    module = _module()
    assert module.mathematics(_result(module))


def test_rejects_zero_vector_shortcut():
    module = _module()
    result = _result(module)
    result["flawed_directions"][0] = [0, 0, 0]
    assert not module.mathematics(result)


def test_rejects_missing_false_positive():
    module = _module()
    result = _result(module)
    result["weak_false_positive_pairs"] = result["weak_false_positive_pairs"][:-1]
    assert not module.mathematics(result)


def test_rejects_unhashable_direction_without_crashing():
    module = _module()
    result = _result(module)
    result["flawed_directions"][0] = [[-1], -1, 0]
    assert not module.mathematics(result)


def test_rejects_non_integer_false_positive_indices():
    module = _module()
    for replacement in (False, 0.0):
        result = _result(module)
        result["weak_false_positive_pairs"][0]["vertex_index"] = replacement
        assert not module.mathematics(result)


def test_evidence_result_comparison_preserves_json_types():
    module = _module()
    result = _result(module)
    evidence_result = _result(module)
    evidence_result["vertex_to_direction"][0] = False
    assert not module._json_equal(evidence_result, result)

    evidence_result = _result(module)
    evidence_result["weak_false_positive_pairs"][0]["direction_index"] = 0.0
    assert not module._json_equal(evidence_result, result)


def test_accepts_reordered_repair():
    module = _module()
    result = _result(module)
    result["repair_directions"].reverse()
    result["vertex_to_direction"].reverse()
    assert module.mathematics(result)
