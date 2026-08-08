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


def test_accepts_reordered_repair():
    module = _module()
    result = _result(module)
    result["repair_directions"].reverse()
    result["vertex_to_direction"].reverse()
    assert module.mathematics(result)
