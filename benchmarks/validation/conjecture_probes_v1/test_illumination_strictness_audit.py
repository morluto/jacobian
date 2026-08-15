from __future__ import annotations

from itertools import product
from pathlib import Path

from benchmarks.validation._source_module import load_task_verifier

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/illumination-strictness-audit"
VERTICES = list(product((-1, 1), repeat=3))


def _module():
    return load_task_verifier(TASK, module_name="illumination_verifier")


def _result(module):
    flawed = [[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]]
    repair = [[-x for x in vector] for vector in VERTICES]
    pairs = [
        {"vertex_index": i, "direction_index": j}
        for i, vertex in enumerate(VERTICES)
        for j, direction in enumerate(flawed)
        if module._weak(vertex, direction) and not module._strict(vertex, direction)
    ]
    return {
        "flawed_directions": flawed,
        "weak_false_positive_pairs": pairs,
        "repair_directions": repair,
        "vertex_to_direction": list(range(8)),
    }


def test_oracle_mathematics() -> None:
    module = _module()
    assert module.mathematics(_result(module))


def test_rejects_zero_vector_and_missing_false_positive() -> None:
    module = _module()
    result = _result(module)
    result["flawed_directions"][0] = [0, 0, 0]
    assert not module.mathematics(result)
    result = _result(module)
    result["weak_false_positive_pairs"] = result["weak_false_positive_pairs"][:-1]
    assert not module.mathematics(result)


def test_rejects_noninteger_false_positive_indices() -> None:
    module = _module()
    for replacement in (False, 0.0):
        result = _result(module)
        result["weak_false_positive_pairs"][0]["vertex_index"] = replacement
        assert not module.mathematics(result)


def test_accepts_reordered_repair() -> None:
    module = _module()
    result = _result(module)
    result["repair_directions"].reverse()
    result["vertex_to_direction"].reverse()
    assert module.mathematics(result)


def test_json_comparison_is_type_strict() -> None:
    module = _module()
    assert not module._json_equal({"count": 1}, {"count": True})
    assert not module._json_equal({"count": 2}, {"count": 2.0})
