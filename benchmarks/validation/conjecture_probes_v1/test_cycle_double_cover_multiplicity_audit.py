from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
TASK = (
    ROOT
    / "benchmarks/datasets/conjecture-probes-v1/cycle-double-cover-multiplicity-audit"
)
CYCLES = [
    [0, 1, 2, 3, 4],
    [0, 1, 6, 8, 5],
    [1, 2, 7, 9, 6],
    [2, 3, 8, 5, 7],
    [3, 4, 9, 6, 8],
    [0, 4, 9, 7, 5],
]


def _module():
    saved_path = sys.path[:]
    saved_modules = dict(sys.modules)
    try:
        sys.path.insert(0, str(TASK / "tests"))
        spec = importlib.util.spec_from_file_location(
            "cycle_double_cover_verifier", TASK / "tests/verifier.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        sys.modules.clear()
        sys.modules.update(saved_modules)


def _result(module):
    repair = [cycle.copy() for cycle in CYCLES]
    flawed = [cycle.copy() for cycle in repair[:-1]]
    flawed_counts = module._multiplicities(flawed)
    repair_counts = module._multiplicities(repair)
    assert flawed_counts and repair_counts
    return {
        "flawed_cycles": flawed,
        "flawed_multiplicities": flawed_counts,
        "non_double_edge_indices": [i for i, c in enumerate(flawed_counts) if c != 2],
        "repair_cycles": repair,
        "repair_multiplicities": repair_counts,
    }


def test_oracle_mathematics() -> None:
    module = _module()
    assert module.mathematics(_result(module))


def test_rejects_union_only_as_repair() -> None:
    module = _module()
    result = _result(module)
    result["repair_cycles"] = result["flawed_cycles"]
    result["repair_multiplicities"] = result["flawed_multiplicities"]
    assert not module.mathematics(result)


def test_rejects_omitted_bad_edge_and_noninteger_multiplicity() -> None:
    module = _module()
    result = _result(module)
    result["non_double_edge_indices"] = result["non_double_edge_indices"][:-1]
    assert not module.mathematics(result)
    result = _result(module)
    result["flawed_multiplicities"][0] = float(result["flawed_multiplicities"][0])
    assert not module.mathematics(result)


def test_accepts_rotations_and_reversal() -> None:
    module = _module()
    result = _result(module)
    result["repair_cycles"] = [cycle[2:] + cycle[:2] for cycle in reversed(CYCLES)]
    assert module.mathematics(result)


def test_rejects_unhashable_cycle_vertex_without_crashing() -> None:
    module = _module()
    result = _result(module)
    result["flawed_cycles"][0][0] = []
    assert not module.mathematics(result)
