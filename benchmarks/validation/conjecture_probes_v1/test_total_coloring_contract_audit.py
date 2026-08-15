from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/total-coloring-contract-audit"


def _module():
    saved_path = sys.path[:]
    saved_modules = dict(sys.modules)
    try:
        sys.path.insert(0, str(TASK / "tests"))
        spec = importlib.util.spec_from_file_location(
            "total_coloring_verifier", TASK / "tests/verifier.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        sys.modules.clear()
        sys.modules.update(saved_modules)


def test_oracle_mathematics():
    module = _module()
    vertices = [0, 1, 0, 1, 2, 1, 2, 3, 3, 1]
    repaired = [2, 3, 2, 3, 1, 0, 2, 3, 1, 2, 3, 0, 1, 0, 0]
    flawed = [(c + 1) % 4 for c in repaired]
    result = {
        "flawed_pass": {"vertex_colors": vertices, "edge_colors": flawed},
        "incidence_collisions": module._collisions(vertices, flawed),
        "repair": {"vertex_colors": vertices, "edge_colors": repaired},
    }
    assert module.mathematics(result)


def test_rejects_missing_collision():
    module = _module()
    vertices = [0, 1, 0, 1, 2, 1, 2, 3, 3, 1]
    repaired = [2, 3, 2, 3, 1, 0, 2, 3, 1, 2, 3, 0, 1, 0, 0]
    flawed = [(c + 1) % 4 for c in repaired]
    result = {
        "flawed_pass": {"vertex_colors": vertices, "edge_colors": flawed},
        "incidence_collisions": module._collisions(vertices, flawed)[:-1],
        "repair": {"vertex_colors": vertices, "edge_colors": repaired},
    }
    assert not module.mathematics(result)


def test_rejects_incidence_collision_in_repair():
    module = _module()
    vertices = [0, 1, 0, 1, 2, 1, 2, 3, 3, 1]
    repaired = [2, 3, 2, 3, 1, 0, 2, 3, 1, 2, 3, 0, 1, 0, 0]
    flawed = [(c + 1) % 4 for c in repaired]
    result = {
        "flawed_pass": {"vertex_colors": vertices, "edge_colors": flawed},
        "incidence_collisions": module._collisions(vertices, flawed),
        "repair": {"vertex_colors": vertices, "edge_colors": flawed},
    }
    assert not module.mathematics(result)


def test_accepts_color_permutation():
    module = _module()
    vertices = [0, 1, 0, 1, 2, 1, 2, 3, 3, 1]
    repaired = [2, 3, 2, 3, 1, 0, 2, 3, 1, 2, 3, 0, 1, 0, 0]
    permutation = {0: 2, 1: 0, 2: 3, 3: 1}
    rv = [permutation[c] for c in vertices]
    re = [permutation[c] for c in repaired]
    flawed = [(c + 1) % 4 for c in repaired]
    result = {
        "flawed_pass": {"vertex_colors": vertices, "edge_colors": flawed},
        "incidence_collisions": module._collisions(vertices, flawed),
        "repair": {"vertex_colors": rv, "edge_colors": re},
    }
    assert module.mathematics(result)


def test_accepts_complete_collision_list_in_alternate_order():
    module = _module()
    vertices = [0, 1, 0, 1, 2, 1, 2, 3, 3, 1]
    repaired = [2, 3, 2, 3, 1, 0, 2, 3, 1, 2, 3, 0, 1, 0, 0]
    flawed = [(c + 1) % 4 for c in repaired]
    collisions = sorted(
        module._collisions(vertices, flawed),
        key=lambda row: (row["vertex"], row["edge_index"]),
    )
    result = {
        "flawed_pass": {"vertex_colors": vertices, "edge_colors": flawed},
        "incidence_collisions": collisions,
        "repair": {"vertex_colors": vertices, "edge_colors": repaired},
    }
    assert module.mathematics(result)


def test_rejects_non_integer_collision_diagnostic():
    module = _module()
    vertices = [0, 1, 0, 1, 2, 1, 2, 3, 3, 1]
    repaired = [2, 3, 2, 3, 1, 0, 2, 3, 1, 2, 3, 0, 1, 0, 0]
    flawed = [(c + 1) % 4 for c in repaired]
    collisions = module._collisions(vertices, flawed)
    collisions[0]["edge_index"] = float(collisions[0]["edge_index"])
    result = {
        "flawed_pass": {"vertex_colors": vertices, "edge_colors": flawed},
        "incidence_collisions": collisions,
        "repair": {"vertex_colors": vertices, "edge_colors": repaired},
    }
    assert not module.mathematics(result)


def test_evidence_comparison_preserves_json_types():
    module = _module()
    assert not module._json_equal({"color": 0}, {"color": False})
    assert not module._json_equal({"color": 1}, {"color": 1.0})
