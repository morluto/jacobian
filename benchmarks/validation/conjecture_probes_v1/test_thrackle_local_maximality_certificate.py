from __future__ import annotations

import importlib.util
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parents[3]
TASK = (
    ROOT
    / "benchmarks/datasets/conjecture-probes-v1/thrackle-local-maximality-certificate"
)
SELECTED = [(0, 2), (0, 3), (1, 3), (1, 4), (2, 4)]


def _module():
    saved_path = sys.path[:]
    saved_modules = dict(sys.modules)
    try:
        sys.path.insert(0, str(TASK / "tests"))
        spec = importlib.util.spec_from_file_location(
            "thrackle_verifier", TASK / "tests/verifier.py"
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
    pairs = [
        {"left": list(e), "right": list(f), "relation": module._relation(e, f)}
        for e, f in combinations(SELECTED, 2)
    ]
    excluded = [e for e in module.ALL if e not in SELECTED]
    witnesses = [
        {
            "excluded": list(e),
            "disjoint_selected": list(
                next(f for f in SELECTED if module._relation(e, f) == "DISJOINT")
            ),
        }
        for e in excluded
    ]
    return {
        "selected_edges": [list(e) for e in SELECTED],
        "pair_classifications": pairs,
        "excluded_edge_witnesses": witnesses,
    }


def test_oracle_mathematics():
    module = _module()
    assert module.mathematics(_result(module))


def test_rejects_corrupt_pair_relation():
    module = _module()
    result = _result(module)
    result["pair_classifications"][0]["relation"] = "PROPER_CROSSING"
    assert not module.mathematics(result)


def test_rejects_non_integer_vertices_in_certificates():
    module = _module()
    result = _result(module)
    result["pair_classifications"][0]["left"][0] = 0.0
    assert not module.mathematics(result)

    result = _result(module)
    result["excluded_edge_witnesses"][0]["excluded"][0] = False
    assert not module.mathematics(result)


def test_evidence_result_comparison_preserves_json_types():
    module = _module()
    result = _result(module)
    evidence_result = _result(module)
    evidence_result["selected_edges"][0][0] = False
    assert not module.json_value_equal(evidence_result, result)


def test_rejects_missing_maximality_witness():
    module = _module()
    result = _result(module)
    result["excluded_edge_witnesses"].pop()
    assert not module.mathematics(result)


def test_rejects_non_list_selected_edges_without_crashing():
    module = _module()
    for malformed in (None, 5, {}):
        result = _result(module)
        result["selected_edges"] = malformed
        assert not module.mathematics(result)


def test_rejects_boundary_cycle():
    module = _module()
    result = _result(module)
    result["selected_edges"] = [[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]
    assert not module.mathematics(result)
