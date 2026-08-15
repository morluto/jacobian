from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "rp2-homology-lattice"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _fixtures._write_json(app / "submission.json", submission)


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_alternative_tree_and_orders_pass(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    result["spanning_tree"] = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    result["non_tree_edges"] = [
        [0, 5],
        [3, 5],
        [2, 5],
        [2, 4],
        [1, 5],
        [1, 4],
        [1, 3],
        [0, 4],
        [0, 3],
        [0, 2],
    ]
    result["facet_order"].reverse()
    facets = [tuple(item) for item in result["facet_order"]]
    result["cycle_coordinate_matrix"] = [
        [{(b, c): 1, (a, c): -1, (a, b): 1}.get(tuple(edge), 0) for a, b, c in facets]
        for edge in result["non_tree_edges"]
    ]
    result["determinant"] = -2
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_corrupted_coordinate_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["cycle_coordinate_matrix"][0][0] = 0
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_non_tree_edge_omission_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["non_tree_edges"][0] = [0, 1]
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_forged_determinant_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["determinant"] = 4
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


@pytest.mark.parametrize(
    "field",
    ["spanning_tree", "non_tree_edges", "facet_order", "cycle_coordinate_matrix"],
)
def test_malformed_result_collections_are_cleanly_rejected(
    tmp_path: Path, field: str
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"][field] = None
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_out_of_domain_vertex_is_cleanly_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["spanning_tree"][0] = [0, 6]
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_transposed_coordinate_matrix_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    matrix = submission["result"]["cycle_coordinate_matrix"]
    transposed = [list(column) for column in zip(*matrix, strict=True)]
    submission["result"]["cycle_coordinate_matrix"] = transposed
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_input_tampering_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _rewrite(app, submission)
    (app / "input.json").write_text("{}")
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0
