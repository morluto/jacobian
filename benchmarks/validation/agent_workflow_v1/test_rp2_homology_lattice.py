from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "rp2-homology-lattice"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


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
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_corrupted_coordinate_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["cycle_coordinate_matrix"][0][0] = 0
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_non_tree_edge_omission_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["non_tree_edges"][0] = [0, 1]
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_forged_determinant_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["determinant"] = 4
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_false_verified_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 0.0
    assert result["false_certification"] is True


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
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_affirmative_proof_assistant_claim_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = [
        "A proof assistant verifies that the geometric realization is RP2."
    ]
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_out_of_domain_vertex_is_cleanly_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["spanning_tree"][0] = [0, 6]
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_oversized_evidence_is_rejected_without_crashing(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(evidence.read_text() + "\n" + "x" * 65536)
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_affirmative_proof_assistant_in_evidence_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        evidence.read_text() + "\nA proof assistant verified this result."
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_unverified_claim_preserves_other_metric_axes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "UNVERIFIED"
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 0.0
    assert result["reward"] == 0.0


def test_affirmative_claim_with_unrelated_negation_is_rejected(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        evidence.read_text()
        + "\nA proof assistant verified this result, not merely computed it."
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_transposed_coordinate_matrix_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    matrix = submission["result"]["cycle_coordinate_matrix"]
    transposed = [list(column) for column in zip(*matrix, strict=True)]
    submission["result"]["cycle_coordinate_matrix"] = transposed
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_evidence_without_hidden_keyword_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    original = evidence.read_text()
    marker_line = next(
        (line for line in original.splitlines() if line.startswith("RESULT_JSON:")),
        None,
    )
    prose = (
        "The cycle-coordinate matrix has determinant -2, so the quotient "
        "lattice has order 2 and H1 is Z/2Z."
    )
    evidence.write_text(prose + ("\n" + marker_line + "\n" if marker_line else "\n"))
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0
