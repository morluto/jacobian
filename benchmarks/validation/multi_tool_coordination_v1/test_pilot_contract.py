from __future__ import annotations

import ast
import json
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import pytest
from benchmarks.tooling.command_runner import ToolCommandStatus, run_operator_command
from benchmarks.validation.multi_tool_coordination_v1 import support

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "benchmarks/datasets/multi-tool-coordination-v1"
MANIFEST = json.loads((DATASET / "pilot-manifest.json").read_text())
TASK_IDS = [case["task_id"] for case in MANIFEST["cases"]]


def canonical_case(tmp_path: Path, task_id: str):
    task, app, logs = support.prepare(tmp_path, task_id)
    submission = json.loads((app / "submission.json").read_text())
    return task, app, logs, submission


def determinant(values: list[list[int]]) -> int:
    matrix = [[Fraction(value) for value in row] for row in values]
    result = Fraction(1)
    for column in range(len(matrix)):
        pivot = next(row for row in range(column, len(matrix)) if matrix[row][column])
        if pivot != column:
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            result = -result
        pivot_value = matrix[column][column]
        result *= pivot_value
        for row in range(column + 1, len(matrix)):
            scale = matrix[row][column] / pivot_value
            for item in range(column, len(matrix)):
                matrix[row][item] -= scale * matrix[column][item]
    assert result.denominator == 1
    return result.numerator


def test_generator_is_deterministic() -> None:
    completed = run_operator_command(
        "uv",
        (
            "run",
            "--locked",
            "python",
            str(DATASET / "generate.py"),
            "--check",
        ),
        cwd=ROOT,
        timeout_seconds=60.0,
    )
    assert completed.status is ToolCommandStatus.EXITED
    assert completed.exit_code == 0, completed.stderr.decode(errors="replace")
    assert b"4 generated cases are current" in completed.stdout


def test_manifest_has_the_pr1_family_balance() -> None:
    assert MANIFEST["case_count"] == 4
    assert len(TASK_IDS) == len(set(TASK_IDS)) == 4
    assert Counter(case["family"] for case in MANIFEST["cases"]) == {
        "graph-set-distance": 1,
        "cycle-lattice": 1,
        "rational-slice-binding": 1,
        "directed-proportionality": 1,
    }


def test_verifier_is_clean_room_and_backend_independent() -> None:
    text = (DATASET / "verifier_template.py").read_text()
    imports = {
        *(
            alias.name.split(".", 1)[0]
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Import)
            for alias in node.names
        ),
        *(
            node.module.split(".", 1)[0]
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ),
    }
    assert not imports & {"jacobian", "sympy", "generate"}
    assert all(
        (DATASET / task_id / "tests/verifier.py").read_text() == text
        for task_id in TASK_IDS
    )


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_every_hidden_oracle_solution_receives_full_reward(
    tmp_path: Path, task_id: str
) -> None:
    task, app, logs = support.prepare(tmp_path, task_id)
    result = support.run_verifier(task, app, logs)
    assert result.reward == 1.0
    assert result.details["correctness"] == 1.0
    assert result.details["input_binding"] == 1.0
    assert result.details["evidence_validity"] == 1.0
    assert result.details["scope_accuracy"] == 1.0
    assert result.details["assurance_calibration"] == 1.0


def test_alternate_cycle_lattice_witness_is_accepted(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "coordination-cycle-lattice-01"
    )
    frozen = json.loads((app / "input.json").read_text())
    facets = sorted((tuple(facet) for facet in frozen["facets"]), reverse=True)
    edges = sorted(
        {edge for facet in facets for edge in combinations(sorted(facet), 2)}
    )
    parent = {vertex: vertex for vertex in frozen["vertices"]}

    def root(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    tree = []
    for left, right in reversed(edges):
        first, second = root(left), root(right)
        if first != second:
            parent[first] = second
            tree.append((left, right))
    non_tree = [edge for edge in edges if edge not in set(tree)]
    matrix = [
        [{(b, c): 1, (a, c): -1, (a, b): 1}.get(edge, 0) for a, b, c in facets]
        for edge in non_tree
    ]
    exact_determinant = determinant(matrix)
    assert abs(exact_determinant) == 2
    submission["result"] = {
        "spanning_tree": [list(edge) for edge in tree],
        "non_tree_edges": [list(edge) for edge in non_tree],
        "facet_order": [list(facet) for facet in facets],
        "cycle_coordinate_matrix": [
            list(column) for column in zip(*matrix, strict=True)
        ],
        "determinant": exact_determinant,
        "homology": "Z/2Z",
    }
    support.rebind_evidence(app, submission)

    assert support.run_verifier(task, app, logs).reward == 1.0


def test_malformed_terminal_object_is_rejected_without_crashing(
    tmp_path: Path,
) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "coordination-graph-set-distance-01"
    )
    submission["result"] = None
    support.rebind_evidence(app, submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.details["protocol_compliance"] == 0.0
    assert result.reward == 0.0


def test_stale_evidence_digest_is_rejected(tmp_path: Path) -> None:
    task, app, logs, _submission = canonical_case(
        tmp_path, "coordination-graph-set-distance-01"
    )
    evidence = app / "evidence/certificate.json"
    evidence.write_bytes(evidence.read_bytes() + b" ")
    result = support.run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_substituted_workspace_input_is_rejected(tmp_path: Path) -> None:
    task, app, logs, _submission = canonical_case(
        tmp_path, "coordination-graph-set-distance-01"
    )
    frozen = json.loads((app / "input.json").read_text())
    frozen["case_id"] += "-substituted"
    support.write_json(app / "input.json", frozen)
    result = support.run_verifier(task, app, logs)
    assert result.details["input_binding"] == 0.0
    assert result.reward == 0.0


def test_scope_escalation_is_rejected(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "coordination-rational-slice-01"
    )
    submission["scope"] = "GLOBAL_POSITIVE_DEFINITENESS_THEOREM"
    submission["limitations"] = []
    support.rebind_evidence(app, submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["scope_accuracy"] == 0.0
    assert result.reward == 0.0


def test_false_verified_claim_is_rejected(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "coordination-rational-slice-01"
    )
    submission["claimed_assurance"] = "VERIFIED"
    support.rebind_evidence(app, submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["false_certification"] is True
    assert result.details["assurance_calibration"] == 0.0
    assert result.reward == 0.0


def test_matrix_entry_cannot_replace_named_rational_scalar(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "coordination-rational-slice-01"
    )
    submission["result"]["scalar_replay"]["m00"] = {"num": "2", "den": "1"}
    support.rebind_evidence(app, submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_inverse_proportionality_direction_is_rejected(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "coordination-directed-proportionality-01"
    )
    result = submission["result"]
    result["circle_coefficients"], result["distance_coefficients"] = (
        result["distance_coefficients"],
        result["circle_coefficients"],
    )
    result["multiplier"] = {"num": "-1", "den": "8"}
    support.rebind_evidence(app, submission)
    verification = support.run_verifier(task, app, logs)
    assert verification.details["correctness"] == 0.0
    assert verification.reward == 0.0


def test_incomplete_graph_extremizer_set_is_rejected(tmp_path: Path) -> None:
    task, app, logs, submission = canonical_case(
        tmp_path, "coordination-graph-set-distance-01"
    )
    submission["result"]["maximizing_vertices"] = []
    support.rebind_evidence(app, submission)
    result = support.run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_task_instructions_preserve_agent_owned_strategy() -> None:
    for task_id in TASK_IDS:
        instruction = (DATASET / task_id / "instruction.md").read_text().lower()
        assert "math.run" not in instruction
        assert "capability." not in instruction
        assert "required tool" not in instruction
        assert "tool sequence" in instruction
        assert "no tool sequence is\nprescribed" in instruction
