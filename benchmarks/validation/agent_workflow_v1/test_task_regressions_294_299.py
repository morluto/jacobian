from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support
from jsonschema import Draft202012Validator


def test_lcm_scope_audit_accepts_alternative_earlier_index(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "lcm-highly-abundant-scope-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    earlier = submission["result"]["witnesses"][1]
    earlier.update(
        {
            "n": 73,
            "lcm_factorization": earlier["lcm_factorization"]
            + [{"prime": 73, "exponent": 1}],
            "lcm_value": 410555180440430163438262940577600,
            "competitor": 409087987237258561004281340832000,
            "sigma_lcm": 3068535475037360330537152020480000,
            "sigma_competitor": 3071037991057009848454773473280000,
        }
    )
    support._write_json(submission_path, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("witnesses", 0, "sigma_competitor"), 1),
        (("witnesses", 1, "n"), 97),
        (("witnesses", 1, "exponent_deltas", 0, "prime"), 4),
        (("minimality_claim",), "CONFIRMED"),
    ],
)
def test_lcm_scope_audit_rejects_corrupted_or_overclaimed_certificates(
    tmp_path: Path,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "lcm-highly-abundant-scope-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    target = submission["result"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_local_density_rejects_boolean_integer_fields(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "dead-end-local-density-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["cases"][1]["density_numerator"] = True
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0


def test_local_density_rejects_duplicate_evidence_descriptors(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "dead-end-local-density-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"].append(dict(submission["evidence"][0]))
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


def test_lean_axiom_fixture_requires_genuine_transitive_closure() -> None:
    task = support._task("lean-transitive-axiom-audit")
    source = json.loads((task / "environment" / "input.json").read_text())
    case = next(
        item for item in source["cases"] if item["case_id"] == "axiom-type-closure"
    )
    assert "A0" not in case["dependencies"]["A2"]
    assert case["dependencies"]["A1"] == ["A0"]


def test_degree_sequence_accepts_reversed_one_based_edges(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "erdos-gallai-realization-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    case = next(
        item for item in submission["result"]["cases"] if item["status"] == "GRAPHICAL"
    )
    case["edges"] = [[v + 1, u + 1] for u, v in case["edges"]]
    support._write_json(submission_path, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_degree_sequence_reference_solution_is_schema_valid() -> None:
    task = support._task("erdos-gallai-realization-audit")
    schema = json.loads((task / "environment" / "submission_schema.json").read_text())
    submission = json.loads((task / "solution" / "submission.json").read_text())
    Draft202012Validator(schema).validate(submission)


def test_fourth_power_scope_rejects_boolean_joint_gcd(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "euler-fourth-power-scope-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["joint_gcd"] = True
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0


def test_fourth_power_scope_accepts_reference_solution(tmp_path: Path) -> None:
    result = support._run_verifier(
        *support._prepare_case(tmp_path, "euler-fourth-power-scope-audit", "computed")
    )
    assert result["correctness"] == 1.0
    assert result["reward"] == pytest.approx(1.0)


def test_dead_end_local_density_audit_accepts_case_reordering(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "dead-end-local-density-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["cases"].reverse()
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_dead_end_local_density_audit_rejects_duplicate_case_id(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "dead-end-local-density-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["cases"][1]["case_id"] = submission["result"]["cases"][0][
        "case_id"
    ]
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_subset_incidence_rejects_boolean_diagonal_weights(
    tmp_path: Path,
) -> None:
    """Thread PRRT_kwDOThEfjc6VuwyR: reject boolean diagonal weights.

    Replacing every 1 in diagonal_weights with JSON true must not pass,
    because True == 1 in Python but booleans violate the enum: [-1, 1]
    contract.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "subset-incidence-determinant", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["diagonal_weights"] = [
        True if w == 1 else w for w in submission["result"]["diagonal_weights"]
    ]
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_subset_incidence_rejects_boolean_mask_order(
    tmp_path: Path,
) -> None:
    """Thread PRRT_kwDOThEfjc6Vu43n: reject booleans in mask_order."""
    task, app, logs = support._prepare_case(
        tmp_path, "subset-incidence-determinant", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["mask_order"][0] = True
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_subset_incidence_rejects_boolean_trace_fields(
    tmp_path: Path,
) -> None:
    """Thread PRRT_kwDOThEfjc6Vu43n: reject booleans in trace numeric fields."""
    task, app, logs = support._prepare_case(
        tmp_path, "subset-incidence-determinant", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["trace"][0]["n"] = True
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_subset_incidence_rejects_boolean_sample_n(
    tmp_path: Path,
) -> None:
    """Thread PRRT_kwDOThEfjc6Vu43n: reject booleans in sample_n."""
    task, app, logs = support._prepare_case(
        tmp_path, "subset-incidence-determinant", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["sample_n"] = True
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_subset_incidence_rejects_duplicate_evidence_descriptors(
    tmp_path: Path,
) -> None:
    """Thread PRRT_kwDOThEfjc6Vu43q: enforce the maxItems: 1 evidence limit.

    Repeating the same evidence descriptor must not pass even though each
    copy individually binds the expected evidence file.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "subset-incidence-determinant", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"].append(dict(submission["evidence"][0]))
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_subset_incidence_accepts_canonical_solution(
    tmp_path: Path,
) -> None:
    """The canonical solution must still earn full reward after the fixes."""
    task, app, logs = support._prepare_case(
        tmp_path, "subset-incidence-determinant", "computed"
    )
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)
