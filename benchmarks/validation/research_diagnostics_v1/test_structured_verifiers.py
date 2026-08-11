from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from benchmarks.validation.research_diagnostics_v1 import support
from jsonschema import Draft202012Validator

TASKS = tuple(support.TASK_EVIDENCE)


@pytest.mark.parametrize("task_name", TASKS)
def test_structured_oracle_receives_full_reward(
    tmp_path: Path,
    task_name: str,
) -> None:
    result = support.run_verifier(*support.prepare_case(tmp_path, task_name))

    expected = {
        "correctness": 1.0,
        "evidence_validity": 1.0,
        "scope_accuracy": 1.0,
        "assurance_calibration": 1.0,
        "reward": 1.0,
        "false_certification": False,
    }
    if task_name == "jcb-postdoc-019":
        expected["protocol_compliance"] = 1.0
    if task_name in {"jcb-postdoc-015", "jcb-postdoc-016"}:
        expected["limitation_accuracy"] = 1.0
    assert result.reward == expected.pop("reward")
    assert result.details == expected


@pytest.mark.parametrize("task_name", TASKS)
def test_oracle_submission_and_certificate_match_agent_visible_schemas(
    task_name: str,
) -> None:
    task = support.DATASET / task_name
    submission_schema = json.loads(
        (task / "environment" / "submission_schema.json").read_text()
    )
    certificate_schema = json.loads(
        (task / "environment" / support.TASK_EVIDENCE_SCHEMAS[task_name]).read_text()
    )
    submission = json.loads((task / "solution" / "submission.json").read_text())
    certificate = json.loads(
        (task / "solution" / support.TASK_EVIDENCE[task_name]).read_text()
    )
    certificate_path = task / "solution" / support.TASK_EVIDENCE[task_name]

    Draft202012Validator(submission_schema).validate(submission)
    Draft202012Validator(certificate_schema).validate(certificate)
    assert submission["evidence"] == [
        {
            "path": f"evidence/{support.TASK_EVIDENCE[task_name]}",
            "sha256": support.digest(certificate_path),
        }
    ]


@pytest.mark.parametrize("task_name", TASKS)
@pytest.mark.parametrize(
    "attack",
    (
        "malformed-output",
        "missing-output",
        "unknown-field",
        "wrong-result",
        "wrong-scope",
        "wrong-digest",
        "escaped-evidence",
        "missing-evidence",
        "malformed-evidence",
        "symlink-evidence",
        "false-verified",
    ),
)
def test_structured_verifiers_fail_closed_on_protocol_attacks(
    tmp_path: Path,
    task_name: str,
    attack: str,
) -> None:
    task, app, logs = support.prepare_case(tmp_path, task_name)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / support.TASK_EVIDENCE[task_name]

    def rewrite_submission(update):
        update(submission)
        support.write_json(submission_path, submission)

    def malformed_evidence():
        evidence_path.write_text("{", encoding="utf-8")
        submission["evidence"][0]["sha256"] = support.digest(evidence_path)
        support.write_json(submission_path, submission)

    def symlink_evidence():
        target = evidence_path.with_name("target.json")
        evidence_path.rename(target)
        evidence_path.symlink_to(target.name)

    attack_handlers = {
        "malformed-output": lambda: submission_path.write_text("{", encoding="utf-8"),
        "missing-output": submission_path.unlink,
        "unknown-field": lambda: rewrite_submission(
            lambda value: value.update(unexpected=True)
        ),
        "wrong-result": lambda: rewrite_submission(
            lambda value: value.update(result={})
        ),
        "wrong-scope": lambda: rewrite_submission(
            lambda value: value.update(scope="incomplete")
        ),
        "wrong-digest": lambda: rewrite_submission(
            lambda value: value["evidence"][0].update(sha256="sha256:" + "0" * 64)
        ),
        "escaped-evidence": lambda: rewrite_submission(
            lambda value: value["evidence"][0].update(path="../certificate.json")
        ),
        "missing-evidence": evidence_path.unlink,
        "malformed-evidence": malformed_evidence,
        "symlink-evidence": symlink_evidence,
        "false-verified": lambda: rewrite_submission(
            lambda value: value.update(claimed_assurance="VERIFIED")
        ),
    }
    attack_handlers[attack]()

    rejected = support.run_verifier(task, app, logs)
    assert rejected.reward == 0.0
    if attack == "false-verified":
        assert rejected.details["false_certification"] is True


def test_structured_verifier_scores_math_separately_from_protocol_compliance(
    tmp_path: Path,
) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-019")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["unexpected"] = True
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 1.0
    assert result.details["scope_accuracy"] == 1.0
    assert result.details["assurance_calibration"] == 1.0
    assert result.details["protocol_compliance"] == 0.0
    assert result.reward == 0.0


def test_nullstellensatz_verifier_rejects_nonregular_workspace_input(
    tmp_path: Path,
) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-019")
    input_path = app / "input.json"
    input_path.unlink()
    os.mkfifo(input_path)

    result = support.run_verifier(task, app, logs)

    assert result.details["protocol_compliance"] == 0.0
    assert result.reward == 0.0


def test_nullstellensatz_verifier_accepts_reordered_polynomial_terms(
    tmp_path: Path,
) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-019")
    certificate_path = app / "evidence" / "nullstellensatz-certificate.json"
    certificate = json.loads(certificate_path.read_text())
    for chart in certificate["charts"]:
        for named in chart["generators"]:
            named["polynomial"]["terms"] = list(reversed(named["polynomial"]["terms"]))
        for multiplier in chart["multipliers"]:
            multiplier["multiplier"]["terms"] = list(
                reversed(multiplier["multiplier"]["terms"])
            )
    support.write_json(certificate_path, certificate)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(certificate_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.reward == 1.0


def test_syzygy_verifier_rejects_mutated_workspace_input(tmp_path: Path) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-014")
    input_path = app / "input.json"
    input_data = json.loads(input_path.read_text())
    input_data["instance"]["linear_factors"]["f"][0] = [
        2 * coefficient
        for coefficient in input_data["instance"]["linear_factors"]["f"][0]
    ]
    support.write_json(input_path, input_data)

    result = support.run_verifier(task, app, logs)

    assert result.reward == 0.0


def test_graph_verifier_accepts_a_relabelled_counterexample(tmp_path: Path) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-004")
    certificate_path = app / "evidence" / "counterexample.json"
    certificate = json.loads(certificate_path.read_text())
    n = certificate["vertex_count"]
    relabel = {vertex: n - vertex - 1 for vertex in range(n)}
    certificate["edges"] = sorted(
        sorted((relabel[left], relabel[right])) for left, right in certificate["edges"]
    )
    local = certificate["claims"]["neighborhood_independence"]
    relabelled_local = [0] * n
    for old_vertex, value in enumerate(local):
        relabelled_local[relabel[old_vertex]] = value
    certificate["claims"]["neighborhood_independence"] = relabelled_local
    support.write_json(certificate_path, certificate)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(certificate_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)
    assert result.reward == 1.0


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong-alpha",
        "loop",
        "too-many-vertices",
        "hamiltonian-claim",
    ),
)
def test_graph_verifier_rejects_corrupted_witnesses(
    tmp_path: Path,
    mutation: str,
) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-004")
    certificate_path = app / "evidence" / "counterexample.json"
    certificate = json.loads(certificate_path.read_text())
    if mutation == "wrong-alpha":
        certificate["claims"]["independence_number"] += 1
    elif mutation == "loop":
        certificate["edges"][0] = [0, 0]
    elif mutation == "too-many-vertices":
        certificate["vertex_count"] = 21
    else:
        certificate["claims"]["hamiltonian_path_exists"] = True
    support.write_json(certificate_path, certificate)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(certificate_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_graph_verifier_rejects_boolean_summary_scalar(tmp_path: Path) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-004")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["local_average"]["denominator"] = True
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize(
    "mutation",
    ("missing-difference", "wrong-candidate-count", "false-negative-decision"),
)
def test_sidon_extension_verifier_rejects_corrupted_finite_core(
    tmp_path: Path,
    mutation: str,
) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-015")
    evidence_path = app / "evidence" / "finite-core.json"
    evidence = json.loads(evidence_path.read_text())
    if mutation == "missing-difference":
        evidence["ordered_differences"].pop()
    elif mutation == "wrong-candidate-count":
        evidence["fixed_order_checks"][1]["candidate_space_size"] = 25
    else:
        evidence["fixed_order_checks"][2]["decision"] = "EXTENDS"
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_sidon_extension_rejects_unhashable_target_order(tmp_path: Path) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-015")
    evidence_path = app / "evidence" / "finite-core.json"
    evidence = json.loads(evidence_path.read_text())
    for check in evidence["fixed_order_checks"]:
        check["target_order"] = [check["target_order"]]
    support.write_json(evidence_path, evidence)
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(app / "submission.json", submission)

    result = support.run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize(
    "mutation",
    ("wrong-factor-power", "missing-value", "wrong-triple-witness"),
)
def test_powerful_window_verifier_rejects_corrupted_finite_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-016")
    evidence_path = app / "evidence" / "powerful-window.json"
    evidence = json.loads(evidence_path.read_text())
    if mutation == "wrong-factor-power":
        evidence["values"][2]["factors"][0]["power"] = 2
    elif mutation == "missing-value":
        evidence["values"].pop()
    else:
        evidence["triple_checks"][0]["non_powerful_witnesses"] = []
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize("field", ("power", "start"))
def test_powerful_window_rejects_boolean_integer_fields(
    tmp_path: Path, field: str
) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-016")
    evidence_path = app / "evidence" / "powerful-window.json"
    evidence = json.loads(evidence_path.read_text())
    if field == "power":
        evidence["values"][0]["factors"][0]["power"] = True
    else:
        evidence["triple_checks"][0]["start"] = True
    support.write_json(evidence_path, evidence)
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(app / "submission.json", submission)

    result = support.run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_powerful_window_rejects_overflowing_json_number(tmp_path: Path) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-016")
    evidence_path = app / "evidence" / "powerful-window.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["triple_checks"][0]["start"] = float("inf")
    support.write_json(evidence_path, evidence)
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(app / "submission.json", submission)

    result = support.run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_powerful_window_scope_survives_extra_envelope_field(tmp_path: Path) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-016")
    submission = json.loads((app / "submission.json").read_text())
    submission["unexpected"] = True
    support.write_json(app / "submission.json", submission)

    result = support.run_verifier(task, app, logs)
    assert result.details["scope_accuracy"] == 1.0
    assert result.reward == 0.0


@pytest.mark.parametrize("task_name", ("jcb-postdoc-015", "jcb-postdoc-016"))
def test_structured_verifier_marks_malformed_evidence_invalid(
    tmp_path: Path,
    task_name: str,
) -> None:
    """A digest-correct but structurally empty evidence object is an evidence
    failure, not a wrong mathematical answer."""

    task, app, logs = support.prepare_case(tmp_path, task_name)
    evidence_path = app / "evidence" / support.TASK_EVIDENCE[task_name]
    evidence_path.write_text("{}", encoding="utf-8")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["evidence_validity"] == 0.0
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize("task_name", ("jcb-postdoc-015", "jcb-postdoc-016"))
def test_structured_verifier_preserves_correctness_under_protocol_failure(
    tmp_path: Path,
    task_name: str,
) -> None:
    """A protocol failure (false VERIFIED) must zero reward but must not
    collapse the independently correct finite mathematics into a math failure."""

    task, app, logs = support.prepare_case(tmp_path, task_name)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 1.0
    assert result.details["false_certification"] is True
    assert result.reward == 0.0


def test_sidon_extension_verifier_accepts_reordered_differences(tmp_path: Path) -> None:
    """A complete but differently ordered difference profile is equivalent."""

    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-015")
    evidence_path = app / "evidence" / "finite-core.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["ordered_differences"] = list(reversed(evidence["ordered_differences"]))
    evidence["fixed_order_checks"] = list(reversed(evidence["fixed_order_checks"]))
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.reward == 1.0


def test_powerful_window_verifier_accepts_reordered_rows(tmp_path: Path) -> None:
    """Complete but differently ordered value and triple rows are equivalent."""

    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-016")
    evidence_path = app / "evidence" / "powerful-window.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["values"] = list(reversed(evidence["values"]))
    evidence["triple_checks"] = list(reversed(evidence["triple_checks"]))
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.reward == 1.0


def test_powerful_window_verifier_accepts_reordered_factor_collections(
    tmp_path: Path,
) -> None:
    """Reversing the factor list or violating_primes list for a value row
    is mathematically equivalent; the instruction and schema do not
    prescribe an ordering for these collections."""

    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-016")
    evidence_path = app / "evidence" / "powerful-window.json"
    evidence = json.loads(evidence_path.read_text())
    for row in evidence["values"]:
        row["factors"] = list(reversed(row["factors"]))
        row["violating_primes"] = list(reversed(row["violating_primes"]))
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["evidence_validity"] == 1.0
    assert result.reward == 1.0


def test_powerful_window_verifier_rejects_duplicate_factors(tmp_path: Path) -> None:
    """Duplicate factor entries for the same prime are not a valid
    factorization even if the collection is otherwise correct."""

    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-016")
    evidence_path = app / "evidence" / "powerful-window.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["values"][0]["factors"].append(evidence["values"][0]["factors"][0])
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_sidon_extension_verifier_accepts_reordered_base_residues(
    tmp_path: Path,
) -> None:
    """Reversing the base_residues list in a fixed-order check is
    mathematically equivalent; the schema requires only unique integer
    items and the instruction treats residues as a set."""

    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-015")
    evidence_path = app / "evidence" / "finite-core.json"
    evidence = json.loads(evidence_path.read_text())
    for check in evidence["fixed_order_checks"]:
        check["base_residues"] = list(reversed(check["base_residues"]))
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["evidence_validity"] == 1.0
    assert result.reward == 1.0


def test_sidon_extension_verifier_rejects_duplicate_base_residues(
    tmp_path: Path,
) -> None:
    """Duplicate base residues are not a valid set even if the sorted
    presentation would match."""

    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-015")
    evidence_path = app / "evidence" / "finite-core.json"
    evidence = json.loads(evidence_path.read_text())
    residues = evidence["fixed_order_checks"][0]["base_residues"]
    evidence["fixed_order_checks"][0]["base_residues"] = residues + residues
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize("task_name", ("jcb-postdoc-015", "jcb-postdoc-016"))
def test_structured_verifier_rejects_duplicate_evidence_rows(
    tmp_path: Path,
    task_name: str,
) -> None:
    """Duplicate rows that pad the array to the expected length are rejected."""

    task, app, logs = support.prepare_case(tmp_path, task_name)
    evidence_path = app / "evidence" / support.TASK_EVIDENCE[task_name]
    evidence = json.loads(evidence_path.read_text())
    if task_name == "jcb-postdoc-015":
        evidence["ordered_differences"][1] = evidence["ordered_differences"][0]
    else:
        evidence["values"][1] = evidence["values"][0]
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["evidence_validity"] == 0.0
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_sidon_verifier_rejects_unhashable_target_orders(tmp_path: Path) -> None:
    """A list-valued target_order must not crash the duplicate-check set."""

    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-015")
    evidence_path = app / "evidence" / "finite-core.json"
    evidence = json.loads(evidence_path.read_text())
    for check in evidence["fixed_order_checks"]:
        check["target_order"] = []
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["evidence_validity"] == 0.0
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_sidon_verifier_rejects_bool_candidate_space_size(tmp_path: Path) -> None:
    """A boolean where an integer is required must not pass as 1."""

    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-015")
    evidence_path = app / "evidence" / "finite-core.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["fixed_order_checks"][0]["candidate_space_size"] = True
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["evidence_validity"] == 0.0
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_powerful_window_verifier_rejects_bool_factor_power(tmp_path: Path) -> None:
    """A boolean where a factor power is required must not pass as 1."""

    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-016")
    evidence_path = app / "evidence" / "powerful-window.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["values"][2]["factors"][0]["power"] = True
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["evidence_validity"] == 0.0
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_powerful_window_verifier_rejects_bool_all_powerful(tmp_path: Path) -> None:
    """A boolean 0 where all_powerful False is expected must not pass."""

    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-016")
    evidence_path = app / "evidence" / "powerful-window.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["triple_checks"][0]["all_powerful"] = 0
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["evidence_validity"] == 0.0
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize("task_name", ("jcb-postdoc-015", "jcb-postdoc-016"))
def test_structured_verifier_rejects_overflowing_json_numbers(
    tmp_path: Path,
    task_name: str,
) -> None:
    """A float infinity where a string integer is expected must not crash."""

    task, app, logs = support.prepare_case(tmp_path, task_name)
    evidence_path = app / "evidence" / support.TASK_EVIDENCE[task_name]
    evidence = json.loads(evidence_path.read_text())
    if task_name == "jcb-postdoc-015":
        evidence["ordered_differences"][0]["minuend"] = 1e309
    else:
        evidence["triple_checks"][0]["start"] = 1e309
    support.write_json(evidence_path, evidence)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(evidence_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["evidence_validity"] == 0.0
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize("task_name", ("jcb-postdoc-015", "jcb-postdoc-016"))
def test_structured_verifier_scores_scope_independently_of_envelope(
    tmp_path: Path,
    task_name: str,
) -> None:
    """An extra envelope field zeros reward via contract but scope remains 1.0."""

    task, app, logs = support.prepare_case(tmp_path, task_name)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["unexpected"] = True
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.details["scope_accuracy"] == 1.0
    assert result.details["assurance_calibration"] == 1.0
    assert result.details["limitation_accuracy"] == 1.0
    assert result.reward == 0.0


def test_syzygy_verifier_accepts_scaled_relations(tmp_path: Path) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-014")
    certificate_path = app / "evidence" / "syzygy-certificate.json"
    certificate = json.loads(certificate_path.read_text())
    for relation in certificate["relations"].values():
        for name in ("A", "B", "C"):
            for term in relation[name]:
                term["coefficient"] *= -3
    support.write_json(certificate_path, certificate)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(certificate_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)
    assert result.reward == 1.0


@pytest.mark.parametrize(
    "mutation",
    ("wrong-flat", "wrong-coefficient", "zero-relation", "wrong-degree"),
)
def test_syzygy_verifier_rejects_corrupted_certificates(
    tmp_path: Path,
    mutation: str,
) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-014")
    certificate_path = app / "evidence" / "syzygy-certificate.json"
    certificate = json.loads(certificate_path.read_text())
    if mutation == "wrong-flat":
        certificate["non_double_flats"][0] = [1, 2, 4]
    elif mutation == "wrong-coefficient":
        certificate["relations"]["f"]["A"][0]["coefficient"] += 1
    elif mutation == "zero-relation":
        relation = certificate["relations"]["f"]
        relation["A"] = []
        relation["B"] = []
        relation["C"] = []
    else:
        certificate["relations"]["f"]["degree"] = 5
    support.write_json(certificate_path, certificate)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(certificate_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong-generator",
        "reordered-variables",
        "altered-domain",
        "truncated-multiplier",
        "missing-term",
        "incorrect-constant",
        "missing-chart",
    ),
)
def test_nullstellensatz_verifier_rejects_corrupted_certificates(
    tmp_path: Path,
    mutation: str,
) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-019")
    certificate_path = app / "evidence" / "nullstellensatz-certificate.json"
    certificate = json.loads(certificate_path.read_text())
    first = certificate["charts"][0]
    if mutation == "wrong-generator":
        first["generators"][0]["polynomial"]["terms"][0]["coefficient"]["num"] = "7"
    elif mutation == "reordered-variables":
        first["variable_order"][0:2] = reversed(first["variable_order"][0:2])
    elif mutation == "altered-domain":
        certificate["coefficient_domain"] = "RR"
    elif mutation == "truncated-multiplier":
        first["multipliers"].pop()
    elif mutation == "missing-term":
        next(
            item["multiplier"]["terms"]
            for item in first["multipliers"]
            if item["multiplier"]["terms"]
        ).pop()
    elif mutation == "incorrect-constant":
        first["identity_rhs"] = {"num": "2", "den": "1"}
    else:
        certificate["charts"].pop()
    support.write_json(certificate_path, certificate)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support.digest(certificate_path)
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result.reward == 0.0
