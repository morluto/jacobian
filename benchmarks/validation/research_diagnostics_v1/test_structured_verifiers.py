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
    assert result == expected


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
    assert rejected["reward"] == 0.0
    if attack == "false-verified":
        assert rejected["false_certification"] is True


def test_structured_verifier_scores_math_separately_from_protocol_compliance(
    tmp_path: Path,
) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-019")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["unexpected"] = True
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 1.0
    assert result["protocol_compliance"] == 0.0
    assert result["reward"] == 0.0


def test_nullstellensatz_verifier_rejects_nonregular_workspace_input(
    tmp_path: Path,
) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-019")
    input_path = app / "input.json"
    input_path.unlink()
    os.mkfifo(input_path)

    result = support.run_verifier(task, app, logs)

    assert result["protocol_compliance"] == 0.0
    assert result["reward"] == 0.0


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

    assert result["reward"] == 1.0


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

    assert result["reward"] == 0.0


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
    assert result["reward"] == 1.0


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
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_graph_verifier_rejects_boolean_summary_scalar(tmp_path: Path) -> None:
    task, app, logs = support.prepare_case(tmp_path, "jcb-postdoc-004")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["local_average"]["denominator"] = True
    support.write_json(submission_path, submission)

    result = support.run_verifier(task, app, logs)

    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


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

    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


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

    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


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
    assert result["reward"] == 1.0


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
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


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

    assert result["reward"] == 0.0
