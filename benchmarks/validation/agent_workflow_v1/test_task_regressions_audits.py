from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support
from jsonschema import Draft202012Validator, ValidationError


def test_metric_tsp_scope_is_part_of_correctness(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "metric-tsp-proof-repair", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["scope"] = "wrong scope"
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["scope_accuracy"] == 0.0
    assert rejected["reward"] == 0.0


def test_metric_tsp_evidence_rejects_mismatched_result_marker(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "metric-tsp-proof-repair", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "MST Euler shortcut optimal approximation\nRESULT_JSON: {}\n"
    )
    support._bind_result_evidence(app, submission)
    tampered = json.loads(submission_path.read_text())
    tampered["result"]["weights"]["optimal"] = 999
    support._write_json(submission_path, tampered)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_metric_tsp_accepts_factor_two_claim(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "metric-tsp-proof-repair", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["corrected_claim"] = "factor-2 approximation"
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0


def test_divisibility_accepts_schema_valid_integral_numbers(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "divisibility-construction-witness", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"] = {
        key: float(value) for key, value in submission["result"].items()
    }
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0


def test_modular_obstruction_requires_the_certified_modulus(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "modular-cubic-obstruction", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["modulus"] = 14
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_natural_subtraction_schema_requires_both_basis_entries() -> None:
    task = support._task("natural-subtraction-proof-repair")
    schema = json.loads((task / "environment" / "submission_schema.json").read_text())
    submission = json.loads((task / "solution" / "submission.json").read_text())
    submission["result"]["basis_order"] = []
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(submission)

    submission = json.loads((task / "solution" / "submission.json").read_text())
    submission["result"]["multipliers"] = ["1"]
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(submission)


def test_complex_power_sum_accepts_reversed_branch_order(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "complex-power-sum-elimination", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["branches"].reverse()
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (
            ("recurrence", "power_sums", "5", 3),
            {"numerator": 19, "denominator": 1},
        ),
        (
            ("branches", 0, "target", "sqrt17"),
            {"numerator": 3, "denominator": 1},
        ),
        (("branches", 0, "denominator_norms", "s_minus_12"), 31),
        (("branches",), []),
    ],
)
def test_complex_power_sum_rejects_corrupted_certificates(
    tmp_path: Path,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "complex-power-sum-elimination", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    target = submission["result"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_autoformalization_rejects_positive_lean_compile_claim(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "autoformalization-semantic-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "dimension dot product coordinate\n"
        "Both Lean declarations compile.\n"
        "RESULT_JSON: {}\n"
    )
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0


def test_series_domain_audit_accepts_alternative_denominator(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "series-domain-junk-zero", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    result = submission["result"]
    result["reciprocal_denominator"] = 5
    result["real_part"] = "1/5"
    result["general_block_power_exponent"] = {
        "level_coefficient": 4,
        "constant": -1,
    }
    for block in result["blocks"]:
        block["block_sum_power_lower_bound"] = 2 ** (4 * block["level"] - 1)
    (app / "evidence" / "answer.txt").write_text(
        "For q=5 the general dyadic block lower bound is 2^(4k-1), "
        "which proves divergence. "
        "The block sums do not tend to zero. The returned zero is a fallback "
        "artifact, not an analytic-continuation zero.\n"
    )
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_series_domain_audit_rejects_corrupted_general_bound(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "series-domain-junk-zero", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["general_block_power_exponent"]["constant"] = 0
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_series_domain_audit_accepts_double_star_bound_notation(
    tmp_path: Path,
) -> None:
    """Equivalent bound notation such as 2**(4*k-1) is accepted."""
    task, app, logs = support._prepare_case(
        tmp_path, "series-domain-junk-zero", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    result = submission["result"]
    result["reciprocal_denominator"] = 5
    result["real_part"] = "1/5"
    result["general_block_power_exponent"] = {
        "level_coefficient": 4,
        "constant": -1,
    }
    for block in result["blocks"]:
        block["block_sum_power_lower_bound"] = 2 ** (4 * block["level"] - 1)
    (app / "evidence" / "answer.txt").write_text(
        "For q=5 the general dyadic block lower bound is 2**(4*k-1), "
        "which proves divergence. "
        "The block sums do not tend to zero. The returned zero is a fallback "
        "artifact, not an analytic-continuation zero.\n"
    )
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_series_domain_audit_rejects_affirmative_analytic_claim_in_evidence(
    tmp_path: Path,
) -> None:
    """Evidence that affirmatively claims analytic continuation is rejected."""
    task, app, logs = support._prepare_case(
        tmp_path, "series-domain-junk-zero", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    (app / "evidence" / "answer.txt").write_text(
        "For q=3 the general dyadic block lower bound is 2^(2k-1), "
        "which proves divergence. "
        "The block sums do not tend to zero. The returned zero is a fallback "
        "artifact. This verifies the analytic continuation and proves a "
        "genuine zeta zero.\n"
    )
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0


def test_research_status_audit_accepts_equivalent_case_order(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["cases"].reverse()
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_research_status_audit_rejects_wrong_scope_without_base_reward(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["scope"] = "wrong scope"
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["scope_accuracy"] == 0.0
    assert rejected["reward"] == pytest.approx(0.2)


def test_research_status_audit_rejects_escaped_evidence_without_base_reward(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"] = [
        {
            "path": "../answer.txt",
            "sha256": submission["evidence"][0]["sha256"],
        }
    ]
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == pytest.approx(0.2)


def test_research_status_audit_rejects_wrong_assurance_without_base_reward(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["claimed_assurance"] = "UNVERIFIED"
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["assurance_calibration"] == 0.0
    assert rejected["reward"] == pytest.approx(0.2)


def test_research_status_audit_rejects_tampered_frozen_input(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    (app / "input.json").write_text('{"cases":[]}')
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_research_status_audit_rejects_extra_selected_citations(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    for case in submission["result"]["cases"]:
        if case["case_id"] == "binomial-equation-resolution":
            case["selected_evidence_ids"].append("SW1999")
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_research_status_audit_rejects_undeclared_result_field(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["current_status"] = "ALL_FOUR_PROBLEMS_RESOLVED"
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_research_status_audit_rejects_non_string_case_id(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["cases"][0]["case_id"] = ["workshop-equation-status"]
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_research_status_audit_rejects_evidence_without_result_marker(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text(
        "The frozen packets separate direct resolution evidence, "
        "partial-progress evidence, historical openness, and a bare "
        "problem listing. A negative web search or an old open-problem "
        "citation cannot establish present status.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == pytest.approx(0.2)


def test_research_status_audit_rejects_escaped_semantics_in_result_marker(
    tmp_path: Path,
) -> None:
    """The RESULT_JSON: marker carries the classifications, case IDs, and
    selected evidence IDs, which already encode the resolution, partial
    progress, historical openness, and problem-listing terms. A submission
    whose prose supports none of the classifications must not pass the
    semantic_claims checks or earn evidence validity, even though its result
    marker is canonical and its prose is long enough.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text("This is not a problem statement.\nRESULT_JSON: {}\n")
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == pytest.approx(0.2)


def test_research_status_audit_rejects_oversized_evidence(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text("x" * 2_097_152)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


def test_research_status_audit_oversized_evidence_does_not_block_reward_record(
    tmp_path: Path,
) -> None:
    """The evidence byte bound fires before the digest-bound resolution hashes
    the file, so an oversized malformed artifact yields a complete reward.json
    (math still correct, evidence rejected) instead of a verifier timeout.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text("x" * 2_097_152)
    # A deliberately wrong digest proves the size gate, not the hash, rejects.
    submission["evidence"][0]["sha256"] = "sha256:" + "0" * 64
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == pytest.approx(0.2)


def test_research_status_audit_rejects_deeply_nested_submission(
    tmp_path: Path,
) -> None:
    """A deeply nested JSON payload overflows the CPython recursion limit; the
    verifier treats it as a malformed submission and writes reward.json with
    zero reward instead of crashing without a record.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    (app / "submission.json").write_text("[" * 12000 + "1" + "]" * 12000)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_research_status_audit_rejects_checked_assurance_above_ceiling(
    tmp_path: Path,
) -> None:
    """CHECKED is above the task's COMPUTED assurance ceiling, so it is an
    unsupported certification for this task and forces reward to zero rather
    than granting the partial reward a below-ceiling mismatch would earn.
    """
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["claimed_assurance"] = "CHECKED"
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_research_status_audit_rejects_invalid_utf8_evidence(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_bytes(
        b"\xff\xfe resolution partial-progress historical problem listing"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


def test_research_status_audit_requires_exponent_range_inference(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "research-status-evidence-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    for case in submission["result"]["cases"]:
        if case["case_id"] == "lebesgue-nagell-progress":
            case["unsupported_inferences"] = []
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_inverse_distance_audit_accepts_alternative_rational_direction(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "inverse-distance-remainder-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["directional_witnesses"][0] = {
        "direction": [
            {"numerator": 4, "denominator": 5},
            {"numerator": 3, "denominator": 5},
        ],
        "quadratic_coefficient": {"numerator": 23, "denominator": 50},
        "sign": "POSITIVE",
        "normalized_residual_limit": "quadratic_coefficient",
    }
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_lagrangian_projection_audit_accepts_alternative_coefficients(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "lagrangian-projection-proof-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    result = submission["result"]
    result.update(
        P=[["2", "0"], ["0", "2"]],
        W=[["2", "2/5"], ["0", "2"], ["-1", "1"], ["0", "-4/5"]],
        naive_P=[["2", "2/5"], ["-2/5", "2"]],
        naive_Q=[["1", "-1"], ["1", "1"]],
        corrected_first_projection=[["2", "2/5"], ["-2/5", "2"]],
        corrected_second_projection=[["1", "-1"], ["1", "1"]],
    )
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "The nonzero Lagrangian defect mixes the two naive projections; "
        "the corrected coupled identities reconstruct the exact witness "
        "with scaled coefficients P=2I and Q=I.\n"
        "RESULT_JSON: {}\n"
    )
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_lagrangian_projection_audit_rejects_tampered_frozen_input(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "lagrangian-projection-proof-audit", "computed"
    )
    (app / "input.json").write_text(
        '{"frozen_claim":{"standard_symplectic_matrix":[["0","0","1","0"],["0","0","0","1"],["0","0","0","0"],["0","0","0","0"]]}}'
    )
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_lagrangian_projection_audit_rejects_extra_result_field(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "lagrangian-projection-proof-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["extra_field"] = "malicious"
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_lagrangian_projection_audit_rejects_multiple_evidence_descriptors(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "lagrangian-projection-proof-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"] = [
        submission["evidence"][0],
        submission["evidence"][0],
    ]
    support._write_json(submission_path, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


def test_lagrangian_projection_audit_rejects_oversized_evidence(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "lagrangian-projection-proof-audit", "computed"
    )
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text("x" * 2_097_152)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


def test_lagrangian_projection_audit_rejects_evidence_without_result_marker(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "lagrangian-projection-proof-audit", "computed"
    )
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text(
        "The nonzero Lagrangian defect mixes the two naive projections; "
        "the corrected coupled identities reconstruct the exact witness.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0


def test_continuant_reversal_rejects_missing_symbolic_monomial(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "continuant-reversal-certificate", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["forward_monomials"].pop()
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_continuant_reversal_rejects_corrupted_reflection(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "continuant-reversal-certificate", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["reflection_pairs"][10]["reflected"] = [1]
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (
            ("second_order_term", "dot_square_coefficient"),
            {"numerator": 1, "denominator": 1},
        ),
        (
            ("directional_witnesses", 0, "quadratic_coefficient"),
            {"numerator": 2, "denominator": 1},
        ),
        (
            ("response_audit", "defects"),
            ["CUBIC_REMAINDER_FALSE"],
        ),
    ],
)
def test_inverse_distance_audit_rejects_corrupted_certificates(
    tmp_path: Path,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "inverse-distance-remainder-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    target = submission["result"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


@pytest.mark.parametrize(
    "task_name",
    [
        "calendar-good-days-audit",
        "distinct-sum-pairing-optimum",
        "modular-cubic-obstruction",
        "random-function-expectation-audit",
        "well-total-domination-counterexample",
    ],
)
def test_keyword_only_evidence_is_accepted_with_bound_result(
    tmp_path: Path,
    task_name: str,
) -> None:
    """Keyword-only prose is accepted once RESULT_JSON binds the structured result.

    Replaces the former keyword-gate attack: evidence that mentions none of the
    previously required terms but carries a correct RESULT_JSON marker and has
    non-empty prose should now pass evidence validation, because correctness
    depends on the structured result, not on prose vocabulary.
    """
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("Brief explanation.\nRESULT_JSON: {}\n")
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "task_name",
    [
        "calendar-good-days-audit",
        "distinct-sum-pairing-optimum",
        "modular-cubic-obstruction",
    ],
)
def test_evidence_without_result_marker_is_rejected(
    tmp_path: Path,
    task_name: str,
) -> None:
    """Evidence lacking a RESULT_JSON marker must fail evidence validation."""
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "A complete explanation with all the right ideas but no structured marker.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0


def test_divisibility_evidence_rejects_mismatched_result_marker(
    tmp_path: Path,
) -> None:
    """Divisibility evidence must bind the exact structured result, not just prose."""
    task, app, logs = support._prepare_case(
        tmp_path, "divisibility-construction-witness", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("Brief explanation.\nRESULT_JSON: {}\n")
    support._bind_result_evidence(app, submission)
    tampered = json.loads(submission_path.read_text())
    tampered["result"]["a"] = 999
    support._write_json(submission_path, tampered)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_autoformalization_keyword_only_evidence_is_accepted(
    tmp_path: Path,
) -> None:
    """Keyword-only evidence without the previously required terms is accepted
    once RESULT_JSON binds the structured result and no positive compile claim
    is present."""
    task, app, logs = support._prepare_case(
        tmp_path, "autoformalization-semantic-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("The audit found problems.\nRESULT_JSON: {}\n")
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["evidence_validity"] == 1.0


def test_complex_power_sum_keyword_only_evidence_is_accepted(
    tmp_path: Path,
) -> None:
    """Complex power-sum evidence without the previously required phrases is
    accepted once RESULT_JSON binds the structured result."""
    task, app, logs = support._prepare_case(
        tmp_path, "complex-power-sum-elimination", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("The elimination works.\nRESULT_JSON: {}\n")
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["evidence_validity"] == 1.0


def test_lean_transitive_evidence_rejects_empty_prose(tmp_path: Path) -> None:
    """Lean transitive audit evidence must have non-empty prose."""
    task, app, logs = support._prepare_case(
        tmp_path, "lean-transitive-axiom-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


def test_trigonometric_power_sum_rejects_corrupted_recurrence_term(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "trigonometric-power-sum-valuation", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["terms"][12]["value"] += 7
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_trigonometric_power_sum_rejects_corrupted_induction_case(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "trigonometric-power-sum-valuation", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["induction_cases"][1]["coefficient_adjusted_offsets"] = [
        1,
        1,
        0,
    ]
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_trigonometric_power_sum_rejects_negated_scope(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "trigonometric-power-sum-valuation", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["scope"] = (
        "This certificate does not cover the cubic recurrence or 7-adic induction."
    )
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["scope_accuracy"] == 0.0
    assert rejected["reward"] == 0.0


def test_putnam_2adic_evidence_rejects_empty_prose(tmp_path: Path) -> None:
    """Putnam 2-adic audit evidence must have non-empty prose."""
    task, app, logs = support._prepare_case(
        tmp_path, "putnam-2adic-induction-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


def test_putnam_2adic_evidence_requires_result_binding(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "putnam-2adic-induction-audit", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("Arbitrary nonempty evidence.\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0


def test_infinite_shift_spectrum_accepts_reversed_operator_orientation(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "infinite-shift-spectrum-counterexample", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    result = submission["result"]
    result["orientation"] = "S_LEFT_T_RIGHT"
    result["zero_eigenvalue_product"] = "TS"
    result["identity_product"] = "ST"
    for action in result["actions"]:
        index = action["basis_index"]
        action.update(
            {
                "s_output": None if index == 0 else index - 1,
                "t_output": index + 1,
                "st_output": index,
                "ts_output": None if index == 0 else index,
            }
        )
    support._bind_result_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0


def test_infinite_shift_spectrum_rejects_corrupted_composition(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "infinite-shift-spectrum-counterexample", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["actions"][5]["st_output"] = 4
    support._bind_result_evidence(app, submission)

    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_indexed_pairwise_vacuity_accepts_canonical_case(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "indexed-pairwise-vacuity", "computed"
    )
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_indexed_pairwise_vacuity_rejects_boolean_witness_element(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "indexed-pairwise-vacuity", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["subgroup"] = [False, 4, 8]
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_indexed_pairwise_vacuity_accepts_unordered_elements(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "indexed-pairwise-vacuity", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    result = submission["result"]
    result["subgroup"] = [8, 0, 4]
    result["cosets"][0] = [8, 4, 0]
    result["part_artifact"]["elements"] = [8, 0, 4]
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
