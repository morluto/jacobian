from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "positive-lower-density-separation"


def _run(tmp_path: Path, mutate=None):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    if mutate:
        mutate(submission)
        support._bind_result_evidence(app, submission)
        support._write_json(app / "submission.json", submission)
    return support._run_verifier(task, app, logs)


def test_accepts_equivalent_limitation_wording(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        lambda submission: submission.update(
            limitations=[
                "These eight finite levels do not prove the infinite limit or the Erdos problem."
            ]
        ),
    )
    assert result["scope_accuracy"] == 1.0
    assert result["reward"] == 1.0


def _set_base(submission, b):
    levels = []
    for m in range(8):
        high, low = b ** (2 * m + 1), b ** (2 * m + 2)
        count = (low - 1) // (b + 1)
        levels.append(
            {
                "level": m,
                "included_endpoint": high,
                "excluded_endpoint": low,
                "cumulative_count": count,
                "included_density": str(Fraction(count, high)),
                "excluded_density": str(Fraction(count, low)),
            }
        )
    submission["result"].update(
        {
            "base": b,
            "levels": levels,
            "lower_density": str(Fraction(1, b + 1)),
            "upper_density": str(Fraction(b, b + 1)),
        }
    )


def test_oracle_and_alternative_base_pass(tmp_path: Path) -> None:
    assert _run(tmp_path / "oracle")["reward"] == 1.0
    assert _run(tmp_path / "alternative", lambda s: _set_base(s, 7))["reward"] == 1.0


def test_published_evidence_sentence_needs_no_private_marker(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "The lower density is positive, while the two endpoint subsequences have different limits, so the "
        "natural density does not exist. The finite levels replay instances of the general formula rather than "
        "proving every infinite case.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 1.0
    assert result["reward"] == 1.0


def test_visible_input_tamper_preserves_math_diagnostic(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("{}")
    result = support._run_verifier(task, app, logs)
    assert result["input_binding"] == 0.0
    assert result["correctness"] == 1.0
    assert result["reward"] == 0.0


def test_integral_float_level_fields_are_rejected(tmp_path: Path) -> None:
    def mutate(s):
        for row in s["result"]["levels"]:
            row["level"] = float(row["level"])
            row["included_endpoint"] = float(row["included_endpoint"])
            row["excluded_endpoint"] = float(row["excluded_endpoint"])
            row["cumulative_count"] = float(row["cumulative_count"])

    result = _run(tmp_path, mutate)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_level_rows_are_order_independent(tmp_path: Path) -> None:
    result = _run(tmp_path, lambda s: s["result"]["levels"].reverse())
    assert result["correctness"] == 1.0
    assert result["reward"] == 1.0


def test_bad_scope_preserves_math_diagnostic(tmp_path: Path) -> None:
    result = _run(tmp_path, lambda s: s.__setitem__("scope", "wrong"))
    assert result["correctness"] == 1.0
    assert result["scope_accuracy"] == 0.0
    assert result["reward"] == 0.0


def test_rejects_corrupted_endpoint_count(tmp_path: Path) -> None:
    def mutate(s):
        s["result"]["levels"][4]["cumulative_count"] += 1

    assert _run(tmp_path, mutate)["reward"] == 0.0


def test_rejects_equal_subsequence_limits(tmp_path: Path) -> None:
    def mutate(s):
        s["result"]["upper_density"] = s["result"]["lower_density"]

    assert _run(tmp_path, mutate)["reward"] == 0.0


def test_rejects_out_of_scope_base(tmp_path: Path) -> None:
    assert _run(tmp_path, lambda s: _set_base(s, 10))["reward"] == 0.0


def test_rejects_false_verified_claim(tmp_path: Path) -> None:
    def mutate(s):
        s["claimed_assurance"] = "VERIFIED"

    result = _run(tmp_path, mutate)
    assert result["false_certification"] is True
    assert result["reward"] == 0.0


def _set_evidence(app: Path, submission: dict, text: str) -> None:
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(text)
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)


def test_malformed_envelope_preserves_math_correctness(tmp_path: Path) -> None:
    """A schema-invalid non-result field must not erase mathematical correctness."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["extra_field"] = "schema-invalid"
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["protocol_compliance"] == 0.0
    assert result["correctness"] == 1.0
    assert result["reward"] == 0.0


def test_evidence_descriptor_missing_sha256_preserves_correctness(
    tmp_path: Path,
) -> None:
    """A bad evidence descriptor fails evidence but preserves correctness."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0] = {"path": "evidence/answer.txt"}
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_large_valid_evidence_has_no_arbitrary_byte_cap(tmp_path: Path) -> None:
    """An otherwise valid explanation larger than 65536 bytes is accepted."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    _set_evidence(
        app,
        submission,
        "derivation filler\n"
        * 70_000
        + "The lower density is positive, while the two endpoint subsequences have "
        "different limits, so the natural density does not exist. The finite levels "
        "replay instances of the general formula rather than proving every infinite "
        "case.\n",
    )
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 1.0
    assert result["reward"] == 1.0


def test_equivalent_explanation_paraphrase_is_accepted(tmp_path: Path) -> None:
    """Equivalent phrasing of the certified separation is accepted."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    _set_evidence(
        app,
        submission,
        "The lower density is positive, but the two subsequential limits differ, so "
        "no natural density exists. The eight finite cases are instances of the "
        "general formula, not a proof of the infinite limit.\n",
    )
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 1.0
    assert result["reward"] == 1.0


def test_contradictory_explanation_is_rejected(tmp_path: Path) -> None:
    """Text that asserts the opposite of the certified separation is rejected."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    _set_evidence(
        app,
        submission,
        "The natural density exists and the limits agree; the finite levels prove "
        "every infinite case.\n",
    )
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_late_contradiction_is_rejected(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    _set_evidence(
        app,
        submission,
        "The lower density is positive and the subsequential limits differ, so "
        "the natural density does not exist. The finite levels replay instances "
        "of the general formula and are not a proof of the infinite claim.\n"
        + "derivation filler\n" * 70_000
        + "Contrary conclusion: the limits agree.\n",
    )
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_unrelated_explanation_is_rejected(tmp_path: Path) -> None:
    """Nonempty but unrelated text is rejected."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    _set_evidence(app, submission, "The weather is sunny today.\n")
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_empty_explanation_is_rejected(tmp_path: Path) -> None:
    """An empty digest-bound explanation is rejected."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    _set_evidence(app, submission, "\n")
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_input_binding_decoupled_is_declared_in_task_metadata() -> None:
    """The verifier decouples correctness from input binding via task metadata."""
    assert support.is_input_binding_decoupled(TASK) is True
    assert support.is_scope_independent_assurance(TASK) is True
    metadata = support.load_task_contract_metadata(TASK)
    assert metadata["input_binding_decoupled"] is True
    assert metadata["scope_independent_assurance"] is True
