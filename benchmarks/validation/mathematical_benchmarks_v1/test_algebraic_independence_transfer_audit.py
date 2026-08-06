from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "algebraic-independence-transfer-audit"


def _prepare(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _mutate(tmp_path: Path, mutation):
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    mutation(submission)
    support._write_json(app / "submission.json", submission)
    return support._run_verifier(task, app, logs)


def test_oracle_transfer_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["reward"] == 1.0


def test_published_evidence_sentence_needs_no_private_marker(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "The first and second coordinate changes are birational with the displayed "
        "inverse formulas. The conjugate norm is computed exactly over QQ. The "
        "modular-form independence theorem remains a trusted premise.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["evidence_validity"] == 1.0
    assert reward["reward"] == 1.0


def test_visible_input_tamper_preserves_math_diagnostic(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    (app / "input.json").write_text("{}")
    reward = support._run_verifier(task, app, logs)
    assert reward["input_binding"] == 0.0
    assert reward["correctness"] == 1.0
    assert reward["reward"] == 0.0


def test_alternative_term_order_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["norm_polynomial"].reverse()
    evidence = app / "evidence/answer.txt"
    lines = evidence.read_text().splitlines()
    lines[-1] = "RESULT_JSON:" + json.dumps(
        submission["result"], sort_keys=True, separators=(",", ":")
    )
    evidence.write_text("\n".join(lines) + "\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["reward"] == 1.0


def test_corrupted_inverse_coefficient_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"]["d2_delta_numerator"][0].__setitem__("coefficient", "12"),
    )
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_duplicate_monomial_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"]["q_numerator"].append(
            dict(s["result"]["q_numerator"][0])
        ),
    )
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_corrupted_conjugate_norm_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"]["norm_polynomial"][4].__setitem__("coefficient", "2"),
    )
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_noncanonical_rational_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"]["p_numerator"][0].__setitem__("coefficient", "2/2"),
    )
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_false_verified_claim_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(tmp_path, lambda s: s.__setitem__("claimed_assurance", "VERIFIED"))
    assert reward["false_certification"] is True
    assert reward["reward"] == 0.0
