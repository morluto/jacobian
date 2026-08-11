from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "finite-scheme-rational-points-audit"


def _prepare(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _mutate(tmp_path: Path, mutation):
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    mutation(submission)
    support._write_json(app / "submission.json", submission)
    return support._run_verifier(task, app, logs)


def test_accepts_equivalent_limitation_wording(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda submission: submission.update(
            limitations=[
                "This finite affine F5 countermodel concerns rational points only, not a general scheme theorem."
            ]
        ),
    )
    assert reward.details["scope_accuracy"] == 1.0
    assert reward.reward == 1.0


def test_oracle_countermodel_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    reward = support._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 1.0


def test_published_evidence_sentence_needs_no_private_marker(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "Both affine schemes are nonempty and have the same three rational points under the induced map. "
        "A has a nonzero order-three nilpotent while B is reduced, so they are not isomorphic.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward.details["evidence_validity"] == 1.0
    assert reward.reward == 1.0


def test_visible_input_tamper_preserves_math_diagnostic(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    (app / "input.json").write_text("{}")
    reward = support._run_verifier(task, app, logs)
    assert reward.details["input_binding"] == 0.0
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 0.0


def test_corrupted_multiplication_table_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path, lambda s: s["result"]["a_multiplication"][3][3].__setitem__(4, 0)
    )
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_incomplete_rational_point_enumeration_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(tmp_path, lambda s: s["result"]["a_points"].pop())
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_nonmultiplicative_morphism_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path, lambda s: s["result"]["morphism_columns"][0].__setitem__(3, 1)
    )
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_valid_but_unspecified_morphism_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"].__setitem__(
            "morphism_columns",
            [
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ],
        ),
    )
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_bad_scope_preserves_math_diagnostic(tmp_path: Path) -> None:
    reward = _mutate(tmp_path, lambda s: s.__setitem__("scope", "wrong"))
    assert reward.details["correctness"] == 1.0
    assert reward.details["scope_accuracy"] == 0.0
    assert reward.reward == 0.0


def test_false_nilpotent_order_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"]["nilpotent"].__setitem__("power2", [0, 0, 0, 0, 0]),
    )
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_false_verified_claim_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(tmp_path, lambda s: s.__setitem__("claimed_assurance", "VERIFIED"))
    assert reward.details["false_certification"] is True
    assert reward.reward == 0.0


def test_equivalent_evidence_wording_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "The two affine schemes each have three rational points, and the "
        "induced map gives a bijection between them. The algebra A contains "
        "a nonzero nilpotent element of order three. Since B is reduced, "
        "the two schemes cannot be isomorphic.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward.details["evidence_validity"] == 1.0
    assert reward.reward == 1.0


def test_unrelated_evidence_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text("The quick brown fox jumps over the lazy dog.\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward.details["evidence_validity"] == 0.0
    assert reward.reward == 0.0


def test_contradictory_keyword_evidence_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        "The induced map is a bijection on the same three rational points. "
        "A has a nonzero order-three nilpotent. B is not reduced and the "
        "schemes are isomorphic.\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward.details["evidence_validity"] == 0.0
    assert reward.reward == 0.0


def test_large_evidence_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    explanation = (
        "Both affine schemes are nonempty and have the same three rational "
        "points under the induced map. A has a nonzero order-three nilpotent "
        "while B is reduced, so they are not isomorphic.\n"
    )
    # Put the required facts beyond both the old cap and a fixed prefix scan.
    evidence.write_text("derivation filler\n" * 70_000 + explanation)
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward.details["evidence_validity"] == 1.0
    assert reward.reward == 1.0


def test_malformed_point_entries_do_not_crash(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"].__setitem__("a_points", [{}, {}, {}]),
    )
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_wrong_length_nilpotent_vector_does_not_crash(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"]["nilpotent"].__setitem__("vector", [0, 0, 0, 0, 0, 0]),
    )
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_nonnumeric_nilpotent_vector_does_not_crash(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"]["nilpotent"].__setitem__("vector", [0, 0, 0, 0, "x"]),
    )
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0
