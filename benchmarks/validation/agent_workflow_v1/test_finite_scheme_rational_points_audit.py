from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

TASK = "finite-scheme-rational-points-audit"


def _prepare(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _mutate(tmp_path: Path, mutation):
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    mutation(submission)
    support._write_json(app / "submission.json", submission)
    return support._run_verifier(task, app, logs)


def test_oracle_countermodel_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["reward"] == 1.0


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
    assert reward["evidence_validity"] == 1.0
    assert reward["reward"] == 1.0


def test_visible_input_tamper_preserves_math_diagnostic(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    (app / "input.json").write_text("{}")
    reward = support._run_verifier(task, app, logs)
    assert reward["input_binding"] == 0.0
    assert reward["correctness"] == 1.0
    assert reward["reward"] == 0.0


def test_corrupted_multiplication_table_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path, lambda s: s["result"]["a_multiplication"][3][3].__setitem__(4, 0)
    )
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_incomplete_rational_point_enumeration_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(tmp_path, lambda s: s["result"]["a_points"].pop())
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_nonmultiplicative_morphism_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path, lambda s: s["result"]["morphism_columns"][0].__setitem__(3, 1)
    )
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


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
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_bad_scope_preserves_math_diagnostic(tmp_path: Path) -> None:
    reward = _mutate(tmp_path, lambda s: s.__setitem__("scope", "wrong"))
    assert reward["correctness"] == 1.0
    assert reward["scope_accuracy"] == 0.0
    assert reward["reward"] == 0.0


def test_false_nilpotent_order_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"]["nilpotent"].__setitem__("power2", [0, 0, 0, 0, 0]),
    )
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_false_verified_claim_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(tmp_path, lambda s: s.__setitem__("claimed_assurance", "VERIFIED"))
    assert reward["false_certification"] is True
    assert reward["reward"] == 0.0
