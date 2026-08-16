from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "bounded-variation-uniform-limit"


def _rational(numerator: int, denominator: int = 1) -> dict[str, int]:
    return {"numerator": numerator, "denominator": denominator}


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_alternative_scale_and_indices_pass(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["scale_q"] = 5
    submission["result"]["uniform_certificate"]["sup_norm_denominator_coefficient"] = 5
    submission["result"]["checkpoints"] = []
    for n in (3, 7, 13, 21):
        k = 5 * n
        submission["result"]["checkpoints"].append(
            {
                "n": n,
                "frequency": k,
                "amplitude": _rational(1, k),
                "interior_segments": 2 * k - 1,
                "endpoint_contribution": _rational(2, k),
                "interior_contribution": _rational(4 * k - 2, k),
                "total_variation": _rational(4),
            }
        )
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def _mutate(tmp_path: Path, mutation) -> float:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    mutation(submission)
    _fixtures._write_json(path, submission)
    return _verifier._run_verifier(task, app, logs).reward


def test_wrong_segment_count_is_rejected(tmp_path: Path) -> None:
    assert (
        _mutate(
            tmp_path,
            lambda s: s["result"]["checkpoints"][0].__setitem__("interior_segments", 6),
        )
        == 0.0
    )


def test_wrong_variation_is_rejected(tmp_path: Path) -> None:
    assert (
        _mutate(
            tmp_path,
            lambda s: s["result"]["variation_formula"].__setitem__(
                "total_variation", _rational(0)
            ),
        )
        == 0.0
    )


def test_duplicate_indices_are_rejected(tmp_path: Path) -> None:
    def duplicate(submission):
        submission["result"]["checkpoints"][1] = dict(
            submission["result"]["checkpoints"][0]
        )

    assert _mutate(tmp_path, duplicate) == 0.0


def test_visible_input_tampering_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    data = json.loads((app / "input.json").read_text())
    data["source"]["row"] = 601
    _fixtures._write_json(app / "input.json", data)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_unstructured_argument_claim_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["argument"] = {"implication": "does not force"}
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_uses_result_only_protocol(tmp_path: Path) -> None:
    _fixtures.assert_result_witness_protocol(tmp_path, TASK)


def test_bool_in_integer_certificate_is_rejected(tmp_path: Path) -> None:
    """``True == 1`` in Python must not let a boolean pass as an integer."""

    def mutate(submission):
        submission["result"]["uniform_certificate"]["sup_norm_numerator"] = True

    assert _mutate(tmp_path, mutate) == 0.0


def test_bool_in_endpoint_jump_multiplier_is_rejected(tmp_path: Path) -> None:
    def mutate(submission):
        submission["result"]["variation_formula"]["endpoint_jump_multiplier"] = True

    assert _mutate(tmp_path, mutate) == 0.0


def test_legacy_sequence_string_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["sequence"] = "sin(n*q*x)/(n*q)"
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_equivalent_unreduced_rational_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    value = submission["result"]["checkpoints"][0]["amplitude"]
    value["numerator"] *= 2
    value["denominator"] *= 2
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0
