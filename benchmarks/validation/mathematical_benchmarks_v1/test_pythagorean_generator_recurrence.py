import importlib.util
import json
import sys
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT
    / "benchmarks/datasets/mathematical-benchmarks-v1/pythagorean-generator-recurrence"
)
TASK_NAME = "pythagorean-generator-recurrence"


def load_verifier():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "pythagorean_recurrence_verifier", TASK / "tests/verifier.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def candidate(seed=(2, 1)):
    verifier = load_verifier()
    m, n = seed
    stages = []
    for index in range(8):
        stages.append(verifier.expected_stage(index, m, n))
        m, n = 2 * m + n, m
    return {
        "transform_matrix": [[2, 1], [1, 0]],
        "transform_determinant": -1,
        "invariant_multiplier": -1,
        "stages": stages,
    }


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK_NAME, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(app / "submission.json", submission)


def test_accepts_two_distinct_valid_seeds():
    verifier = load_verifier()
    assert verifier.valid_result(candidate((2, 1)))
    assert verifier.valid_result(candidate((5, 2)))


def test_corrupt_stage_and_transform_are_rejected():
    verifier = load_verifier()
    bad = candidate()
    bad["stages"][4]["c"] += 1
    assert not verifier.valid_result(bad)
    bad = candidate()
    bad["transform_matrix"] = [[1, 2], [0, 1]]
    assert not verifier.valid_result(bad)


def test_nonprimitive_or_same_parity_seed_is_rejected():
    verifier = load_verifier()
    assert not verifier.valid_result(candidate((6, 3)))
    assert not verifier.valid_result(candidate((3, 1)))


def test_booleans_rejected_in_integer_fields():
    verifier = load_verifier()
    bad = candidate()
    bad["transform_determinant"] = True
    assert not verifier.valid_result(bad)
    bad = candidate()
    bad["invariant_multiplier"] = True
    assert not verifier.valid_result(bad)
    bad = candidate()
    bad["transform_matrix"] = [[2, True], [1, 0]]
    assert not verifier.valid_result(bad)
    bad = candidate()
    bad["stages"][0]["gcd"] = True
    assert not verifier.valid_result(bad)
    bad = candidate()
    bad["stages"][0]["q"] = True
    assert not verifier.valid_result(bad)
    bad = candidate()
    bad["stages"][0]["stage"] = False
    assert not verifier.valid_result(bad)


def test_integer_rejected_in_boolean_field():
    verifier = load_verifier()
    bad = candidate()
    bad["stages"][0]["parity_opposite"] = 1
    assert not verifier.valid_result(bad)


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 1.0


def test_large_evidence_without_result_binding_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    (app / "evidence" / "answer.txt").write_text(
        "recurrence coprime pythagorean " + "x" * 65536
    )
    submission["witness"][0]["sha256"] = _fixtures._digest(
        app / "evidence" / "answer.txt"
    )
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.reward == 0.0


def test_oversized_result_marker_is_rejected_without_buffering_it(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text("RESULT_JSON:" + "x" * 65_537, encoding="utf-8")
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence)
    _fixtures._write_json(app / "submission.json", submission)

    result = _verifier._run_verifier(task, app, logs)

    assert result.reward == 0.0
    assert result.reward == 0.0


def test_large_evidence_with_result_binding_is_accepted(tmp_path: Path) -> None:
    """Evidence above the former 64 KiB ceiling with a valid RESULT_JSON
    binding must be accepted now that the arbitrary cap is removed."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    marker = "RESULT_JSON:" + json.dumps(
        submission["result"], sort_keys=True, separators=(",", ":")
    )
    evidence.write_text(
        "recurrence coprime pythagorean explanation. "
        + "x" * 65536
        + "\n"
        + marker
        + "\n"
    )
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence)
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 1.0
    assert result.reward == 1.0


def test_keyword_only_evidence_without_result_binding_is_rejected(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text("recurrence coprime pythagorean\n")
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence)
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.reward == 0.0


def test_evidence_with_wrong_result_json_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    wrong = json.loads(json.dumps(submission["result"]))
    wrong["transform_determinant"] = 1
    evidence.write_text(
        "recurrence coprime pythagorean\n"
        "RESULT_JSON:" + json.dumps(wrong, sort_keys=True, separators=(",", ":")) + "\n"
    )
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence)
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.reward == 0.0


def test_boolean_in_result_json_evidence_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    coerced = json.loads(json.dumps(submission["result"]))
    coerced["stages"][0]["gcd"] = True
    assert coerced == submission["result"]  # Python coerces True == 1
    evidence.write_text(
        "recurrence coprime pythagorean\n"
        "RESULT_JSON:"
        + json.dumps(coerced, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence)
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 0.0


def test_symlinked_workspace_input_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    original = app / "input-original.json"
    (app / "input.json").rename(original)
    (app / "input.json").symlink_to(original)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_input_tamper_preserves_math_correctness(tmp_path: Path) -> None:
    """When the workspace input is altered, mathematical correctness is still
    reported (the recurrence is unchanged) while aggregate reward is gated to
    zero via input binding."""
    task, app, logs = _case(tmp_path)
    (app / "input.json").write_text("{}")
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.reward == 0.0


def test_multiple_result_json_markers_are_rejected(tmp_path: Path) -> None:
    """More than one RESULT_JSON marker must fail evidence binding rather than
    silently picking the first, exercising the streaming scanner's count."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    marker = "RESULT_JSON:" + json.dumps(
        submission["result"], sort_keys=True, separators=(",", ":")
    )
    evidence.write_text(f"recurrence coprime pythagorean\n{marker}\n{marker}\n")
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence)
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.reward == 0.0


def test_instruction_documents_evidence_binding():
    text = (TASK / "instruction.md").read_text().casefold()
    assert "result_json" in text
