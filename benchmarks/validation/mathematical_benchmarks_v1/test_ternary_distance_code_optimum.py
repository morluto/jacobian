from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK_NAME = "ternary-distance-code-optimum"
TASK = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "datasets"
    / "mathematical-benchmarks-v1"
    / TASK_NAME
)


def _inject_marker(app: Path, submission: dict) -> None:
    evidence = app / "evidence" / "answer.txt"
    text = evidence.read_text()
    lines = [line for line in text.splitlines() if not line.startswith("RESULT_JSON:")]
    marker = "RESULT_JSON: " + json.dumps(
        submission["result"], sort_keys=True, separators=(",", ":")
    )
    lines.append(marker)
    evidence.write_text("\n".join(lines) + "\n")
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence)


def _case(
    tmp_path: Path, submission: dict, *, label: str = "case", tamper_input: bool = False
):
    root = tmp_path / label
    app = root / "app"
    logs = root / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    if tamper_input:
        source = json.loads((app / "input.json").read_text())
        source["claimed_optimum"] = 17
        _fixtures._write_json(app / "input.json", source)
    evidence = (TASK / "solution" / "answer.txt").read_bytes()
    (app / "evidence" / "answer.txt").write_bytes(evidence)
    submission = deepcopy(submission)
    _inject_marker(app, submission)
    _fixtures._write_json(app / "submission.json", submission)
    return TASK, app, logs


def _submission() -> dict:
    return json.loads((TASK / "solution" / "submission.json").read_text())


def test_reference_certificate_passes(tmp_path: Path) -> None:
    result = _verifier._run_verifier(*_case(tmp_path, _submission(), label="reference"))
    assert result.reward == pytest.approx(1.0)


def test_alphabet_permutation_is_accepted(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["codewords"] = [
        "".join(str((int(symbol) + 1) % 3) for symbol in word)
        for word in reversed(submission["result"]["codewords"])
    ]
    result = _verifier._run_verifier(*_case(tmp_path, submission, label="alternative"))
    assert result.reward == pytest.approx(1.0)


def test_pair_distance_corruption_is_rejected(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["codewords"][1] = "000001"
    result = _verifier._run_verifier(*_case(tmp_path, submission, label="distance"))
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_wrong_dual_multiplier_is_rejected(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["upper_bound_certificate"]["dual_multipliers"]["order_1"] = (
        "1/2"
    )
    result = _verifier._run_verifier(*_case(tmp_path, submission, label="dual"))
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_noncanonical_rational_is_accepted(tmp_path: Path) -> None:
    """Equivalent unreduced rationals such as 2/12 for 1/6 are accepted as
    the same nonnegative rational value.
    """
    submission = _submission()
    submission["result"]["upper_bound_certificate"]["dual_multipliers"]["order_2"] = (
        "2/12"
    )
    result = _verifier._run_verifier(*_case(tmp_path, submission, label="fraction"))
    assert result.details["correctness"] == 1.0
    assert result.reward == pytest.approx(1.0)


def test_input_tampering_is_rejected(tmp_path: Path) -> None:
    result = _verifier._run_verifier(
        *_case(tmp_path, _submission(), label="input", tamper_input=True)
    )
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_unhashable_codewords_rejected_without_crash(tmp_path: Path) -> None:
    """A malformed submission with array codewords is rejected cleanly instead
    of crashing with a TypeError before reward.json is written.
    """
    submission = _submission()
    submission["result"]["codewords"] = [
        ["0", "0", "0", "0", "0", "0"],
        *submission["result"]["codewords"][1:],
    ]
    result = _verifier._run_verifier(*_case(tmp_path, submission, label="unhashable"))
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_out_of_alphabet_codewords_rejected(tmp_path: Path) -> None:
    """Codewords containing symbols outside the ternary alphabet (e.g. 'x'
    replacing '0') are rejected even when distances are preserved.
    """
    submission = _submission()
    submission["result"]["codewords"] = [
        word.replace("0", "x") for word in submission["result"]["codewords"]
    ]
    result = _verifier._run_verifier(*_case(tmp_path, submission, label="alphabet"))
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_witness_without_result_marker_is_rejected(tmp_path: Path) -> None:
    """Evidence prose without a RESULT_JSON marker is rejected."""
    submission = _submission()
    root = tmp_path / "no-marker"
    app = root / "app"
    logs = root / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "An 18-word ternary code exists with pairwise Hamming distance four or six.\n",
        encoding="utf-8",
    )
    submission = deepcopy(submission)
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence_path)
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(TASK, app, logs)
    assert result.reward == 0.0
    assert result.reward == 0.0


def test_witness_result_mismatch_is_rejected(tmp_path: Path) -> None:
    """Evidence with a RESULT_JSON marker that doesn't match the result is rejected."""
    submission = _submission()
    root = tmp_path / "mismatch"
    app = root / "app"
    logs = root / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("RESULT_JSON: {}\n", encoding="utf-8")
    submission = deepcopy(submission)
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence_path)
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(TASK, app, logs)
    assert result.reward == 0.0
    assert result.reward == 0.0
