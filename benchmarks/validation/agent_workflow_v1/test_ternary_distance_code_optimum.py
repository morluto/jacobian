from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK_NAME = "ternary-distance-code-optimum"
TASK = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "datasets"
    / "agent-workflow-v1"
    / TASK_NAME
)


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
        support._write_json(app / "input.json", source)
    evidence = (TASK / "solution" / "answer.txt").read_bytes()
    (app / "evidence" / "answer.txt").write_bytes(evidence)
    submission = deepcopy(submission)
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence).hexdigest()
    )
    support._write_json(app / "submission.json", submission)
    return TASK, app, logs


def _submission() -> dict:
    return json.loads((TASK / "solution" / "submission.json").read_text())


def test_reference_certificate_passes(tmp_path: Path) -> None:
    result = support._run_verifier(*_case(tmp_path, _submission(), label="reference"))
    assert result["reward"] == pytest.approx(1.0)
    assert result["false_certification"] is False


def test_alphabet_permutation_is_accepted(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["codewords"] = [
        "".join(str((int(symbol) + 1) % 3) for symbol in word)
        for word in reversed(submission["result"]["codewords"])
    ]
    result = support._run_verifier(*_case(tmp_path, submission, label="alternative"))
    assert result["reward"] == pytest.approx(1.0)


def test_pair_distance_corruption_is_rejected(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["codewords"][1] = "000001"
    result = support._run_verifier(*_case(tmp_path, submission, label="distance"))
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_wrong_dual_multiplier_is_rejected(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["upper_bound_certificate"]["dual_multipliers"]["order_1"] = (
        "1/2"
    )
    result = support._run_verifier(*_case(tmp_path, submission, label="dual"))
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_noncanonical_rational_is_accepted(tmp_path: Path) -> None:
    """Equivalent unreduced rationals such as 2/12 for 1/6 are accepted as
    the same nonnegative rational value.
    """
    submission = _submission()
    submission["result"]["upper_bound_certificate"]["dual_multipliers"]["order_2"] = (
        "2/12"
    )
    result = support._run_verifier(*_case(tmp_path, submission, label="fraction"))
    assert result["correctness"] == 1.0
    assert result["reward"] == pytest.approx(1.0)


def test_verified_claim_is_rejected(tmp_path: Path) -> None:
    submission = _submission()
    submission["claimed_assurance"] = "VERIFIED"
    result = support._run_verifier(*_case(tmp_path, submission, label="verified"))
    assert result["false_certification"] is True
    assert result["reward"] == 0.0


def test_input_tampering_is_rejected(tmp_path: Path) -> None:
    result = support._run_verifier(
        *_case(tmp_path, _submission(), label="input", tamper_input=True)
    )
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_evidence_contradictory_distribution_is_rejected(tmp_path: Path) -> None:
    """Evidence prose that claims distribution values contradicting the
    certificate is rejected, binding evidence to the submitted result.
    """
    submission = _submission()
    root = tmp_path / "contradictory"
    app = root / "app"
    logs = root / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        "An 18-word ternary code exists with pairwise Hamming distance four or "
        "six. Its distance distribution is A0=1, A4=0, A6=17. The nonnegative "
        "combination of Delsarte inequalities is a dual certificate proving the "
        "matching upper bound. This is an exact COMPUTED certificate.\n",
        encoding="utf-8",
    )
    submission = deepcopy(submission)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(TASK, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_unhashable_codewords_rejected_without_crash(tmp_path: Path) -> None:
    """A malformed submission with array codewords is rejected cleanly instead
    of crashing with a TypeError before reward.json is written.
    """
    submission = _submission()
    submission["result"]["codewords"] = [
        ["0", "0", "0", "0", "0", "0"],
        *submission["result"]["codewords"][1:],
    ]
    result = support._run_verifier(*_case(tmp_path, submission, label="unhashable"))
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_out_of_alphabet_codewords_rejected(tmp_path: Path) -> None:
    """Codewords containing symbols outside the ternary alphabet (e.g. 'x'
    replacing '0') are rejected even when distances are preserved.
    """
    submission = _submission()
    submission["result"]["codewords"] = [
        word.replace("0", "x") for word in submission["result"]["codewords"]
    ]
    result = support._run_verifier(*_case(tmp_path, submission, label="alphabet"))
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_affirmative_formal_proof_limitation_rejected(tmp_path: Path) -> None:
    """A limitation that affirmatively claims formal proof was checked is
    rejected, even though it contains the phrase 'formal proof'.
    """
    submission = _submission()
    submission["limitations"] = [
        "A formal proof was independently checked, so there are no limitations."
    ]
    result = support._run_verifier(*_case(tmp_path, submission, label="affirmative"))
    assert result["reward"] == 0.0


def test_semantically_correct_limitation_accepted(tmp_path: Path) -> None:
    """A limitation that disclaims formal verification with different wording
    (e.g. 'No independent checker was run') is accepted.
    """
    submission = _submission()
    submission["limitations"] = [
        "No independent checker or proof assistant was run on this certificate."
    ]
    result = support._run_verifier(*_case(tmp_path, submission, label="alt-wording"))
    assert result["reward"] == pytest.approx(1.0)
