import copy
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier

TASK = "closed-set-distance-strengthening-audit"


def _oracle() -> dict[str, object]:
    return json.loads(
        (
            Path("benchmarks/datasets/public-reproductions-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )


def _prepare(
    tmp_path: Path,
    submission: dict[str, object],
    *,
    evidence_payload: dict[str, object] | None = None,
):
    task = Path("benchmarks/datasets/public-reproductions-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    if evidence_payload is None:
        evidence_payload = {
            "schema_version": "1",
            "result": submission["result"],
        }
    evidence_path = app / "evidence/distance-audit.json"
    evidence_path.write_text(json.dumps(evidence_payload, separators=(",", ":")))
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    return task, app, logs


def _verify(tmp_path: Path, submission: dict[str, object]):
    return _run_verifier(*_prepare(tmp_path, submission))


def _pairs(start: int) -> list[dict[str, object]]:
    return [
        {"index": n, "a": [str(n), "0"], "b": [str(n), f"1/{n}"], "distance": f"1/{n}"}
        for n in range(start, start + 8)
    ]


def test_oracle_and_alternative_family_are_accepted(tmp_path: Path) -> None:
    assert _verify(tmp_path / "oracle", _oracle()).reward == 1.0
    alternative = copy.deepcopy(_oracle())
    alternative["result"]["start_index"] = 7
    alternative["result"]["point_pairs"] = _pairs(7)
    alternative["result"]["epsilon_witnesses"] = [
        {"epsilon": "1/4", "index": 7, "distance": "1/7"},
        {"epsilon": "1/8", "index": 9, "distance": "1/9"},
        {"epsilon": "1/16", "index": 17, "distance": "1/17"},
        {"epsilon": "1/32", "index": 33, "distance": "1/33"},
    ]
    assert _verify(tmp_path / "alternative", alternative).reward == 1.0


def test_rejects_corrupt_geometry_and_nonvanishing_gap(tmp_path: Path) -> None:
    for name, mutation in [
        (
            "coordinate",
            lambda result: result["point_pairs"][3]["b"].__setitem__(1, "1/8"),
        ),
        (
            "distance",
            lambda result: result["epsilon_witnesses"][2].update(distance="1/10"),
        ),
        (
            "ordering",
            lambda result: result["epsilon_witnesses"][2].update(epsilon="1/4"),
        ),
    ]:
        submission = copy.deepcopy(_oracle())
        mutation(submission["result"])
        assert _verify(tmp_path / name, submission).reward == 0.0


# ---------------------------------------------------------------------------
# Adversarial regression tests for PR #493 review threads.
# -----------------------------------------------------------------------


def test_accepts_epsilon_witnesses_above_one(tmp_path: Path) -> None:
    """T2: the public contract allows any positive epsilon; no hidden < 1 bound."""
    submission = copy.deepcopy(_oracle())
    submission["result"]["epsilon_witnesses"] = [
        {"epsilon": "2", "index": 4, "distance": "1/4"},
        {"epsilon": "3/2", "index": 6, "distance": "1/6"},
        {"epsilon": "1", "index": 11, "distance": "1/11"},
        {"epsilon": "1/2", "index": 21, "distance": "1/21"},
    ]
    result = _verify(tmp_path / "epsilon-above-one", submission)
    assert result.reward == 1.0
    assert result.details["correctness"] == 1.0


def test_rejects_float_point_pair_indices(tmp_path: Path) -> None:
    """T4: float indices like 4.0 must not bypass integer validation."""
    submission = copy.deepcopy(_oracle())
    for _i, row in enumerate(submission["result"]["point_pairs"]):
        row["index"] = float(row["index"])
    result = _verify(tmp_path / "float-indices", submission)
    assert result.reward == 0.0
    assert result.details["correctness"] == 0.0


def test_rejects_evidence_without_schema_version(tmp_path: Path) -> None:
    """T5: evidence missing the published schema_version field is rejected."""
    submission = copy.deepcopy(_oracle())
    payload = {
        "result": submission["result"],
    }
    result = _run_verifier(
        *_prepare(tmp_path / "no-schema-version", submission, evidence_payload=payload)
    )
    assert result.details["witness_validity"] == 0.0
    assert result.reward == 0.0
    assert result.details["correctness"] == 1.0


def test_tampered_input_preserves_correctness_and_gates_reward(
    tmp_path: Path,
) -> None:
    """T6: input binding is a separate diagnostic; math stays correct."""
    submission = copy.deepcopy(_oracle())
    task, app, logs = _prepare(tmp_path / "tampered-input", submission)
    input_path = app / "input.json"
    input_data = json.loads(input_path.read_text())
    input_data["task_id"] = "tampered"
    input_path.write_text(json.dumps(input_data))
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.details["input_binding"] == 0.0
    assert result.reward == 0.0
