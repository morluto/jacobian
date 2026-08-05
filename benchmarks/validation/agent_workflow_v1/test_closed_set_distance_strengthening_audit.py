import copy
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.agent_workflow_v1.support import _run_verifier

TASK = "closed-set-distance-strengthening-audit"


def _oracle() -> dict[str, object]:
    return json.loads(
        (
            Path("benchmarks/datasets/agent-workflow-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )


def _prepare(tmp_path: Path, submission: dict[str, object]):
    task = Path("benchmarks/datasets/agent-workflow-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    evidence = {
        "schema_version": "1",
        "task_id": f"jacobian/{TASK}",
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence/distance-audit.json"
    evidence_path.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
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
    assert _verify(tmp_path / "oracle", _oracle())["reward"] == 1.0
    alternative = copy.deepcopy(_oracle())
    alternative["result"]["start_index"] = 7
    alternative["result"]["point_pairs"] = _pairs(7)
    alternative["result"]["epsilon_witnesses"] = [
        {"epsilon": "1/4", "index": 7, "distance": "1/7"},
        {"epsilon": "1/8", "index": 9, "distance": "1/9"},
        {"epsilon": "1/16", "index": 17, "distance": "1/17"},
        {"epsilon": "1/32", "index": 33, "distance": "1/33"},
    ]
    assert _verify(tmp_path / "alternative", alternative)["reward"] == 1.0


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
        assert _verify(tmp_path / name, submission)["reward"] == 0.0


def test_rejects_noncanonical_rational_and_false_certification(tmp_path: Path) -> None:
    noncanonical = copy.deepcopy(_oracle())
    noncanonical["result"]["point_pairs"][0]["distance"] = "2/8"
    assert _verify(tmp_path / "noncanonical", noncanonical)["reward"] == 0.0
    verified = copy.deepcopy(_oracle())
    verified["claimed_assurance"] = "VERIFIED"
    assert _verify(tmp_path / "verified", verified)["false_certification"] is True
