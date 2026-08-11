import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1.support import _run_verifier

TASK = "disjoint-closed-distance-scope-audit"


def oracle():
    return json.loads(
        (
            Path("benchmarks/datasets/mathematical-benchmarks-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )


def verify(tmp_path, submission):
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir()
    shutil.copy2(task / "environment/input.json", app / "input.json")
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    p = app / "evidence/disjoint-closed-distance-audit.json"
    p.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    return _run_verifier(task, app, logs)


def test_oracle_and_alternative_family_pass(tmp_path):
    assert verify(tmp_path / "oracle", oracle()).reward == 1.0
    alt = oracle()
    alt["result"].update({"horizontal_step": 5, "vertical_scale": 3, "offset": 7})
    s, c = 3, 7
    alt["result"]["distance_squared"] = [
        {"numerator": s * s, "denominator": (n + c) ** 2}
        for n in alt["result"]["sample_indices"]
    ]
    alt["result"]["separation_certificate"]["same_family_lower_bound_squared"] = {
        "numerator": 25,
        "denominator": 1,
    }
    for k, item in enumerate(alt["result"]["epsilon_witnesses"], 2):
        n = k * s + c
        item.update(
            {
                "index": n,
                "distance_squared": {"numerator": s * s, "denominator": (n + c) ** 2},
            }
        )
    assert verify(tmp_path / "alt", alt).reward == 1.0


def test_distance_closedness_and_assurance_attacks_fail(tmp_path):
    bad = oracle()
    bad["result"]["distance_squared"][0]["numerator"] += 1
    assert verify(tmp_path / "distance", bad).reward == 0
    bad = oracle()
    bad["result"]["separation_certificate"]["same_family_lower_bound_squared"][
        "numerator"
    ] = 1
    assert verify(tmp_path / "closed", bad).reward == 0
    bad = oracle()
    bad["claimed_assurance"] = "VERIFIED"
    assert verify(tmp_path / "assurance", bad).reward == 0


def test_tiny_boolean_and_input_tamper_fail(tmp_path):
    bad = oracle()
    bad["result"]["horizontal_step"] = True
    assert verify(tmp_path / "boolean", bad).reward == 0
    bad = oracle()
    bad["result"]["sample_indices"] = [*range(1, 10), 1]
    assert verify(tmp_path / "duplicate", bad).reward == 0
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "tamper/app", tmp_path / "tamper/logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    (app / "input.json").write_text("{}")
    (app / "submission.json").write_text(json.dumps(oracle()))
    assert _run_verifier(task, app, logs).reward == 0
