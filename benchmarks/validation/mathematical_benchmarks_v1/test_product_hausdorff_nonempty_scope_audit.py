import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1.support import _run_verifier

TASK = "product-hausdorff-nonempty-scope-audit"


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
    p = app / "evidence/product-hausdorff-audit.json"
    p.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    return _run_verifier(task, app, logs)


def test_oracle_and_relabelled_chain_topology_pass(tmp_path):
    assert verify(tmp_path / "oracle", oracle())["reward"] == 1.0
    alt = oracle()
    alt["result"]["bad_factor_topology"] = [[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
    assert verify(tmp_path / "alt", alt)["reward"] == 1.0


def test_topology_t0_product_and_assurance_attacks_fail(tmp_path):
    bad = oracle()
    bad["result"]["bad_factor_topology"].pop(2)
    assert verify(tmp_path / "topology", bad)["reward"] == 0
    bad = oracle()
    bad["result"]["bad_factor_topology"] = [[], [0, 1, 2, 3], [0], [1], [0, 1]]
    assert verify(tmp_path / "not-topology", bad)["reward"] == 0
    bad = oracle()
    bad["result"]["factor_cardinalities"][1] = 1
    assert verify(tmp_path / "product", bad)["reward"] == 0
    bad = oracle()
    bad["claimed_assurance"] = "VERIFIED"
    assert verify(tmp_path / "assurance", bad)["reward"] == 0


def test_boolean_and_input_tamper_fail_closed(tmp_path):
    bad = oracle()
    bad["result"]["factor_cardinalities"][0] = True
    assert verify(tmp_path / "boolean", bad)["reward"] == 0
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "tamper/app", tmp_path / "tamper/logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    (app / "input.json").write_text("{}")
    (app / "submission.json").write_text(json.dumps(oracle()))
    assert _run_verifier(task, app, logs)["reward"] == 0
