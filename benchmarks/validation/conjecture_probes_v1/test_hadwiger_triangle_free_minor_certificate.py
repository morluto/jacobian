from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = (
    ROOT
    / "benchmarks/datasets/conjecture-probes-v1/hadwiger-triangle-free-minor-certificate"
)


def case(tmp_path):
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    subprocess.run(
        [sys.executable, str(TASK / "solution/solve.py"), "--root", str(app)],
        check=True,
    )
    return app, logs, json.loads((app / "submission.json").read_text())


def write(app, s):
    payload = {
        "schema_version": "1",
        "task_id": "jacobian/hadwiger-triangle-free-minor-certificate",
        "result": s["result"],
    }
    e = app / "evidence/answer.txt"
    e.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    s["witness"][0]["sha256"] = "sha256:" + hashlib.sha256(e.read_bytes()).hexdigest()
    (app / "submission.json").write_text(json.dumps(s) + "\n")


def run(app, logs):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def _write_evidence(app, s, result):
    payload = {
        "schema_version": "1",
        "task_id": "jacobian/hadwiger-triangle-free-minor-certificate",
        "result": result,
    }
    e = app / "evidence/answer.txt"
    e.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    s["witness"][0]["sha256"] = "sha256:" + hashlib.sha256(e.read_bytes()).hexdigest()
    (app / "submission.json").write_text(json.dumps(s) + "\n")


def test_evidence_result_requires_exact_json_types(tmp_path):
    app, logs, s = case(tmp_path)
    # JSON booleans in place of integers must not compare equal to 0/1.
    coloring = list(s["result"]["four_coloring"])
    coloring[0] = True
    _write_evidence(app, s, {**s["result"], "four_coloring": coloring})
    r = run(app, logs)
    assert r.details["mathematics"] == 1.0 and r.details["witness_validity"] == 0.0
    assert r.details["aggregate_reward"] == 0.0
    # An integral float must not compare equal to the integer invariant.
    app, logs, s = case(tmp_path / "float")
    _write_evidence(app, s, {**s["result"], "chromatic_number": 4.0})
    r = run(app, logs)
    assert r.details["mathematics"] == 1.0 and r.details["witness_validity"] == 0.0
    assert r.details["aggregate_reward"] == 0.0


def test_oracle_and_relabeling_pass(tmp_path):
    app, logs, _ = case(tmp_path)
    assert run(app, logs).details["aggregate_reward"] == 1.0
    app, logs, s = case(tmp_path / "perm")
    perm = list(reversed(range(11)))
    s["result"]["edges"] = sorted(
        [sorted([perm[a], perm[b]]) for a, b in s["result"]["edges"]]
    )
    colors = [0] * 11
    for old, c in enumerate(s["result"]["four_coloring"]):
        colors[perm[old]] = c
    s["result"]["four_coloring"] = colors
    s["result"]["branch_sets"] = [
        [perm[v] for v in branch] for branch in s["result"]["branch_sets"]
    ]
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 1.0


def test_bad_coloring_and_disconnected_branch_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["four_coloring"][1] = s["result"]["four_coloring"][0]
    write(app, s)
    assert run(app, logs).details["mathematics"] == 0.0
    app, logs, s = case(tmp_path / "branch")
    s["result"]["branch_sets"][2] = [2, 5]
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 0.0


def test_triangle_and_duplicate_edge_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["edges"][0] = [0, 2]
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 0.0
    app, logs, s = case(tmp_path / "dup")
    s["result"]["edges"][-1] = s["result"]["edges"][0]
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 0.0


def test_nonfinite_json_rejected_in_raw_parse(tmp_path):
    app, logs, s = case(tmp_path)
    # NaN in an extra envelope field is invalid JSON; the raw parser must
    # reject it deterministically instead of crediting raw-derived metrics.
    (app / "submission.json").write_text(
        json.dumps({**s, "extra": float("nan")}) + "\n"
    )
    r = run(app, logs)
    assert (
        r.details["mathematics"] == 0.0
        and r.details["aggregate_reward"] == 0.0
        and "error" not in r.details
    )
