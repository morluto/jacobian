from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmarks.validation._solution import run_solution
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
    run_solution(TASK, app)
    return app, logs, json.loads((app / "submission.json").read_text())


def write(app, s):
    (app / "submission.json").write_text(json.dumps({"result": s["result"]}) + "\n")


def run(app, logs):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_boolean_coloring_is_rejected(tmp_path):
    app, logs, s = case(tmp_path)
    coloring = list(s["result"]["four_coloring"])
    coloring[0] = True
    s["result"]["four_coloring"] = coloring
    write(app, s)
    r = run(app, logs)
    assert r.details["mathematics"] == 0.0
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
    (app / "submission.json").write_text(
        json.dumps({**s, "extra": float("nan")}) + "\n"
    )
    r = run(app, logs)
    assert r.details["mathematics"] == 0.0
    assert r.details["aggregate_reward"] == 0.0
    assert "error" not in r.details
