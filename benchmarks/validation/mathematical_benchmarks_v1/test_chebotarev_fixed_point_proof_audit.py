import hashlib
import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "chebotarev-fixed-point-proof-audit"


def rewrite(app: Path, s: dict):
    e = {
        "schema_version": "1",
        "task_id": s["task_id"],
        "result": s["result"],
        "limitations": s["limitations"],
    }
    raw = json.dumps(e, separators=(",", ":")).encode()
    (app / "evidence/chebotarev-audit.json").write_bytes(raw)
    s["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    support._write_json(app / "submission.json", s)


def test_oracle_passes(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    rewrite(app, json.loads((app / "submission.json").read_text()))
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_published_double_transposition_error(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    s = json.loads((app / "submission.json").read_text())
    s["result"]["cycle_types"][2].update({"fixed_points": 4, "contributes": True})
    s["result"].update(
        {
            "fixed_point_total": 18,
            "density": {"numerator": 3, "denominator": 4},
            "encoded_answer": 304,
        }
    )
    rewrite(app, s)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_rejects_correct_density_with_wrong_discriminant(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    s = json.loads((app / "submission.json").read_text())
    s["result"]["actual_discriminant"] = 4352
    rewrite(app, s)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0
