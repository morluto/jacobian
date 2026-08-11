import hashlib
import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "unit-fraction-classification-repair"


def rewrite(app: Path, s: dict):
    e = {
        "schema_version": "1",
        "task_id": s["task_id"],
        "result": s["result"],
        "limitations": s["limitations"],
    }
    raw = json.dumps(e, separators=(",", ":")).encode()
    (app / "evidence/unit-fraction-repair.json").write_bytes(raw)
    s["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    support._write_json(app / "submission.json", s)


def test_accepts_alternative_valid_witnesses(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    s = json.loads((app / "submission.json").read_text())
    s["result"]["witnesses"] = [
        {"n": n, "d": d}
        for n, d in [
            (6, 4),
            (12, 8),
            (15, 9),
            (20, 16),
            (28, 16),
            (35, 25),
            (40, 25),
            (45, 27),
            (56, 32),
            (63, 49),
        ]
    ]
    rewrite(app, s)
    assert support._run_verifier(task, app, logs).reward == 1.0


def test_rejects_published_count(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    s = json.loads((app / "submission.json").read_text())
    s["result"]["corrected_count"] = 2022
    rewrite(app, s)
    assert support._run_verifier(task, app, logs).reward == 0.0


def test_rejects_single_bit_coverage_tamper(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    s = json.loads((app / "submission.json").read_text())
    bitmap = bytearray.fromhex(s["result"]["membership_bitmap_hex"])
    bitmap[0] ^= 1 << 5
    s["result"]["membership_bitmap_hex"] = bitmap.hex()
    rewrite(app, s)
    assert support._run_verifier(task, app, logs).details["correctness"] == 0.0


def test_decouples_evidence_and_scope_from_assurance(tmp_path: Path):
    """An otherwise exact submission claiming VERIFIED must retain
    evidence_validity and scope_accuracy while failing assurance and reward."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "invalid")
    s = json.loads((app / "submission.json").read_text())
    rewrite(app, s)
    result = support._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.details["evidence_validity"] == 0.0
    assert result.details["scope_accuracy"] == 0.0
    assert result.details["assurance_calibration"] == 0.0
    assert result.reward == 0.0
