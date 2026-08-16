import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "unit-fraction-classification-repair"


def rewrite(app: Path, s: dict):
    _fixtures._write_json(app / "submission.json", {"result": s["result"]})


def test_accepts_alternative_valid_witnesses(tmp_path: Path):
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
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
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_rejects_published_count(tmp_path: Path):
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    s = json.loads((app / "submission.json").read_text())
    s["result"]["corrected_count"] = 2022
    rewrite(app, s)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_rejects_single_bit_coverage_tamper(tmp_path: Path):
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    s = json.loads((app / "submission.json").read_text())
    bitmap = bytearray.fromhex(s["result"]["membership_bitmap_hex"])
    bitmap[0] ^= 1 << 5
    s["result"]["membership_bitmap_hex"] = bitmap.hex()
    rewrite(app, s)
    assert _verifier._run_verifier(task, app, logs).details["correctness"] == 0.0
