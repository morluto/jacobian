from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def test_process_runs_matrix_determinant_with_no_state_dir(tmp_path: Path) -> None:
    payload = {
        "matrix": {
            "domain": "QQ",
            "entries": [
                [
                    {"num": "1", "den": "1"},
                    {"num": "2", "den": "1"},
                ],
                [
                    {"num": "3", "den": "1"},
                    {"num": "4", "den": "1"},
                ],
            ],
        }
    }
    missing = tmp_path / "no-state"
    jacobian = shutil.which("jacobian")
    assert jacobian is not None
    completed = subprocess.run(
        [
            jacobian,
            "run",
            "matrix.determinant.compute",
            "--json",
            json.dumps(payload),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout)
    assert response["runtime_ms"] >= 0
    assert response["output"]["determinant"] == {"num": "-2", "den": "1"}
    assert not missing.exists()
