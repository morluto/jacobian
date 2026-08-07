from __future__ import annotations

import hashlib
import json
from pathlib import Path

APP = Path("/app")
REPORT = APP / "evidence" / "provider-report.json"
TASK_ID = "jacobian/gudhi"
PROVIDER = "gudhi"
CONTRACT = "jacobian.gudhi-persistence-spike/v1"
PIN = "sha256:329e57f50ad88ddbf2f6a3fe7cbbb0b5121be7c92a01d3a0b57db9241b1e6813"

report = json.loads(REPORT.read_text())
status = report.get("status", "ERROR")
complete = status == "COMPLETED"
(APP / "submission.json").write_text(
    json.dumps(
        {
            "task_id": TASK_ID,
            "conclusion": "FEASIBLE" if complete else "NO_CONCLUSION",
            "result": {
                "provider": PROVIDER,
                "contract": CONTRACT,
                "status": status,
                "pin_sha256": PIN,
            },
            "claimed_assurance": "COMPUTED" if complete else "UNVERIFIED",
            "scope": "pinned bounded provider reproduction",
            "completeness": "COMPLETE" if complete else "UNKNOWN",
            "evidence": [
                {
                    "path": "evidence/provider-report.json",
                    "sha256": "sha256:"
                    + hashlib.sha256(REPORT.read_bytes()).hexdigest(),
                }
            ],
            "limitations": [
                "provider output is not an operator-authorized independent verification",
                "task failure is a non-conclusion for Jacobian core availability",
            ],
        },
        sort_keys=True,
    )
    + "\n"
)
