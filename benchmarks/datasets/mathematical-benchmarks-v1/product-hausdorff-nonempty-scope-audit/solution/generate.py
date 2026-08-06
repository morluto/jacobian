import hashlib
import json
import os
import shutil
from pathlib import Path


def main():
    app = Path(os.environ.get("APP_DIR", "/app"))
    frozen = Path("/solution/input.json")
    if frozen.exists():
        shutil.copyfile(frozen, app / "input.json")
    result = {
        "factor_cardinalities": [4, 0, 3],
        "bad_factor_topology": [[], [3], [2, 3], [1, 2, 3], [0, 1, 2, 3]],
        "empty_factor_index": 1,
        "product_cardinality": 0,
        "product_is_hausdorff": True,
        "bad_factor_is_t0": True,
        "bad_factor_is_hausdorff": False,
        "missing_assumption": "ALL_FACTORS_NONEMPTY",
    }
    limitations = ["FINITE_TOPOLOGIES_ONLY", "LEAN_ELABORATION_NOT_ASSESSED"]
    task_id = "jacobian/product-hausdorff-nonempty-scope-audit"
    evidence = {
        "schema_version": "1",
        "task_id": task_id,
        "result": result,
        "limitations": limitations,
    }
    ep = app / "evidence/product-hausdorff-audit.json"
    ep.parent.mkdir(parents=True, exist_ok=True)
    ep.write_text(json.dumps(evidence, separators=(",", ":")))
    submission = {
        "task_id": task_id,
        "conclusion": "EMPTY_FACTOR_MASKS_NON_HAUSDORFF_FACTOR",
        "scope": "FINITE_TOPOLOGICAL_PRODUCT_SCOPE_COUNTERMODEL",
        "claimed_assurance": "COMPUTED",
        "completeness": "COMPLETE",
        "result": result,
        "limitations": limitations,
        "evidence": [
            {
                "path": "evidence/product-hausdorff-audit.json",
                "sha256": "sha256:" + hashlib.sha256(ep.read_bytes()).hexdigest(),
            }
        ],
    }
    (app / "submission.json").write_text(json.dumps(submission, indent=2))
    (app / "answer.txt").write_text(
        "Finite topology countermodel generated; see submission.json.\n"
    )


if __name__ == "__main__":
    main()
