import hashlib
import json
import sys
from pathlib import Path


def derive_case(n):
    masks = [mask for mask in range(1 << n) if not (mask & (mask << 1))]
    compatible = sum(not (left & right) for left in masks for right in masks)
    counts = dict.fromkeys(masks, 1)
    layers = [sum(counts.values())]
    for _ in range(1, n):
        counts = {
            mask: sum(value for prior, value in counts.items() if not (mask & prior))
            for mask in masks
        }
        layers.append(sum(counts.values()))
    return {
        "n": n,
        "valid_row_masks": masks,
        "compatible_pair_count": compatible,
        "layer_totals": layers,
        "independent_set_count": layers[-1],
    }


cases = [derive_case(n) for n in range(2, 6)]
result = {"cases": cases, "total": sum(case["independent_set_count"] for case in cases)}
limitations = ["FINITE_GRIDS_N_2_THROUGH_5", "NO_GENERAL_GRID_FORMULA"]
evidence = {
    "schema_version": "1",
    "task_id": "jacobian/grid-independent-set-transfer",
    "result": result,
    "limitations": limitations,
}
root = (
    Path(sys.argv[2])
    if len(sys.argv) == 3 and sys.argv[1] == "--root"
    else Path("/app")
)
(root / "evidence").mkdir(parents=True, exist_ok=True)
evidence_path = (
    root / "answer.txt" if root != Path("/app") else root / "evidence/answer.txt"
)
evidence_path.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
digest = "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
submission = {
    "task_id": "jacobian/grid-independent-set-transfer",
    "conclusion": "TRANSFER_COUNTS_REPLAYED",
    "result": result,
    "claimed_assurance": "COMPUTED",
    "scope": "ALL_ROW_MASK_STATES_FOR_SQUARE_GRIDS_2_THROUGH_5",
    "completeness": "COMPLETE",
    "evidence": [{"path": "evidence/answer.txt", "sha256": digest}],
    "limitations": limitations,
}
(root / "submission.json").write_text(
    json.dumps(submission, sort_keys=True, separators=(",", ":"))
)
