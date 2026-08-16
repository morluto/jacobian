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
root = (
    Path(sys.argv[2])
    if len(sys.argv) == 3 and sys.argv[1] == "--root"
    else Path("/app")
)
submission = {
    "result": result,
}
(root / "submission.json").write_text(
    json.dumps(submission, sort_keys=True, separators=(",", ":"))
)
