from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    app = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("/app")

    numbers = [2**a * 3**b for a in range(10) for b in range(10)]
    pairs = []
    for i, left in enumerate(numbers):
        for j in range(i + 1, len(numbers)):
            low, high = sorted((left, numbers[j]))
            if high in (2 * low, 3 * low):
                pairs.append([i, j])
    factors = [
        {"index": 10 * a + b, "core": 1, "two_exponent": a, "three_exponent": b}
        for a in range(10)
        for b in range(10)
    ]
    result = {
        "numbers": numbers,
        "good_pairs": pairs,
        "factorizations": factors,
        "projection_summary": {
            "component_count": 1,
            "nonempty_rows": 10,
            "nonempty_columns": 10,
            "witness_projection_cost": 20,
            "universal_projection_cost": 20,
            "universal_edge_bound": 180,
        },
        "claimed_maximum": 180,
        "conclusion": "EXACT_MAXIMUM_CERTIFIED",
    }
    submission = {"result": result}
    (app / "submission.json").write_text(
        json.dumps(submission, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
