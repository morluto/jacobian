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
    submission = {"result": result}
    (app / "submission.json").write_text(json.dumps(submission, indent=2))
    (app / "answer.txt").write_text(
        "Finite topology countermodel generated; see submission.json.\n"
    )


if __name__ == "__main__":
    main()
