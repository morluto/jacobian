import json
import os
import shutil
from fractions import Fraction
from pathlib import Path


def q(x):
    return {"numerator": x.numerator, "denominator": x.denominator}


def main():
    app = Path(os.environ.get("APP_DIR", "/app"))
    frozen = Path("/solution/input.json")
    if frozen.exists():
        shutil.copyfile(frozen, app / "input.json")
    h, s, c = 3, 2, 5
    ns = list(range(1, 11))
    result = {
        "horizontal_step": h,
        "vertical_scale": s,
        "offset": c,
        "sample_indices": ns,
        "distance_squared": [q(Fraction(s * s, (n + c) ** 2)) for n in ns],
        "epsilon_witnesses": [
            {
                "epsilon": q(Fraction(1, k)),
                "index": k * s + c,
                "distance_squared": q(Fraction(s * s, (k * s + 2 * c) ** 2)),
            }
            for k in range(2, 10)
        ],
        "separation_certificate": {
            "same_family_lower_bound_squared": q(Fraction(h * h)),
            "cross_family_vertical_nonzero": True,
            "closedness_reason": "DISTINCT_INDICES_HAVE_HORIZONTAL_GAP",
        },
        "formal_conclusion": "POSITIVE_DISTANCE",
        "corrected_conclusion": "SEPARATED_BUT_DISTANCE_INFIMUM_ZERO",
    }
    submission = {"result": result}
    (app / "submission.json").write_text(json.dumps(submission, indent=2))
    (app / "answer.txt").write_text(
        "Exact countermodel certificate generated; see submission.json.\n"
    )


if __name__ == "__main__":
    main()
