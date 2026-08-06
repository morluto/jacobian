import hashlib
import json
from fractions import Fraction
from pathlib import Path

b = 3
levels = []
for m in range(8):
    high, low = b ** (2 * m + 1), b ** (2 * m + 2)
    count = (low - 1) // (b + 1)
    levels.append(
        {
            "level": m,
            "included_endpoint": high,
            "excluded_endpoint": low,
            "cumulative_count": count,
            "included_density": str(Fraction(count, high)),
            "excluded_density": str(Fraction(count, low)),
        }
    )
result = {
    "base": b,
    "family": "ALTERNATING_GEOMETRIC_BLOCKS",
    "count_formula": "(b^(2m+2)-1)/(b+1)",
    "levels": levels,
    "lower_density": str(Fraction(1, b + 1)),
    "upper_density": str(Fraction(b, b + 1)),
    "lower_density_positive": True,
    "natural_density_exists": False,
    "semantic_relation": "FORMALIZED_PREDICATE_STRICTLY_STRONGER",
}
text = (
    "The lower density is positive, while the two endpoint subsequences have different limits, so the natural density does not exist. The finite levels replay instances of the general formula rather than proving every infinite case.\nRESULT_JSON:"
    + json.dumps(result, sort_keys=True, separators=(",", ":"))
    + "\n"
)
Path("/app/evidence").mkdir(parents=True, exist_ok=True)
Path("/app/evidence/answer.txt").write_text(text)
digest = hashlib.sha256(text.encode()).hexdigest()
submission = {
    "task_id": "jacobian/positive-lower-density-separation",
    "conclusion": "POSITIVE_LOWER_DENSITY_DOES_NOT_IMPLY_DENSITY_EXISTS",
    "result": result,
    "claimed_assurance": "COMPUTED",
    "scope": "parameterized-geometric-block-family-with-eight-replayed-levels",
    "completeness": "COMPLETE",
    "evidence": [{"path": "evidence/answer.txt", "sha256": f"sha256:{digest}"}],
    "limitations": [
        "Eight exact levels replay the general formula but do not machine-prove the infinite limit or the Erdős problem."
    ],
}
Path("/app/submission.json").write_text(json.dumps(submission, indent=2) + "\n")
