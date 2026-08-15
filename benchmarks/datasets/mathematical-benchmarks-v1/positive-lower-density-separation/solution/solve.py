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
Path("/app/submission.json").write_text(json.dumps({"result": result}, indent=2) + "\n")
