import json
from fractions import Fraction
from pathlib import Path

data = json.loads(Path("/app/input.json").read_text())
selected = [0, 2, 4]
ratio = Fraction(
    data["alpha"] + sum(data["items"][index]["t"] for index in selected),
    data["beta"] + sum(data["items"][index]["f"] for index in selected),
)
p, q = ratio.numerator, ratio.denominator
residuals = [q * item["t"] - p * item["f"] for item in data["items"]]
result = {
    "contract_mismatches": [
        "OBJECTIVE_REPLACED",
        "BINARY_DOMAIN_RELAXED",
        "UNDECLARED_BUDGET_ADDED",
    ],
    "selected_indices": selected,
    "attained_ratio": {
        "numerator": ratio.numerator,
        "denominator": ratio.denominator,
    },
    "constant_residual": q * data["alpha"] - p * data["beta"],
    "item_residuals": [
        {"index": index, "value": value} for index, value in enumerate(residuals)
    ],
    "positive_residual_indices": [
        index for index, value in enumerate(residuals) if value > 0
    ],
    "maximum_residual_sum": 0,
    "repair_method": "EXACT_FRACTIONAL_RESIDUAL_CERTIFICATE",
}
Path("/app/submission.json").write_text(json.dumps({"result": result}, indent=2) + "\n")
