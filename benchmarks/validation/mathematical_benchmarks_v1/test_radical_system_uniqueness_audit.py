from __future__ import annotations

import json
from pathlib import Path

TASK = Path(
    "benchmarks/datasets/mathematical-benchmarks-v1/radical-system-uniqueness-audit"
)


def test_public_schema_exposes_coefficient_order_and_scope() -> None:
    schema = json.loads((TASK / "environment/submission_schema.json").read_text())
    properties = schema["properties"]

    assert properties["scope"] == {
        "const": "Complete real solution set under principal-root semantics.",
        "type": "string",
    }
    coefficient_schema = properties["result"]["properties"]["elimination_coefficients"]
    assert "ascending power order" in coefficient_schema["description"]
    assert "[u^0, u^1, u^2, u^3, u^4]" in coefficient_schema["description"]
