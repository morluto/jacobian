from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures

TASK = Path(
    "benchmarks/datasets/mathematical-benchmarks-v1/radical-system-uniqueness-audit"
)


def test_uses_result_only_protocol(tmp_path: Path) -> None:
    _fixtures.assert_result_witness_protocol(
        tmp_path, "radical-system-uniqueness-audit"
    )


def test_public_schema_exposes_coefficient_order() -> None:
    schema = json.loads((TASK / "environment/submission_schema.json").read_text())
    properties = schema["properties"]

    assert "scope" not in properties
    coefficient_schema = properties["result"]["properties"]["elimination_coefficients"]
    assert "ascending power order" in coefficient_schema["description"]
    assert "[u^0, u^1, u^2, u^3, u^4]" in coefficient_schema["description"]
