from __future__ import annotations

import json

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "lean-transitive-axiom-audit"


def test_fixture_requires_genuine_transitive_closure() -> None:
    task = support._task(TASK)
    source = json.loads((task / "environment" / "input.json").read_text())
    case = next(
        item for item in source["cases"] if item["case_id"] == "axiom-type-closure"
    )
    assert "A0" not in case["dependencies"]["A2"]
    assert case["dependencies"]["A1"] == ["A0"]
