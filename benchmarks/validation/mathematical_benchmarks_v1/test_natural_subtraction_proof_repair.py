from __future__ import annotations

import json

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support
from jsonschema import Draft202012Validator, ValidationError

TASK = "natural-subtraction-proof-repair"


def test_schema_requires_both_basis_entries() -> None:
    task = support._task(TASK)
    schema = json.loads((task / "environment" / "submission_schema.json").read_text())
    submission = json.loads((task / "solution" / "submission.json").read_text())
    submission["result"]["basis_order"] = []
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(submission)

    submission = json.loads((task / "solution" / "submission.json").read_text())
    submission["result"]["multipliers"] = ["1"]
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(submission)
