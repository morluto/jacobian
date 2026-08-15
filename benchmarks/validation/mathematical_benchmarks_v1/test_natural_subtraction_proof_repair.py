from __future__ import annotations

import json

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
)
from jsonschema import Draft202012Validator, ValidationError

TASK = "natural-subtraction-proof-repair"


def test_schema_requires_both_basis_entries() -> None:
    task = _fixtures._task(TASK)
    schema = json.loads((task / "environment" / "submission_schema.json").read_text())
    submission = json.loads((task / "solution" / "submission.json").read_text())
    submission["result"]["basis_order"] = []
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(submission)

    submission = json.loads((task / "solution" / "submission.json").read_text())
    submission["result"]["multipliers"] = [{"numerator": 1, "denominator": 1}]
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(submission)


def test_schema_rejects_string_coerced_rationals() -> None:
    task = _fixtures._task(TASK)
    schema = json.loads((task / "environment" / "submission_schema.json").read_text())
    submission = json.loads((task / "solution" / "submission.json").read_text())
    submission["result"]["derived_coefficients"][0] = "0"
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(submission)
