"""Shared public-contract regressions for the legacy diagnostic result shape."""

from __future__ import annotations

import json

import pytest
from benchmarks.validation.research_diagnostics_v1 import support
from jsonschema import Draft202012Validator, ValidationError

TASKS = (
    "jcb-postdoc-001",
    "jcb-postdoc-003",
    "jcb-postdoc-005",
    "jcb-postdoc-006",
    "jcb-postdoc-007",
    "jcb-postdoc-008",
    "jcb-postdoc-009",
    "jcb-postdoc-010",
    "jcb-postdoc-011",
    "jcb-postdoc-012",
    "jcb-postdoc-017",
)


@pytest.mark.parametrize("task_name", TASKS)
def test_legacy_key_facts_are_nonempty_string_maps(task_name: str) -> None:
    schema = json.loads(
        (
            support.DATASET / task_name / "environment" / "submission_schema.json"
        ).read_text(encoding="utf-8")
    )
    facts_schema = schema["properties"]["result"]["properties"]["key_facts"]
    validator = Draft202012Validator(facts_schema)

    validator.validate({"public_fact": "value"})
    for invalid in ({}, {"public_fact": 1}, {"public_fact": True}, {"Bad": "value"}):
        with pytest.raises(ValidationError):
            validator.validate(invalid)
