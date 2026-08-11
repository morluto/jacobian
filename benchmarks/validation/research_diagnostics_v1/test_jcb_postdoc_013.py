"""Public-contract regressions for the answer-visible Problem 397 task."""

from __future__ import annotations

import copy
import json

import pytest
from benchmarks.validation.research_diagnostics_v1 import support
from jsonschema import Draft202012Validator, ValidationError

TASK = support.DATASET / "jcb-postdoc-013"


def test_required_key_facts_are_fully_agent_visible() -> None:
    schema = json.loads(
        (TASK / "environment" / "submission_schema.json").read_text(encoding="utf-8")
    )
    facts = schema["properties"]["result"]["properties"]["key_facts"]

    assert facts == {
        "type": "object",
        "additionalProperties": False,
        "required": ["identity_holds", "index_order"],
        "properties": {
            "identity_holds": {"const": "true"},
            "index_order": {"const": "a<a+1<2a<2a+2<c<c+1"},
        },
    }


def test_public_schema_rejects_renamed_hidden_lexical_keys() -> None:
    schema = json.loads(
        (TASK / "environment" / "submission_schema.json").read_text(encoding="utf-8")
    )
    submission = json.loads(
        (TASK / "solution" / "submission.json").read_text(encoding="utf-8")
    )
    renamed = copy.deepcopy(submission)
    renamed["result"]["key_facts"] = {
        "exact_identity": "true",
        "strict_index_order": "a<a+1<2a<2a+2<c<c+1",
    }

    Draft202012Validator(schema).validate(submission)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(renamed)
