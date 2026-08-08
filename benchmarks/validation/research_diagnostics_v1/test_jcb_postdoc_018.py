"""Public-protocol integrity checks for the WOWII Conjecture 18 diagnostic."""

from __future__ import annotations

import json

from benchmarks.validation.research_diagnostics_v1 import support

TASK_NAME = "jcb-postdoc-018"


def test_visible_assurance_terms_match_schema_and_verifier_policy() -> None:
    task = support.DATASET / TASK_NAME
    schema = json.loads(
        (task / "environment" / "submission_schema.json").read_text(encoding="utf-8")
    )
    instruction = (task / "instruction.md").read_text(encoding="utf-8")

    assert schema["properties"]["claimed_assurance"]["enum"] == [
        "UNVERIFIED",
        "COMPUTED",
    ]
    assert "You may claim `UNVERIFIED` or `COMPUTED`" in instruction
    assert "only `COMPUTED` receives\nfull aggregate credit" in instruction
