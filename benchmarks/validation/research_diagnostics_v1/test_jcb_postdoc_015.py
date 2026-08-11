"""Agent-visible evidence-contract regressions for Problem 707."""

from __future__ import annotations

from benchmarks.validation.research_diagnostics_v1 import support

TASK = support.DATASET / "jcb-postdoc-015"


def test_evidence_schema_is_named_and_copied_into_the_agent_environment() -> None:
    instruction = (TASK / "instruction.md").read_text(encoding="utf-8")
    dockerfile = (TASK / "environment" / "Dockerfile").read_text(encoding="utf-8")

    assert "/app/evidence_schema.json" in instruction
    assert "COPY input.json submission_schema.json evidence_schema.json /app/" in (
        dockerfile
    )
