"""Regression coverage for the stateless MCP visibility evaluator."""

from __future__ import annotations

from benchmarks.tooling.codex_visibility import (
    AdoptionExpectation,
    CueLevel,
    VisibilityCase,
    classify_visibility,
    surface_snapshot_digest,
)


def test_completed_math_run_satisfies_visibility_without_a_verification_record() -> (
    None
):
    case = VisibilityCase(
        case_id="determinant",
        cue_level=CueLevel.EXPLICIT,
        prompt="Compute a determinant.",
        expectation=AdoptionExpectation.USE,
        expected_operation_ids=("matrix.determinant.compute",),
    )

    classification = classify_visibility(
        case,
        {
            "operation_attempt_ids": ["matrix.determinant.compute"],
            "operation_ids": ["matrix.determinant.compute"],
            "operation_invocations": [
                {
                    "operation_id": "matrix.determinant.compute",
                    "output": {"determinant": "-2"},
                }
            ],
            "mcp_calls": ["math.run"],
        },
    )

    assert classification["contract_satisfied"] is True
    assert "verified" not in classification["observed"]


def test_surface_snapshot_digest_accepts_the_current_catalog_shape() -> None:
    surface = {
        "server": {"name": "jacobian", "version": "1.0.0"},
        "instructions": "Use math.find and math.run.",
        "tools": [{"name": "math.find"}, {"name": "math.run"}],
        "catalog": {
            "catalog_version": "1",
            "catalog_digest": "sha256:" + "0" * 64,
            "operation_count": 2,
            "content_sha256": "sha256:" + "1" * 64,
        },
    }

    assert surface_snapshot_digest(surface).startswith("sha256:")
