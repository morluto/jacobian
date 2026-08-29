"""Regression coverage for the stateless MCP visibility evaluator."""

from __future__ import annotations

from pathlib import Path

from benchmarks.tooling.codex_visibility import (
    _DEFAULT_CASES,
    AdoptionExpectation,
    CueLevel,
    VisibilityCase,
    VisibilityCompositionRequirement,
    classify_visibility,
    load_suite,
    surface_snapshot_digest,
)

from jacobian.catalog.builtins import BUILTIN_TOOLS


def test_default_visibility_suite_is_repository_relative_and_loadable() -> None:
    assert _DEFAULT_CASES.is_file()
    assert load_suite(_DEFAULT_CASES).cases


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


def test_exact_producer_consumer_handoff_satisfies_composition() -> None:
    canonical_cnf = {"variables": ["a", "b"], "clauses": [[-1, 2], [1]]}
    case = VisibilityCase(
        case_id="cnf-composition",
        cue_level=CueLevel.AFFORDANCE,
        prompt="Canonicalize then check.",
        expected_operation_ids=("sat.cnf.canonicalize", "sat.assignment.check"),
        required_compositions=(
            VisibilityCompositionRequirement(
                producer_operation_id="sat.cnf.canonicalize",
                producer_output_field="cnf",
                consumer_operation_id="sat.assignment.check",
                consumer_input_field="cnf",
            ),
        ),
    )
    telemetry = {
        "operation_attempt_ids": [
            "sat.cnf.canonicalize",
            "sat.assignment.check",
        ],
        "operation_ids": ["sat.cnf.canonicalize", "sat.assignment.check"],
        "operation_invocations": [
            {
                "operation_id": "sat.cnf.canonicalize",
                "input": {"clauses": [[1, -2], [2]]},
                "output": {"cnf": canonical_cnf},
            },
            {
                "operation_id": "sat.assignment.check",
                "input": {"cnf": canonical_cnf, "assignment": [True, True]},
                "output": {"satisfies": True},
            },
        ],
        "mcp_calls": ["sat.cnf.canonicalize", "sat.assignment.check"],
    }

    classification = classify_visibility(case, telemetry)

    assert classification["contract_satisfied"] is True
    assert classification["compositions"]["satisfied"] is True


def test_reconstructed_consumer_value_fails_exact_composition() -> None:
    case = VisibilityCase(
        case_id="cnf-composition-mismatch",
        cue_level=CueLevel.AFFORDANCE,
        prompt="Canonicalize then check.",
        expected_operation_ids=("sat.cnf.canonicalize", "sat.assignment.check"),
        required_compositions=(
            VisibilityCompositionRequirement(
                producer_operation_id="sat.cnf.canonicalize",
                producer_output_field="cnf",
                consumer_operation_id="sat.assignment.check",
                consumer_input_field="cnf",
            ),
        ),
    )
    telemetry = {
        "operation_attempt_ids": [
            "sat.cnf.canonicalize",
            "sat.assignment.check",
        ],
        "operation_ids": ["sat.cnf.canonicalize", "sat.assignment.check"],
        "operation_invocations": [
            {
                "operation_id": "sat.cnf.canonicalize",
                "output": {"cnf": {"variables": ["a"], "clauses": [[1]]}},
            },
            {
                "operation_id": "sat.assignment.check",
                "input": {"cnf": {"variables": ["a"], "clauses": []}},
                "output": {"satisfies": True},
            },
        ],
        "mcp_calls": ["sat.cnf.canonicalize", "sat.assignment.check"],
    }

    classification = classify_visibility(case, telemetry)

    assert classification["contract_satisfied"] is False
    assert classification["compositions"]["satisfied"] is False


def test_checked_in_visibility_suites_reference_public_operations() -> None:
    public_ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    config_dir = Path(__file__).parents[1] / "config"

    for name in (
        "codex-visibility-v2.json",
        "direct-mcp-adoption-v1.json",
        "schema-bound-selection-v1.json",
        "typed-checker-handoff-v1.json",
    ):
        suite = load_suite(config_dir / name)
        expected_ids = {
            operation_id
            for case in suite.cases
            for operation_id in case.expected_operation_ids
        }
        assert expected_ids <= public_ids, name
