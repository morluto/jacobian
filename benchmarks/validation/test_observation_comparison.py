"""Tests for observation comparison behavior (preserved v2 public contract)."""

from __future__ import annotations

from copy import deepcopy

from benchmarks.tooling.observation_comparison import compare_evidence, render_markdown
from benchmarks.tooling.observation_results import _comparison_job
from benchmarks.validation.observation_results_support import _evidence


def test_paired_report_keeps_public_claim_boundary() -> None:
    report = compare_evidence(
        _evidence("control", [0.0, 1.0]), _evidence("treatment", [1.0, 1.0])
    )

    assert report["status"] == "VALID"
    assert report["causal_claim_authorized"] is False
    assert report["metrics"]["correctness"]["paired_delta"] == 0.5
    assert (
        report["metrics"]["correctness"]["interpretation"] == "descriptive-small-sample"
    )
    assert "does not itself authorize a causal" in render_markdown(report)


def test_comparison_rejects_invariant_drift() -> None:
    control = _evidence("control", [1.0])
    treatment = deepcopy(_evidence("treatment", [1.0]))
    treatment["fixed_invariants"]["model"] = "different"

    report = compare_evidence(control, treatment)

    assert report["status"] == "INVALID"
    assert "fixed invariants differ" in report["validation_failures"]


def test_comparison_rejects_unpaired_repetitions() -> None:
    report = compare_evidence(
        _evidence("control", [1.0, 1.0]), _evidence("treatment", [1.0])
    )

    assert report["status"] == "INVALID"
    assert (
        "control/treatment trials do not pair exactly" in report["validation_failures"]
    )


def test_comparison_rejects_duplicate_pair_keys() -> None:
    control = _evidence("control", [1.0])
    control["trials"].append(deepcopy(control["trials"][0]))

    report = compare_evidence(control, _evidence("treatment", [1.0]))

    assert report["status"] == "INVALID"
    assert "duplicate" in " ".join(report["validation_failures"])


def test_comparison_derives_heldout_class_from_both_inputs() -> None:
    control = _evidence("C1", [1.0])
    treatment = _evidence("C2", [1.0])
    control["evidence_class"] = "held-out-comparative-evaluation"
    treatment["evidence_class"] = "held-out-comparative-evaluation"

    report = compare_evidence(control, treatment)

    assert report["evidence_class"] == "held-out-comparison"
    assert report["status"] == "VALID"


def test_comparison_rejects_same_condition_inputs() -> None:
    report = compare_evidence(_evidence("control", [1.0]), _evidence("control", [1.0]))

    assert report["status"] == "INVALID"
    assert (
        "conditions must be a distinct control/treatment or C1/C2 pair"
        in report["validation_failures"]
    )


def test_comparison_normalization_allows_only_frozen_jacobian_differences() -> None:
    control = {
        "environment": {
            "extra_docker_compose": ["benchmarks/config/agent-eval-proxy.compose.yaml"]
        },
        "agents": [{"name": "codex"}],
    }
    treatment = {
        "environment": {
            "extra_docker_compose": [
                "benchmarks/config/agent-eval-proxy.compose.yaml",
                "/tmp/rendered/c2.compose.json",
            ]
        },
        "agents": [
            {
                "name": "codex",
                "mcp_servers": [
                    {
                        "name": "jacobian",
                        "transport": "streamable-http",
                        "url": "http://jacobian:8000/mcp",
                    }
                ],
            }
        ],
    }

    assert _comparison_job(control) == _comparison_job(treatment)

    heldout_treatment = {
        "environment": {
            "extra_docker_compose": [
                "benchmarks/config/agent-eval-proxy.compose.yaml",
                "/tmp/rendered/c2.compose.json",
            ]
        },
        "agents": [
            {
                "name": "codex",
                "mcp_servers": [
                    {
                        "name": "jacobian",
                        "transport": "streamable-http",
                        "url": "http://jacobian:8000/mcp",
                    }
                ],
            }
        ],
    }
    assert _comparison_job(control) == _comparison_job(heldout_treatment)

    treatment["environment"]["extra_docker_compose"].append("unexpected.yaml")
    assert _comparison_job(control) != _comparison_job(treatment)

    treatment["environment"]["extra_docker_compose"].pop()
    treatment["agents"][0]["mcp_servers"].append(
        {"name": "unexpected", "transport": "stdio", "url": "http://other"}
    )
    assert _comparison_job(control) != _comparison_job(treatment)
