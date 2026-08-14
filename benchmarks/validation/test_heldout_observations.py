from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.tooling.heldout_observations import (
    _heldout_plan_failures,
    _mark_invoked_if_operation_used,
)


@pytest.mark.parametrize("runs", (["bad-run"], None))
def test_heldout_evidence_rejects_malformed_run_entries(runs: object) -> None:
    selected, failures = _heldout_plan_failures(
        {
            "runs": runs,
            "pair_count": 1,
            "manifest_digest": "sha256:" + "a" * 64,
            "plan_digest": "sha256:" + "b" * 64,
        },
        {
            "plan_digest": "sha256:" + "b" * 64,
            "manifest_digest": "sha256:" + "a" * 64,
            "status": "COMPLETE",
        },
        "C1",
    )

    assert selected == []
    assert any("held-out plan runs" in failure for failure in failures)


def _c2_routing_contract(routing_status: str = "AVAILABLE_UNUSED") -> dict:
    return {
        "schema_version": "2",
        "manifest_digest": "sha256:" + "a" * 64,
        "condition_id": "C2",
        "infrastructure_status": "READY",
        "routing_status": routing_status,
        "treatment": {
            "image": "registry.invalid/jacobian@sha256:" + "1" * 64,
            "server_version": "1.0.0",
            "catalog_digest": "sha256:" + "2" * 64,
        },
        "routing": {"compose_file": "c2.compose.json", "mcp_url": "http://x/mcp"},
        "probe": {
            "reachable": True,
            "server_version_observed": "1.0.0",
            "catalog_digest_observed": "sha256:" + "2" * 64,
            "tool_names": ["math.find", "math.run"],
            "discovery_matches": ["cap-1"],
            "probe_digest": "sha256:" + "0" * 64,
            "diagnostic": None,
        },
        "checks": {
            "image_digest_pinned": True,
            "catalog_digest_bound": True,
            "server_version_bound": True,
            "server_version_match": True,
            "catalog_digest_match": True,
            "required_tools_present": True,
            "describe_responded": True,
        },
        "failures": [],
    }


def test_mark_invoked_transitions_on_successful_operation_invoke(
    tmp_path: Path,
) -> None:
    ledger = {"routing_status": {"C2": _c2_routing_contract()}}
    trials = [
        {
            "status": "COMPLETED",
            "tool_calls": {"math.run": 1},
            "tool_errors": 0,
        }
    ]
    _mark_invoked_if_operation_used(ledger, trials, contract_dir=tmp_path)

    assert ledger["routing_status"]["C2"]["routing_status"] == "AVAILABLE_INVOKED"
    assert (tmp_path / "routing-status-c2.json").is_file()


def test_mark_invoked_fail_closed_on_errored_invocation(tmp_path: Path) -> None:
    """A failed/errored math.run must not transition to AVAILABLE_INVOKED."""

    ledger = {"routing_status": {"C2": _c2_routing_contract()}}
    trials = [
        {
            "status": "COMPLETED",
            "tool_calls": {"math.run": 1},
            "tool_errors": 2,
        }
    ]
    _mark_invoked_if_operation_used(ledger, trials, contract_dir=tmp_path)

    assert ledger["routing_status"]["C2"]["routing_status"] == "AVAILABLE_UNUSED"
    assert not (tmp_path / "routing-status-c2.json").exists()


def test_mark_invoked_fail_closed_on_non_completed_trial(tmp_path: Path) -> None:
    """A non-COMPLETED trial with math.run must not transition."""

    ledger = {"routing_status": {"C2": _c2_routing_contract()}}
    trials = [
        {
            "status": "ERROR",
            "tool_calls": {"math.run": 1},
            "tool_errors": 0,
        }
    ]
    _mark_invoked_if_operation_used(ledger, trials, contract_dir=tmp_path)

    assert ledger["routing_status"]["C2"]["routing_status"] == "AVAILABLE_UNUSED"


def test_mark_invoked_fail_closed_on_timeout_trial(tmp_path: Path) -> None:
    """A timed-out trial with math.run must not transition."""

    ledger = {"routing_status": {"C2": _c2_routing_contract()}}
    trials = [
        {
            "status": "TIMEOUT",
            "tool_calls": {"math.run": 3},
            "tool_errors": 0,
        }
    ]
    _mark_invoked_if_operation_used(ledger, trials, contract_dir=tmp_path)

    assert ledger["routing_status"]["C2"]["routing_status"] == "AVAILABLE_UNUSED"


def test_mark_invoked_no_transition_when_already_invoked(tmp_path: Path) -> None:
    """If routing_status is already AVAILABLE_INVOKED, do not re-write."""

    ledger = {"routing_status": {"C2": _c2_routing_contract("AVAILABLE_INVOKED")}}
    trials = [
        {
            "status": "COMPLETED",
            "tool_calls": {"math.run": 1},
            "tool_errors": 0,
        }
    ]
    _mark_invoked_if_operation_used(ledger, trials, contract_dir=tmp_path)

    assert ledger["routing_status"]["C2"]["routing_status"] == "AVAILABLE_INVOKED"
    assert not (tmp_path / "routing-status-c2.json").exists()


def test_mark_invoked_mixed_trials_one_success_transitions(tmp_path: Path) -> None:
    """One successful invocation among errored trials is enough to transition."""

    ledger = {"routing_status": {"C2": _c2_routing_contract()}}
    trials = [
        {
            "status": "COMPLETED",
            "tool_calls": {"math.run": 1},
            "tool_errors": 2,
        },
        {
            "status": "COMPLETED",
            "tool_calls": {"math.run": 1},
            "tool_errors": 0,
        },
    ]
    _mark_invoked_if_operation_used(ledger, trials, contract_dir=tmp_path)

    assert ledger["routing_status"]["C2"]["routing_status"] == "AVAILABLE_INVOKED"
