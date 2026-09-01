from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.heldout_manifest import _digest
from benchmarks.tooling.heldout_routing import (
    control_routing_status,
    treatment_readiness_preflight,
)
from benchmarks.validation.heldout_fixtures import _manifest, _write


def _ready_probe(*, mcp_url, expected_version, timeout_seconds):
    return {
        "reachable": True,
        "report": {
            "server": {"name": "jacobian", "version": "1.2.3"},
            "tool_names": ["math.find", "math.run"],
            "catalog": {
                "catalog_version": "1",
                "operations": 1,
                "catalog_digest": "sha256:" + "5" * 64,
                "sha256": "abc",
            },
            "discovery": {"bytes": 100, "matches": ["cap-1"]},
        },
    }


def _unreachable_probe(*, mcp_url, expected_version, timeout_seconds):
    return {"reachable": False, "diagnostic": "connection refused"}


def test_treatment_readiness_preflight_ready_with_successful_probe(
    tmp_path: Path,
) -> None:
    value = _manifest()
    manifest_path = _write(tmp_path, value)
    contract = treatment_readiness_preflight(
        manifest_path,
        mcp_url="http://127.0.0.1:8000/mcp",
        probe_fn=_ready_probe,
    )

    assert contract["infrastructure_status"] == "READY"
    assert contract["routing_status"] == "AVAILABLE_UNUSED"
    assert contract["manifest_digest"] == _digest(manifest_path)
    assert contract["condition_id"] == "C2"
    assert contract["checks"]["image_digest_pinned"] is True
    assert contract["checks"]["catalog_digest_bound"] is True
    assert contract["checks"]["server_version_bound"] is True
    assert contract["checks"]["server_version_match"] is True
    assert contract["checks"]["catalog_digest_match"] is True
    assert contract["checks"]["required_tools_present"] is True
    assert contract["checks"]["describe_responded"] is True
    assert contract["failures"] == []
    assert contract["probe"]["reachable"] is True
    assert contract["probe"]["server_version_observed"] == "1.2.3"
    assert contract["probe"]["catalog_digest_observed"] == "sha256:" + "5" * 64


def test_treatment_readiness_preflight_fail_closed_without_probe_url(
    tmp_path: Path,
) -> None:
    value = _manifest()
    manifest_path = _write(tmp_path, value)
    contract = treatment_readiness_preflight(manifest_path)

    assert contract["infrastructure_status"] == "MISCONFIGURED"
    assert contract["routing_status"] == "CONFIGURED_UNCALLABLE"
    assert any("probe URL is not configured" in f for f in contract["failures"])
    assert contract["probe"]["reachable"] is False


def test_treatment_readiness_preflight_unavailable_when_probe_fails(
    tmp_path: Path,
) -> None:
    value = _manifest()
    manifest_path = _write(tmp_path, value)
    contract = treatment_readiness_preflight(
        manifest_path,
        mcp_url="http://127.0.0.1:8000/mcp",
        probe_fn=_unreachable_probe,
        readiness_retries=0,
    )

    assert contract["infrastructure_status"] == "UNAVAILABLE"
    assert contract["routing_status"] == "CONFIGURED_UNCALLABLE"
    assert any("not reachable" in f for f in contract["failures"])
    assert contract["probe"]["reachable"] is False
    assert contract["probe"]["diagnostic"] == "connection refused"


def test_treatment_readiness_preflight_retries_until_probe_succeeds(
    tmp_path: Path,
) -> None:
    value = _manifest()
    manifest_path = _write(tmp_path, value)
    call_count = 0

    def eventually_ready(*, mcp_url, expected_version, timeout_seconds):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return {"reachable": False, "diagnostic": "connection refused"}
        return _ready_probe(
            mcp_url=mcp_url,
            expected_version=expected_version,
            timeout_seconds=timeout_seconds,
        )

    contract = treatment_readiness_preflight(
        manifest_path,
        mcp_url="http://127.0.0.1:8000/mcp",
        probe_fn=eventually_ready,
        readiness_retries=5,
        readiness_retry_delay_seconds=0,
    )

    assert contract["infrastructure_status"] == "READY"
    assert contract["routing_status"] == "AVAILABLE_UNUSED"
    assert contract["probe"]["reachable"] is True
    assert call_count == 3


def test_treatment_readiness_preflight_exhausts_retries_and_fails_closed(
    tmp_path: Path,
) -> None:
    value = _manifest()
    manifest_path = _write(tmp_path, value)
    call_count = 0

    def always_unreachable(*, mcp_url, expected_version, timeout_seconds):
        nonlocal call_count
        call_count += 1
        return {"reachable": False, "diagnostic": "connection refused"}

    contract = treatment_readiness_preflight(
        manifest_path,
        mcp_url="http://127.0.0.1:8000/mcp",
        probe_fn=always_unreachable,
        readiness_retries=3,
        readiness_retry_delay_seconds=0,
    )

    assert contract["infrastructure_status"] == "UNAVAILABLE"
    assert contract["routing_status"] == "CONFIGURED_UNCALLABLE"
    assert contract["probe"]["reachable"] is False
    assert call_count == 4


def test_treatment_readiness_preflight_misconfigured_on_digest_mismatch(
    tmp_path: Path,
) -> None:
    value = _manifest()
    manifest_path = _write(tmp_path, value)

    def mismatched_probe(*, mcp_url, expected_version, timeout_seconds):
        return {
            "reachable": True,
            "report": {
                "server": {"name": "jacobian", "version": "1.2.3"},
                "tool_names": ["math.find", "math.run"],
                "catalog": {
                    "catalog_version": "1",
                    "operations": 1,
                    "catalog_digest": "sha256:" + "9" * 64,
                    "sha256": "abc",
                },
                "discovery": {"bytes": 100, "matches": ["cap-1"]},
            },
        }

    contract = treatment_readiness_preflight(
        manifest_path,
        mcp_url="http://127.0.0.1:8000/mcp",
        probe_fn=mismatched_probe,
    )

    assert contract["infrastructure_status"] == "MISCONFIGURED"
    assert contract["routing_status"] == "MISROUTED"
    assert contract["checks"]["catalog_digest_match"] is False
    assert any("catalog_digest" in f for f in contract["failures"])


def test_control_routing_status_is_not_configured(tmp_path: Path) -> None:
    value = _manifest()
    manifest_path = _write(tmp_path, value)
    contract = control_routing_status(manifest_path)

    assert contract["condition_id"] == "C1"
    assert contract["infrastructure_status"] == "NOT_CONFIGURED"
    assert contract["routing_status"] == "NOT_APPLICABLE"
    assert contract["treatment"] is None
    assert contract["routing"] is None
    assert contract["probe"] is None
    assert contract["failures"] == []


def test_treatment_readiness_preflight_fails_for_unpinned_image(tmp_path: Path) -> None:
    value = _manifest()
    value["conditions"][1]["image"] = "registry.invalid/jacobian:latest"
    manifest_path = _write(tmp_path, value)

    with pytest.raises(HarborSuiteError, match="held-out manifest is invalid"):
        treatment_readiness_preflight(manifest_path)
