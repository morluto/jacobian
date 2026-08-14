"""Bounded live routing checks for held-out Harbor treatment jobs."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from benchmarks.tooling.command_runner import operator_environment, run_operator_command
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.harbor_suite import BENCHMARKS, ROOT
from benchmarks.tooling.heldout_bundle import (
    _DIGEST_RE,
    _digest,
    _json_digest,
    _read_json,
    validate_manifest,
)

_REQUIRED_MCP_TOOLS = {"math.find", "math.run"}
_PROBE_SCRIPT = ROOT / "deploy" / "smoke_remote.py"
_ZERO_DIGEST = "sha256:" + "0" * 64


def _probe_digest(report: dict[str, Any]) -> str:
    return _json_digest(
        {
            "server_version": report.get("server", {}).get("version"),
            "catalog_digest": report.get("catalog", {}).get("catalog_digest"),
            "tool_names": report.get("tool_names"),
            "discovery_matches": report.get("discovery", {}).get("matches"),
        }
    )


def _run_mcp_probe(
    *,
    mcp_url: str,
    expected_version: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Bounded MCP initialize + catalog + describe probe via command_runner.

    Runs ``deploy/smoke_remote.py`` under the locked environment.  Returns a
    dict with ``reachable`` and either the probe report or a safe diagnostic.
    """

    args = [
        "run",
        "--locked",
        "python",
        str(_PROBE_SCRIPT),
        mcp_url,
        "--expect-version",
        expected_version,
        "--timeout-seconds",
        str(int(timeout_seconds)),
    ]
    env = operator_environment(include=["PATH", "HOME"])
    result = run_operator_command(
        "uv",
        args,
        cwd=ROOT,
        timeout_seconds=timeout_seconds + 60.0,
        stdout_limit_bytes=2 * 1024 * 1024,
        stderr_limit_bytes=512 * 1024,
        environment=env,
    )
    if result.exit_code is None or result.exit_code != 0:
        diagnostic = (
            result.diagnostic or result.stderr.decode("utf-8", "replace").strip()[:512]
        )
        return {"reachable": False, "diagnostic": diagnostic or "probe failed"}
    try:
        report = json.loads(result.stdout.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {"reachable": False, "diagnostic": f"probe output unparseable: {exc}"}
    return {"reachable": True, "report": report}


def _empty_checks() -> dict[str, Any]:
    return {
        "image_digest_pinned": False,
        "catalog_digest_bound": False,
        "server_version_bound": False,
        "server_version_match": None,
        "catalog_digest_match": None,
        "required_tools_present": None,
        "describe_responded": None,
    }


def _validate_routing_contract(contract: dict[str, Any]) -> None:
    schema = _read_json(BENCHMARKS / "schemas" / "held-out-routing-status.schema.json")
    errors = list(Draft202012Validator(schema).iter_errors(contract))
    if errors:
        raise HarborSuiteError(
            f"routing status contract is invalid: {errors[0].message}"
        )


def control_routing_status(
    manifest_path: Path,
    *,
    compose_filename: str = "",
    mcp_url: str = "",
) -> dict[str, Any]:
    """Emit a NOT_CONFIGURED routing status contract for the control condition.

    Control performs no probe and records NOT_CONFIGURED / NOT_APPLICABLE.
    """

    validate_manifest(manifest_path)
    manifest_digest = _digest(manifest_path)
    contract = {
        "schema_version": "2",
        "manifest_digest": manifest_digest,
        "condition_id": "C1",
        "infrastructure_status": "NOT_CONFIGURED",
        "routing_status": "NOT_APPLICABLE",
        "treatment": None,
        "routing": None,
        "probe": None,
        "checks": _empty_checks(),
        "failures": [],
    }
    _validate_routing_contract(contract)
    return contract


def _static_binding_checks(
    treatment: dict[str, Any],
) -> tuple[dict[str, bool], list[str]]:
    """Check that manifest-level image, version, and catalog fields are bound."""
    image = str(treatment.get("image", ""))
    expected_server_version = str(treatment.get("server_version", ""))
    expected_catalog_digest = str(treatment.get("catalog_digest", ""))
    checks = _empty_checks()
    failures: list[str] = []
    if "@" in image and _DIGEST_RE.match(image.rsplit("@", 1)[1]):
        checks["image_digest_pinned"] = True
    else:
        failures.append("treatment image is not digest-pinned")
    if _DIGEST_RE.match(expected_catalog_digest):
        checks["catalog_digest_bound"] = True
    else:
        failures.append("treatment catalog_digest is not bound")
    if expected_server_version:
        checks["server_version_bound"] = True
    else:
        failures.append("treatment server_version is not bound")
    return checks, failures


def _probe_until_ready(
    *,
    mcp_url: str,
    expected_version: str,
    timeout_seconds: float,
    probe_fn: Any,
    retries: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    probe_result: dict[str, Any] = {"reachable": False}
    for attempt in range(retries + 1):
        probe_result = probe_fn(
            mcp_url=mcp_url,
            expected_version=expected_version,
            timeout_seconds=timeout_seconds,
        )
        if probe_result.get("reachable") or attempt == retries:
            return probe_result
        time.sleep(retry_delay_seconds)
    return probe_result


def _validate_readiness_retry_policy(retries: int, delay_seconds: float) -> None:
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise HarborSuiteError("readiness retries must be a non-negative integer")
    if (
        isinstance(delay_seconds, bool)
        or not isinstance(delay_seconds, (int, float))
        or not math.isfinite(delay_seconds)
        or delay_seconds < 0
    ):
        raise HarborSuiteError("readiness retry delay must be finite and non-negative")


def _probe_match_checks(
    probe: dict[str, Any],
    expected_server_version: str,
    expected_catalog_digest: str,
) -> tuple[dict[str, bool], list[str]]:
    """Compare probe observations against manifest expectations."""
    checks: dict[str, bool] = {}
    failures: list[str] = []
    checks["server_version_match"] = (
        probe["server_version_observed"] == expected_server_version
    )
    checks["catalog_digest_match"] = (
        probe["catalog_digest_observed"] == expected_catalog_digest
    )
    observed_tools = set(probe.get("tool_names") or [])
    checks["required_tools_present"] = _REQUIRED_MCP_TOOLS.issubset(observed_tools)
    checks["describe_responded"] = bool(probe.get("discovery_matches") is not None)
    if not checks["server_version_match"]:
        failures.append("probe server_version does not match manifest")
    if not checks["catalog_digest_match"]:
        failures.append("probe catalog_digest does not match manifest")
    if not checks["required_tools_present"]:
        failures.append("probe is missing required MCP tools")
    if not checks["describe_responded"]:
        failures.append("probe math.find did not respond")
    return checks, failures


def treatment_readiness_preflight(
    manifest_path: Path,
    *,
    compose_filename: str = "c2.compose.json",
    mcp_url: str = "",
    probe_timeout_seconds: float = 120.0,
    probe_fn: Any | None = None,
    readiness_retries: int = 3,
    readiness_retry_delay_seconds: float = 5.0,
) -> dict[str, Any]:
    """Emit a routing status contract for the treatment condition.

    Performs static digest-binding checks.  When *mcp_url* is non-empty,
    performs a bounded MCP initialize + catalog + describe probe (via
    *probe_fn* or the default ``deploy/smoke_remote.py`` runner) and
    classifies ``infrastructure_status``:

    * ``READY`` — probe reachable and every observed digest/version matches.
    * ``UNAVAILABLE`` — probe could not reach the endpoint.
    * ``MISCONFIGURED`` — probe reached the endpoint but digests/version
      mismatched or required tools are missing.

    ``routing_status`` is ``AVAILABLE_UNUSED`` when READY, otherwise
    ``CONFIGURED_UNCALLABLE`` (unreachable) or ``MISROUTED`` (mismatched).

    The probe is retried up to *readiness_retries* additional times with a
    bounded *readiness_retry_delay_seconds* sleep between attempts, so that
    a treatment container that is still starting up does not cause a
    premature ``UNAVAILABLE`` classification.  Each attempt uses the
    existing bounded ``run_operator_command`` abstraction; the total retry
    budget is bounded by ``(retries + 1) * probe_timeout_seconds + retries *
    readiness_retry_delay_seconds``.
    """

    _validate_readiness_retry_policy(readiness_retries, readiness_retry_delay_seconds)

    manifest = validate_manifest(manifest_path)
    manifest_digest = _digest(manifest_path)
    conditions = {item["id"]: item for item in manifest["conditions"]}
    treatment = conditions.get("C2")
    if not isinstance(treatment, dict):
        raise HarborSuiteError("treatment condition C2 is missing from manifest")
    image = str(treatment.get("image", ""))
    expected_server_version = str(treatment.get("server_version", ""))
    expected_catalog_digest = str(treatment.get("catalog_digest", ""))
    checks, failures = _static_binding_checks(treatment)
    probe: dict[str, Any] = {
        "reachable": False,
        "server_version_observed": None,
        "catalog_digest_observed": None,
        "tool_names": None,
        "discovery_matches": None,
        "probe_digest": _ZERO_DIGEST,
        "diagnostic": None,
    }
    if mcp_url:
        probe_result = _probe_until_ready(
            mcp_url=mcp_url,
            expected_version=expected_server_version,
            timeout_seconds=probe_timeout_seconds,
            probe_fn=_run_mcp_probe if probe_fn is None else probe_fn,
            retries=readiness_retries,
            retry_delay_seconds=readiness_retry_delay_seconds,
        )
        if probe_result.get("reachable"):
            report = probe_result["report"]
            probe = {
                "reachable": True,
                "server_version_observed": report.get("server", {}).get("version"),
                "catalog_digest_observed": report.get("catalog", {}).get(
                    "catalog_digest"
                ),
                "tool_names": report.get("tool_names"),
                "discovery_matches": report.get("discovery", {}).get("matches"),
                "probe_digest": _probe_digest(report),
                "diagnostic": None,
            }
        else:
            probe["diagnostic"] = probe_result.get("diagnostic", "probe failed")
    if probe["reachable"]:
        match_checks, match_failures = _probe_match_checks(
            probe,
            expected_server_version,
            expected_catalog_digest,
        )
        checks.update(match_checks)
        failures.extend(match_failures)
    elif mcp_url:
        failures.append("treatment MCP endpoint was not reachable")
    static_ok = all(
        checks[key]
        for key in (
            "image_digest_pinned",
            "catalog_digest_bound",
            "server_version_bound",
        )
    )
    if not mcp_url:
        failures.append("treatment MCP probe URL is not configured")
        infrastructure_status = "MISCONFIGURED"
        routing_status = "CONFIGURED_UNCALLABLE"
    elif not probe["reachable"]:
        infrastructure_status = "UNAVAILABLE"
        routing_status = "CONFIGURED_UNCALLABLE"
    elif static_ok and all(
        checks[key] is True
        for key in (
            "server_version_match",
            "catalog_digest_match",
            "required_tools_present",
            "describe_responded",
        )
    ):
        infrastructure_status = "READY"
        routing_status = "AVAILABLE_UNUSED"
    else:
        infrastructure_status = "MISCONFIGURED"
        routing_status = "MISROUTED"
    contract = {
        "schema_version": "2",
        "manifest_digest": manifest_digest,
        "condition_id": "C2",
        "infrastructure_status": infrastructure_status,
        "routing_status": routing_status,
        "treatment": {
            "image": image,
            "server_version": expected_server_version,
            "catalog_digest": expected_catalog_digest,
        },
        "routing": {"compose_file": compose_filename, "mcp_url": mcp_url},
        "probe": probe,
        "checks": checks,
        "failures": failures,
    }
    _validate_routing_contract(contract)
    return contract
