import json
from pathlib import Path
from tomllib import loads

import yaml

ROOT = Path(__file__).resolve().parents[2]
RELEASE_PLEASE_CONFIG = ROOT / "release-please-config.json"
SERVER_METADATA = ROOT / "server.json"
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
RELEASE_PLEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release-please.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow() -> dict[str, object]:
    value = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _job(workflow: dict[str, object], name: str) -> dict[str, object]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs[name]
    assert isinstance(job, dict)
    return job


def _named_step(job: dict[str, object], name: str) -> dict[str, object]:
    steps = job["steps"]
    assert isinstance(steps, list)
    step = next(
        step for step in steps if isinstance(step, dict) and step.get("name") == name
    )
    assert isinstance(step, dict)
    return step


def _step_names(job: dict[str, object]) -> list[str]:
    steps = job["steps"]
    assert isinstance(steps, list)
    return [step["name"] for step in steps if isinstance(step, dict) and "name" in step]


def test_release_build_resolves_and_verifies_one_immutable_sha() -> None:
    workflow = _workflow()
    triggers = workflow[True]
    assert isinstance(triggers, dict)
    assert triggers["release"] == {"types": ["published"]}
    assert "workflow_dispatch" in triggers

    build = _job(workflow, "build")
    outputs = build["outputs"]
    assert isinstance(outputs, dict)
    assert outputs["release_sha"] == "${{ steps.release.outputs.sha }}"

    steps = build["steps"]
    assert isinstance(steps, list)
    checkout = next(
        step
        for step in steps
        if isinstance(step, dict)
        and isinstance(step.get("uses"), str)
        and step["uses"].startswith("actions/checkout@")
    )
    checkout_with = checkout["with"]
    assert isinstance(checkout_with, dict)
    assert checkout_with["ref"] == "${{ steps.release.outputs.sha }}"

    verification = _named_step(build, "Verify release commit")
    assert verification["run"] == 'test "$(git rev-parse HEAD)" = "$RELEASE_SHA"'
    ci_gate = _named_step(build, "Require successful CI for release commit")
    ci_gate_run = ci_gate["run"]
    assert isinstance(ci_gate_run, str)
    assert (
        "actions/workflows/ci.yml/runs?head_sha=$RELEASE_SHA&status=success"
        in ci_gate_run
    )
    assert _step_names(build).index(
        "Require successful CI for release commit"
    ) < _step_names(build).index("Build Python distributions")


def test_release_candidate_dispatches_full_ci_after_lockfile_sync() -> None:
    release_please = RELEASE_PLEASE_WORKFLOW.read_text(encoding="utf-8")
    ci = CI_WORKFLOW.read_text(encoding="utf-8")

    ci_triggers = ci.split("on:", 1)[1].split("concurrency:", 1)[0]
    assert "workflow_dispatch:" in ci_triggers
    assert "name: required" in ci

    lockfile_sync = release_please.index("name: Synchronize release lockfile")
    candidate_dispatch = release_please.index("name: Dispatch release candidate CI")
    assert lockfile_sync < candidate_dispatch

    dispatch = release_please[candidate_dispatch:]
    assert "steps.release.outputs.prs_created == 'true'" in dispatch
    assert "gh workflow run ci.yml" in dispatch
    assert '--ref "$RELEASE_BRANCH"' in dispatch


def test_mcp_publisher_is_verified_before_oidc_or_publication() -> None:
    publisher = _job(_workflow(), "publish-mcp")
    install = _named_step(publisher, "Install MCP Registry publisher")
    environment = install["env"]
    assert isinstance(environment, dict)
    assert environment["MCP_PUBLISHER_VERSION"] == "v1.8.0"
    script = install["run"]

    assert isinstance(script, str)
    assert "1370446bbe74d562608e8005a6ccce02d146a661fbd78674e11cc70b9618d6cf" in script
    assert "c978982c60e1b4903a976de090f04dc4fac4a320daa50704fcad2dbc93433d62" in script
    assert script.index("sha256sum --check --strict") < script.index(
        'tar xzf "$archive" mcp-publisher'
    )

    steps = _step_names(publisher)
    assert (
        steps.index("Verify immutable release commit")
        < steps.index("Install MCP Registry publisher")
        < steps.index("Authenticate to MCP Registry")
        < steps.index("Publish server metadata")
    )


def test_release_please_updates_all_mcp_server_versions() -> None:
    configuration = json.loads(RELEASE_PLEASE_CONFIG.read_text(encoding="utf-8"))
    extra_files = configuration["packages"]["."]["extra-files"]
    server_updates = [
        entry
        for entry in extra_files
        if isinstance(entry, dict) and entry["path"] == "server.json"
    ]

    assert {entry["jsonpath"] for entry in server_updates} == {"$..version"}

    metadata = json.loads(SERVER_METADATA.read_text(encoding="utf-8"))
    assert metadata["version"] == metadata["packages"][0]["version"]


def test_local_diagnostics_are_excluded_from_source_distributions() -> None:
    configuration = loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    exclusions = configuration["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]

    assert "/.diagnostics" in exclusions
    assert "/.diagnostics/**" in exclusions
    assert ".diagnostics/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
