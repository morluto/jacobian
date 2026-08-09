import json
from pathlib import Path
from tomllib import loads

ROOT = Path(__file__).resolve().parents[3]
RELEASE_PLEASE_CONFIG = ROOT / "release-please-config.json"
SERVER_METADATA = ROOT / "server.json"
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
RELEASE_PLEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release-please.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_release_build_resolves_and_verifies_one_immutable_sha() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "release:\n    types: [published]" in source
    assert "workflow_dispatch:" in source
    assert "release_sha: ${{ steps.release.outputs.sha }}" in source
    assert "ref: ${{ steps.release.outputs.sha }}" in source
    assert 'test "$(git rev-parse HEAD)" = "$RELEASE_SHA"' in source
    assert (
        "actions/workflows/ci.yml/runs?head_sha=$RELEASE_SHA&status=success" in source
    )
    assert source.index("Require successful CI for release commit") < source.index(
        "Build Python distributions"
    )


def test_release_candidate_dispatches_full_ci_after_lockfile_sync() -> None:
    release_please = RELEASE_PLEASE_WORKFLOW.read_text(encoding="utf-8")
    ci = CI_WORKFLOW.read_text(encoding="utf-8")

    ci_triggers = ci.split("on:", 1)[1].split("concurrency:", 1)[0]
    assert "workflow_dispatch:" in ci_triggers
    assert "plan_output=$(.github/scripts/classify-ci-paths --force-exhaustive)" in ci

    lockfile_sync = release_please.index("name: Synchronize release lockfile")
    candidate_dispatch = release_please.index("name: Dispatch release candidate CI")
    assert lockfile_sync < candidate_dispatch

    dispatch = release_please[candidate_dispatch:]
    assert "steps.release.outputs.prs_created == 'true'" in dispatch
    assert "gh workflow run ci.yml" in dispatch
    assert '--ref "$RELEASE_BRANCH"' in dispatch


def test_mcp_publisher_is_verified_before_oidc_or_publication() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    publisher = source[source.index("publish-mcp:") :]

    assert "MCP_PUBLISHER_VERSION: v1.8.0" in publisher
    assert (
        "1370446bbe74d562608e8005a6ccce02d146a661fbd78674e11cc70b9618d6cf" in publisher
    )
    assert (
        "c978982c60e1b4903a976de090f04dc4fac4a320daa50704fcad2dbc93433d62" in publisher
    )
    checksum = publisher.index("sha256sum --check --strict")
    extraction = publisher.index('tar xzf "$archive" mcp-publisher')
    source_check = publisher.index("Verify immutable release commit")
    oidc = publisher.index("login github-oidc")
    publication = publisher.index("./mcp-publisher publish")

    assert source_check < checksum < extraction < oidc < publication


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
