from pathlib import Path
from tomllib import loads

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


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


def test_local_diagnostics_are_excluded_from_source_distributions() -> None:
    configuration = loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    exclusions = configuration["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]

    assert "/.diagnostics" in exclusions
    assert "/.diagnostics/**" in exclusions
    assert ".diagnostics/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
