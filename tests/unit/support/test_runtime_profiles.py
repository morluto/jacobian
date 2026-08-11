"""Unit tests for typed complete-runtime profiles."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.runtime_profiles import (
    ATTACHED_COMPUTE,
    ATTACHED_COMPUTE_READ_ONLY,
    AUTHORIZED_VERIFY,
    AUTHORIZED_VERIFY_READ_ONLY,
    FRESH_LIFECYCLE,
    RuntimeTestProfile,
    open_runtime_for,
)
from tests.support.state import copy_template, publish_template

from jacobian.runtime import CheckerAuthorityMode


def test_read_only_attach_still_copies_away_from_session_template(
    tmp_path: Path,
) -> None:
    """Store open mutates SQLite/layout; never attach directly to the template."""

    template = tmp_path / "template"

    def build(staging: Path) -> None:
        (staging / "marker.txt").write_text("portfolio", encoding="utf-8")
        (staging / "blobs" / "sha256").mkdir(parents=True)
        (staging / "metadata.sqlite3").write_bytes(b"")

    publish_template(template, build)
    before = sorted(path.relative_to(template) for path in template.rglob("*"))

    private = copy_template(template, tmp_path / "private")
    assert private != template
    assert sorted(path.relative_to(template) for path in template.rglob("*")) == before
    assert (private / "marker.txt").read_text(encoding="utf-8") == "portfolio"
    assert ATTACHED_COMPUTE_READ_ONLY.state_access == "READ_ONLY"
    assert ATTACHED_COMPUTE.state_access == "PRIVATE_MUTABLE"


def test_runtime_profile_presets_encode_orthogonal_dimensions() -> None:
    assert ATTACHED_COMPUTE.installation == "ATTACH_TEMPLATE"
    assert ATTACHED_COMPUTE.checker_authority is CheckerAuthorityMode.NONE
    assert ATTACHED_COMPUTE.state_access == "PRIVATE_MUTABLE"

    assert ATTACHED_COMPUTE_READ_ONLY.state_access == "READ_ONLY"
    assert ATTACHED_COMPUTE_READ_ONLY.checker_authority is CheckerAuthorityMode.NONE

    assert AUTHORIZED_VERIFY.checker_authority is CheckerAuthorityMode.HYDRATE_EXISTING
    assert AUTHORIZED_VERIFY.state_access == "PRIVATE_MUTABLE"
    assert AUTHORIZED_VERIFY_READ_ONLY.state_access == "READ_ONLY"

    assert FRESH_LIFECYCLE.installation == "FRESH"
    assert FRESH_LIFECYCLE.state_access == "LIFECYCLE_OWNER"


def test_fresh_profile_rejects_non_lifecycle_state_access(tmp_path: Path) -> None:
    bad = RuntimeTestProfile(
        installation="FRESH",
        checker_authority=CheckerAuthorityMode.NONE,
        state_access="PRIVATE_MUTABLE",
    )
    with pytest.raises(ValueError, match="LIFECYCLE_OWNER"):
        next(open_runtime_for(bad, tmp_path=tmp_path))
