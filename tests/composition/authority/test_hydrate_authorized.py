"""Hydrate verify adapters from an already-authorized store without reauth."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.support.state import copy_template

from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime


def _verify_ids(runtime: JacobianRuntime) -> set[str]:
    return {
        entry.capability_id
        for entry in runtime.core.capabilities.catalog().capabilities
        if ".verify" in entry.capability_id
    }


def _audit_count(root: Path) -> int:
    connection = sqlite3.connect(root / "metadata.sqlite3")
    try:
        row = connection.execute("SELECT COUNT(*) FROM checker_audit").fetchone()
    finally:
        connection.close()
    assert row is not None
    return int(row[0])


def test_hydrate_authorized_matches_bundled_authority_without_audit(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "seed"
    with create_runtime(
        seed, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    ) as authorized:
        expected = _verify_ids(authorized)
        baseline_audit = _audit_count(seed)

    attached = copy_template(seed, tmp_path / "attached")
    with create_runtime(
        attached, checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING
    ) as hydrated:
        assert _verify_ids(hydrated) == expected
        assert _audit_count(attached) == baseline_audit


def test_hydrate_authorized_on_empty_store_is_fail_closed(tmp_path: Path) -> None:
    with create_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING
    ) as runtime:
        assert _audit_count(tmp_path) == 0
        # Atomic / resource verify surfaces may still appear; domain checkers must not.
        assert "polynomial.gcd.verify" not in _verify_ids(runtime)
        assert "sat.model.verify" not in _verify_ids(runtime)
        assert "matrix.determinant.verify" not in _verify_ids(runtime)


def test_authorized_runtime_hydrates_reference_checkers(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    ids = _verify_ids(authorized_complete_runtime)
    assert "sat.model.verify" in ids
    assert "polynomial.gcd.verify" in ids
    assert "matrix.multiply.verify" in ids
