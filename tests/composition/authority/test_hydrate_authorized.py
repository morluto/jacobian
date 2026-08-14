"""Hydrate verify adapters from an already-authorized store without reauth."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import atomic_installation, open_domain_services

from jacobian.checker_authorization import install_polytope_checkers
from jacobian.domains.matrix_lattice import matrix_operations
from jacobian.runtime.resources import RuntimeResources


def _verify_ids(core: RuntimeResources) -> set[str]:
    return {
        entry.operation_id
        for entry in core.operations.snapshot().operations
        if ".verify" in entry.operation_id
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
    root = tmp_path / "state"
    bundle = matrix_operations()
    with open_exact_domain_services(root, bundle):
        pass
    baseline_audit = _audit_count(root)

    with open_exact_domain_services(
        root,
        bundle,
        checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING,
    ) as hydrated:
        assert {
            "matrix.multiply.verify",
            "matrix.determinant.verify",
        } <= _verify_ids(hydrated.core)
        assert _audit_count(root) == baseline_audit


def test_hydrate_authorized_on_empty_store_is_fail_closed(tmp_path: Path) -> None:
    bundle = matrix_operations()
    with open_exact_domain_services(
        tmp_path,
        bundle,
        checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING,
    ) as services:
        assert _audit_count(tmp_path) == 0
        assert "matrix.determinant.verify" not in _verify_ids(services.core)


def test_hydrate_authorized_polytope_checkers_without_complete_portfolio(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    with open_domain_services(
        root,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        polytope = services.polytope
        with atomic_installation(services.core):
            installed = install_polytope_checkers(
                services.core.checkers,
                claim_schema_uri=polytope.claim_schema_uri,
                semantics_uri=polytope.semantics_uri,
                point_schema_uri=polytope.point_schema_uri,
            )
        assert installed.witness_checker_id is not None
        assert installed.certificate_checker_id is not None
        baseline_audit = _audit_count(root)

    with open_domain_services(
        root,
        checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING,
    ) as hydrated:
        polytope = hydrated.polytope
        with atomic_installation(hydrated.core):
            rebound = install_polytope_checkers(
                hydrated.core.checkers,
                claim_schema_uri=polytope.claim_schema_uri,
                semantics_uri=polytope.semantics_uri,
                point_schema_uri=polytope.point_schema_uri,
            )
        assert rebound.witness_checker_id == installed.witness_checker_id
        assert rebound.certificate_checker_id == installed.certificate_checker_id
        assert hydrated.core.checkers.select_compatible(
            evidence_kind="WITNESS",
            format_id="polytope.convex_combination",
            format_version="1",
            claim_schema_uri=polytope.claim_schema_uri,
            semantics_uri=polytope.semantics_uri,
            candidate_schema_uri=polytope.point_schema_uri,
        )
        assert hydrated.core.checkers.select_compatible(
            evidence_kind="CERTIFICATE",
            format_id="polytope.linear_separator",
            format_version="1",
            claim_schema_uri=polytope.claim_schema_uri,
            semantics_uri=polytope.semantics_uri,
            candidate_schema_uri=polytope.point_schema_uri,
        )
        assert _audit_count(root) == baseline_audit
