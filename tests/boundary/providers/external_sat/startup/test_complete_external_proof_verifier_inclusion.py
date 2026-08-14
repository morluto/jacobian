"""One complete-portfolio smoke for both external proof-verifier families."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.boundary.providers.external_sat.external_sat_support import (
    fake_carcara,
    fake_drat_trim,
)
from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.catalog_build_runtime import create_catalog_build_runtime

from jacobian.providers.external_solver_runtime import (
    carcara_provider_runtime,
    drat_trim_provider_runtime,
)


def test_complete_portfolio_includes_authorized_external_proof_verifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drat_trim = drat_trim_provider_runtime(
        fake_drat_trim(tmp_path, "print('s VERIFIED')\nraise SystemExit(0)")
    )
    carcara = carcara_provider_runtime(
        fake_carcara(tmp_path, "print('valid')\nraise SystemExit(0)")
    )
    monkeypatch.setattr(
        "jacobian.catalog_build.drat_trim_provider_runtime",
        lambda *_args, **_kwargs: drat_trim,
    )
    monkeypatch.setattr(
        "jacobian.catalog_build.carcara_provider_runtime",
        lambda *_args, **_kwargs: carcara,
    )

    with create_catalog_build_runtime(
        tmp_path / "complete-state",
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as complete:
        operation_ids = {
            descriptor.operation_id
            for descriptor in complete.core.operations.snapshot().operations
        }

    assert {"sat.unsat_proof.verify", "smt.unsat_proof.verify"} <= operation_ids
