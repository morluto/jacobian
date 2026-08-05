from __future__ import annotations

from typing import cast

import pytest

from jacobian.contracts.lean import LeanEnvironment
from jacobian.lean_frontend.service import LeanService
from jacobian.references import LeanCheckerInstallation, reference_catalog


def test_partial_lean_service_explains_how_to_list_environments() -> None:
    service = LeanService.__new__(LeanService)
    service.installations = {}

    with pytest.raises(
        ValueError,
        match=r"math\.find with capability_id='lean\.check'",
    ):
        service.verify(
            statement="1 + 1 = 2",
            proof="rfl",
            environment=LeanEnvironment.CORE,
        )


def test_reference_catalog_omits_incomplete_lean_installation() -> None:
    incomplete = cast(
        LeanCheckerInstallation,
        object(),
    )

    catalog = reference_catalog(
        {},
        lean={LeanEnvironment.MATHLIB: incomplete},
    )

    assert "lean4" not in catalog
