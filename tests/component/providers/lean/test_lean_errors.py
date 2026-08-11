from __future__ import annotations

import pytest

from jacobian.contracts.lean import LeanEnvironment
from jacobian.lean_frontend.service import LeanService


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
