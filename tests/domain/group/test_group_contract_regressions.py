from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.group import GroupOrbitRequest


def test_group_orbit_contract_binds_the_point_to_the_declared_degree() -> None:
    with pytest.raises(ValidationError, match="point"):
        GroupOrbitRequest(degree=2, generators=((1, 0),), point=3)
