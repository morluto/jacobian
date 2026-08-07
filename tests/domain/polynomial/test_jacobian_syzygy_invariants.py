from __future__ import annotations

import pytest

from jacobian.contracts.jacobian_syzygy import GradedJacobianSyzygyRequest
from jacobian.domains.polynomial.jacobian_syzygy import (
    compute_graded_jacobian_syzygy,
)


def test_syzygy_kernel_rejects_an_incomplete_linear_factor_request() -> None:
    request = GradedJacobianSyzygyRequest.model_construct(
        polynomial=None,
        linear_factors=None,
        linear_factor_variables=None,
        max_degree=0,
        coefficient_map_detail="CERTIFICATES",
    )

    with pytest.raises(ValueError, match="linear-factor input is incomplete"):
        compute_graded_jacobian_syzygy(request)
