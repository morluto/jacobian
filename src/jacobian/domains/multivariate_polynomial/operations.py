"""Domain adapter for multivariate polynomial operations."""

from __future__ import annotations

from jacobian.contracts.multivariate_polynomial import (
    MultivariateGCDRequest,
    MultivariateGCDResult,
    MultivariateResultantRequest,
    MultivariateResultantResult,
)
from jacobian.math.multivariate_polynomial import (
    multivariate_gcd,
    multivariate_resultant,
)


def compute_multivariate_gcd(request: MultivariateGCDRequest) -> MultivariateGCDResult:
    gcd = multivariate_gcd(  # type: ignore[no-untyped-call]
        request.left.expression,
        list(request.left.variables),
        request.right.expression,
        list(request.right.variables),
    )
    return MultivariateGCDResult(gcd=gcd)


def compute_multivariate_resultant(
    request: MultivariateResultantRequest,
) -> MultivariateResultantResult:
    res = multivariate_resultant(  # type: ignore[no-untyped-call]
        request.left.expression,
        list(request.left.variables),
        request.right.expression,
        list(request.right.variables),
        request.eliminate_variable,
    )
    return MultivariateResultantResult(resultant=res)
