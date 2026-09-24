"""Exact gauge-invariant observables on bounded permutation-group holonomies."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.gauge._models import (
    PermutationWilsonTraceRequest,
    PermutationWilsonTraceResult,
)
from jacobian.math.gauge.operations import path_holonomy


def permutation_wilson_trace(
    request: PermutationWilsonTraceRequest,
) -> PermutationWilsonTraceResult:
    """Return the natural permutation-representation character of one loop.

    The exact trace of the permutation matrix of ``sigma`` is the number of
    points fixed by ``sigma``.  Holonomy is computed from the supplied bounded
    field/path so this result does not rely on a caller-asserted holonomy.
    """
    if not isinstance(request, PermutationWilsonTraceRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="lattice_gauge.permutation_wilson.request_type",
            message="expected a typed permutation Wilson trace request",
        )
    holonomy = path_holonomy(request.field, request.path)
    if holonomy.start != holonomy.end:
        raise OperationDomainValidationError(
            location=("path",),
            code="lattice_gauge.permutation_wilson.open_path",
            message="a Wilson loop character requires a closed path",
        )
    image = holonomy.holonomy.image
    trace = sum(point == image[point] for point in range(holonomy.holonomy.degree))
    return PermutationWilsonTraceResult(
        field=request.field,
        path=request.path,
        holonomy=holonomy.holonomy,
        trace=trace,
    )


__all__ = ["permutation_wilson_trace"]
