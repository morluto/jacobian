"""Homogeneous progression set system constructor."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.discrepancy._models import FiniteSetSystem
from jacobian.math.combinatorics.discrepancy.homogeneous_progression._models import (
    MAX_N,
)

__all__ = ["construct_homogeneous_progression_set_system"]


def _validate_n(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool):
        raise PydanticCustomError("discrepancy.n_type", "n must be an integer")
    if not 0 <= n <= MAX_N:
        raise PydanticCustomError(
            "discrepancy.n_too_large",
            f"n must be between 0 and {MAX_N}",
        )


def construct_homogeneous_progression_set_system(
    n: int,
) -> FiniteSetSystem:
    """Construct the homogeneous progression set system on [n].

    The ground set is indexed by 0..n-1 (representing 1..n). The sets are
    the zero-based images of homogeneous progressions {d, 2d, ..., kd}
    for every d, k >= 1 with dk <= n, in canonical order.
    """
    try:
        _validate_n(n)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=("n",), code=error.type, message=str(error)
        ) from error

    sets: list[tuple[int, ...]] = []
    for d in range(1, n + 1):
        k = 1
        while d * k <= n:
            progression = tuple(d * i - 1 for i in range(1, k + 1))
            sets.append(progression)
            k += 1

    return FiniteSetSystem(ground_set_size=n, sets=tuple(sets))
