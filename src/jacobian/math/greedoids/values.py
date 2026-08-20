"""Provider-independent values for exact greedoid/antimatroid operations.

A finite greedoid is a pair ``G = (E, F)`` where ``E`` is a finite ground set
and ``F`` is a family of subsets (the *feasible sets*) satisfying:

(G1) the empty set is feasible;
(G2) *accessibility*: every nonempty ``X in F`` has some ``x in X`` with
    ``X \\ {x}`` in ``F``;
(G3) *exchange*: for ``X, Y in F`` with ``|X| > |Y|``, some ``x in X \\ Y``
    has ``Y union {x}`` in ``F``.

The complete feasible-set family is authoritative. No caller-supplied
membership oracle or heuristic exchange flag is allowed.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_GROUND_SIZE = 64
MAX_FEASIBLE_COUNT = 4096


def _check_feasible_row(row: tuple[int, ...], n: int) -> None:
    if not row:
        return
    if list(row) != sorted(row):
        raise ValueError("each feasible set must be a sorted index tuple")
    if len(set(row)) != len(row):
        raise ValueError("feasible sets must not repeat an element")
    if any(not 0 <= i < n for i in row):
        raise ValueError("feasible-set index out of range")


def _check_family_unique(feasible: tuple[tuple[int, ...], ...]) -> None:
    if not feasible:
        return
    seen: set[tuple[int, ...]] = set()
    for row in feasible:
        if row in seen:
            raise ValueError("feasible-set family must be duplicate-free")
        seen.add(row)


class FiniteFeasibleSetSystem(StrictModel):
    """An immutable complete finite feasible-set family.

    ``ground`` is a tuple of unique ground labels. ``feasible`` is a tuple of
    feasible subsets, each a sorted tuple of ground indices (positions in
    ``ground``). The family is authoritative: omission means exact
    infeasibility, not unknown.
    """

    ground: tuple[str, ...] = Field(min_length=1)
    feasible: tuple[tuple[int, ...], ...] = Field(default=())

    @model_validator(mode="after")
    def require_well_formed(self) -> Self:
        if len(self.ground) > MAX_GROUND_SIZE:
            raise ValueError("ground size exceeds the bounded budget")
        if len(set(self.ground)) != len(self.ground):
            raise ValueError("ground labels must be unique")
        if len(self.feasible) > MAX_FEASIBLE_COUNT:
            raise ValueError("feasible-set count exceeds the bounded budget")
        n = len(self.ground)
        for row in self.feasible:
            _check_feasible_row(row, n)
        _check_family_unique(self.feasible)
        return self

    def feasible_index(self) -> dict[tuple[int, ...], int]:
        return {row: i for i, row in enumerate(self.feasible)}


__all__ = [
    "MAX_FEASIBLE_COUNT",
    "MAX_GROUND_SIZE",
    "FiniteFeasibleSetSystem",
]
