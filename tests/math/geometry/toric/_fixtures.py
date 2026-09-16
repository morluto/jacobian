"""Shared exact fixtures for toric-geometry tests."""

from __future__ import annotations

from jacobian.math.geometry.toric._models import (
    CharacterVector,
    ToricFanPresentation,
)

A2_RAYS = ((1, 0), (0, 1))
A2_CONES = ((), (0,), (1,), (0, 1))

P2_RAYS = ((1, 0), (0, 1), (-1, -1))
P2_CONES = ((), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2))

DP6_RAYS = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1))
DP6_CONES = (
    (),
    (0,),
    (1,),
    (2,),
    (3,),
    (4,),
    (5,),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 4),
    (2, 5),
    (3, 5),
)

# The affine chart of A^2 / Z_2: det((1,0),(1,2)) = 2, so the two-dimensional
# cone is simplicial but not smooth.
SINGULAR_RAYS = ((1, 0), (1, 2))
SINGULAR_CONES = ((), (0,), (1,), (0, 1))

# Cone over the square with vertices (1,+-1,+-1): a non-simplicial
# four-generator cone in rank three.
SQUARE_RAYS = ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1))
SQUARE_CONES = (
    (),
    (0,),
    (1,),
    (2,),
    (3,),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (0, 1, 2, 3),
)

# The complete fan of P^4 in rank four: all proper subsets of five rays.
P4_RAYS = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (-1, -1, -1, -1))


def _p4_cones() -> tuple[tuple[int, ...], ...]:
    from itertools import combinations

    cones: list[tuple[int, ...]] = []
    for size in range(5):
        cones.extend(combinations(range(5), size))
    return tuple(cones)


P4_CONES = _p4_cones()


def fan(
    rays: tuple[tuple[int, ...], ...],
    cones: tuple[tuple[int, ...], ...],
    rank: int | None = None,
) -> ToricFanPresentation:
    if rank is None:
        rank = len(rays[0]) if rays else 2
    return ToricFanPresentation(lattice_rank=rank, rays=rays, cones=cones)


def character(*entries: int) -> CharacterVector:
    return CharacterVector(entries=entries)


def a2_fan() -> ToricFanPresentation:
    return fan(A2_RAYS, A2_CONES)


def p2_fan() -> ToricFanPresentation:
    return fan(P2_RAYS, P2_CONES)


def dp6_fan() -> ToricFanPresentation:
    return fan(DP6_RAYS, DP6_CONES)


def singular_fan() -> ToricFanPresentation:
    return fan(SINGULAR_RAYS, SINGULAR_CONES)


def square_fan() -> ToricFanPresentation:
    return fan(SQUARE_RAYS, SQUARE_CONES, rank=3)


def p4_fan() -> ToricFanPresentation:
    return fan(P4_RAYS, P4_CONES, rank=4)


__all__ = [
    "A2_CONES",
    "A2_RAYS",
    "DP6_CONES",
    "DP6_RAYS",
    "P2_CONES",
    "P2_RAYS",
    "P4_CONES",
    "P4_RAYS",
    "SINGULAR_CONES",
    "SINGULAR_RAYS",
    "SQUARE_CONES",
    "SQUARE_RAYS",
    "a2_fan",
    "character",
    "dp6_fan",
    "fan",
    "p2_fan",
    "p4_fan",
    "singular_fan",
    "square_fan",
]
