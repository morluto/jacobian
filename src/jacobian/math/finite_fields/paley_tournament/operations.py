"""Paley tournament kernel."""

from __future__ import annotations

from jacobian.math.finite_fields.paley_tournament._models import (
    PaleyTournamentResult,
)

__all__ = ["construct_paley_tournament"]


def construct_paley_tournament(
    field_order: int,
) -> PaleyTournamentResult:
    """Construct the Paley tournament on F_q.

    For q � 3 (mod 4), the Paley tournament has x -> y iff y - x is a
    nonzero quadratic residue in F_q.
    """
    q = field_order
    if q < 3 or q % 4 != 3:
        raise ValueError("field order q must satisfy q >= 3 and q ≡ 3 (mod 4)")

    # Compute quadratic residues mod q
    residues: set[int] = set()
    for i in range(1, q):
        residues.add((i * i) % q)

    residues.discard(0)

    vertices = tuple(range(q))
    edges: list[tuple[int, int]] = []

    for x in range(q):
        for y in range(q):
            if x == y:
                continue
            diff = (y - x) % q
            if diff in residues:
                edges.append((x, y))

    return PaleyTournamentResult(
        field_order=q,
        vertices=vertices,
        edges=tuple(edges),
        edge_count=len(edges),
    )
