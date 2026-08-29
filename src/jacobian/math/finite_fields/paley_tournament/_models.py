"""Typed contracts for the Paley tournament operation."""

from jacobian._models import StrictModel


class PaleyTournamentRequest(StrictModel):
    """Request to construct the Paley tournament of F_q."""

    field_order: int


class PaleyTournamentResult(StrictModel):
    """The Paley tournament on F_q."""

    field_order: int
    vertices: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    edge_count: int


__all__ = [
    "PaleyTournamentRequest",
    "PaleyTournamentResult",
]
