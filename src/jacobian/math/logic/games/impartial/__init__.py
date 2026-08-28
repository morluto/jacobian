"""Exact bounded native APIs for finite impartial games."""

from jacobian.math.logic.games.impartial.operations import (
    GrundyAnalysis,
    SubtractionGrundyAnalysis,
    birthdays,
    grundy_classes,
    grundy_table,
    mex,
    nim_options,
    nim_sum,
    outcome_profile,
    position_grundy,
    subtraction_game,
    subtraction_grundy_prefix,
)
from jacobian.math.logic.games.impartial.values import (
    GameMove,
    ImpartialGame,
    NimPosition,
)

__all__ = [
    "GameMove",
    "GrundyAnalysis",
    "ImpartialGame",
    "NimPosition",
    "SubtractionGrundyAnalysis",
    "birthdays",
    "grundy_classes",
    "grundy_table",
    "mex",
    "nim_options",
    "nim_sum",
    "outcome_profile",
    "position_grundy",
    "subtraction_game",
    "subtraction_grundy_prefix",
]
