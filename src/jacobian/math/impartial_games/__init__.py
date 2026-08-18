"""Exact bounded native APIs for finite impartial games."""

from jacobian.math.impartial_games.operations import (
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
from jacobian.math.impartial_games.values import GameMove, ImpartialGame

__all__ = [
    "GameMove",
    "GrundyAnalysis",
    "ImpartialGame",
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
