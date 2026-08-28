"""Exact public API contract for jacobian.math.logic.games.impartial."""

from __future__ import annotations

from jacobian.math.logic.games import impartial as impartial_games
from jacobian.math.logic.games.impartial import values


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the impartial_games public API."""
    expected = (
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
    )
    assert tuple(impartial_games.__all__) == expected
    assert len(impartial_games.__all__) == len(set(impartial_games.__all__))
    assert all(not name.startswith("_") for name in impartial_games.__all__)
    assert all(hasattr(impartial_games, name) for name in impartial_games.__all__)


def test_nim_option_rows_are_not_public_canonical_values() -> None:
    """Option rows are context-dependent and stay behind the result boundary."""
    assert "NimOption" not in values.__all__
    assert "NimOption" not in impartial_games.__all__
