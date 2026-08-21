"""Exact public API contract for jacobian.math.impartial_games."""

from __future__ import annotations

from jacobian.math import impartial_games


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the impartial_games public API."""
    expected = (
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
    )
    assert tuple(impartial_games.__all__) == expected
    assert len(impartial_games.__all__) == len(set(impartial_games.__all__))
    assert all(not name.startswith("_") for name in impartial_games.__all__)
    assert all(hasattr(impartial_games, name) for name in impartial_games.__all__)
