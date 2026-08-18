"""Owner-local exact public API contract for impartial_games."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.impartial_games")
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
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
