"""Owner-local exact public API contract for words."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.words")
    expected = (
        "FactorAnalysis",
        "FiniteWord",
        "PeriodAnalysis",
        "WordMorphism",
        "apply_morphism",
        "compose_morphisms",
        "conjugates",
        "factor_occurrences",
        "factors_of_length",
        "incidence_matrix",
        "parikh_vector",
        "periods",
        "prefix_function",
        "primitive_root",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
