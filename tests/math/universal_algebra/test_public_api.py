"""Exact public API contract for jacobian.math.universal_algebra."""

from __future__ import annotations

from jacobian.math import universal_algebra


def test_exact_public_api_symbols() -> None:
    expected = (
        "FiniteAlgebra",
        "FlatTerm",
        "OperationSymbol",
        "Term",
        "congruence_check",
        "equation_profile",
        "evaluate_term",
        "generated_subalgebra",
        "quotient",
    )
    assert tuple(universal_algebra.__all__) == expected
    assert len(universal_algebra.__all__) == len(set(universal_algebra.__all__))
    assert all(not name.startswith("_") for name in universal_algebra.__all__)
    assert all(hasattr(universal_algebra, name) for name in universal_algebra.__all__)
