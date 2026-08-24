"""Exact public API contract for ``jacobian.math.quadratic_forms``."""

from jacobian.math import quadratic_forms


def test_exact_public_api_symbols() -> None:
    expected = (
        "QuadraticCrossTerm",
        "RationalCoordinateVector",
        "RationalQuadraticForm",
        "evaluate_rational_quadratic_form",
    )
    assert tuple(quadratic_forms.__all__) == expected
    assert len(quadratic_forms.__all__) == len(set(quadratic_forms.__all__))
    assert all(not name.startswith("_") for name in quadratic_forms.__all__)
    assert all(hasattr(quadratic_forms, name) for name in quadratic_forms.__all__)
