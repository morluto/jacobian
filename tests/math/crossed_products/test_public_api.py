"""Exact public API contract for jacobian.math.crossed_products."""

from jacobian.math import crossed_products


def test_exact_public_api_symbols() -> None:
    assert tuple(crossed_products.__all__) == (
        "FiniteCosetCrossedProductElement",
        "FiniteCosetCrossedProductPresentation",
        "FiniteCosetCrossedProductTerm",
        "multiply",
    )
    assert len(crossed_products.__all__) == len(set(crossed_products.__all__))
    assert all(not name.startswith("_") for name in crossed_products.__all__)
    assert all(hasattr(crossed_products, name) for name in crossed_products.__all__)
