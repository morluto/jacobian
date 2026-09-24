"""Exact public API contract for ``jacobian.math.number_theory.quadratic_forms.general``."""

from jacobian.math.number_theory.quadratic_forms import general as quadratic_forms


def test_exact_public_api_symbols() -> None:
    expected = (
        "FiniteBoxProfileRequest",
        "FiniteBoxProfileResult",
        "FiniteGaussSumRequest",
        "FiniteGaussSumResult",
        "QuadraticCrossTerm",
        "QuadraticFormDirectSumRequest",
        "QuadraticFormDirectSumResult",
        "QuadraticFormRestrictionRequest",
        "QuadraticFormRestrictionResult",
        "RationalCoordinateVector",
        "RationalQuadraticForm",
        "ThetaSeriesPrefixRequest",
        "ThetaSeriesPrefixResult",
        "bilinear_pairing",
        "coefficient_matrix",
        "coefficient_matrix_entries",
        "evaluate_rational_quadratic_form",
        "finite_box_value_profile",
        "finite_quadratic_gauss_sum",
        "quadratic_form_direct_sum",
        "quadratic_form_restrict_coordinates",
        "require_coefficient_matrix_budget",
        "theta_series_prefix",
    )
    assert tuple(quadratic_forms.__all__) == expected
    assert len(quadratic_forms.__all__) == len(set(quadratic_forms.__all__))
    assert all(not name.startswith("_") for name in quadratic_forms.__all__)
    assert all(hasattr(quadratic_forms, name) for name in quadratic_forms.__all__)
