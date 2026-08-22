"""Exact public API contract for jacobian.math.moments_orthogonal."""

from __future__ import annotations

from jacobian.math import moments_orthogonal


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the moments_orthogonal public API."""
    expected = (
        "ChristoffelDarbouxKernel",
        "GaussianQuadrature",
        "HankelMatrix",
        "JacobiMatrix",
        "RecurrenceCoefficients",
        "christoffel_darboux",
        "gaussian_quadrature",
        "hankel_matrix",
        "jacobi_matrix",
        "recurrence_coefficients",
    )
    assert tuple(moments_orthogonal.__all__) == expected
    assert len(moments_orthogonal.__all__) == len(set(moments_orthogonal.__all__))
    assert all(not name.startswith("_") for name in moments_orthogonal.__all__)
    assert all(hasattr(moments_orthogonal, name) for name in moments_orthogonal.__all__)
