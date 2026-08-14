"""SymPy-backed exact operations for sparse rational polynomial maps."""

from jacobian.polynomials.installation import install_polynomial_operations
from jacobian.polynomials.resources import PolynomialInstallation, PolynomialResources

__all__ = [
    "PolynomialInstallation",
    "PolynomialResources",
    "install_polynomial_operations",
]
