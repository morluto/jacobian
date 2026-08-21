"""Provider-independent exact values for moment-functional operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_MOMENT_DEGREE = 64
MAX_HANKEL_ORDER = 32
MAX_POLYNOMIAL_DEGREE = 32
MAX_QUADRATURE_ORDER = 16


class MomentFunctionalPrefix(StrictModel):
    """A bounded prefix of a linear functional L(x^k) = mu_k over QQ."""

    moments: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=MAX_MOMENT_DEGREE + 1)
    variable: str = Field(min_length=1, max_length=64)

    @property
    def degree(self) -> int:
        return len(self.moments) - 1


class HankelMomentMatrix(StrictModel):
    """The exact Hankel matrix H_r[i,j] = mu_(i+j)."""

    order: int = Field(ge=0, le=MAX_HANKEL_ORDER)
    entries: tuple[tuple[CanonicalRational, ...], ...]
    determinant: CanonicalRational
    rank: int = Field(ge=0)
    variable: str = Field(min_length=1, max_length=64)


class OrthogonalPolynomialTerm(StrictModel):
    """One monic orthogonal polynomial p_k(x) with its squared norm."""

    degree: int = Field(ge=0)
    coefficients: tuple[CanonicalRational, ...] = Field(min_length=1)
    squared_norm: CanonicalRational


class OrthogonalPolynomialFamily(StrictModel):
    """A family of monic orthogonal polynomials p_0,...,p_n."""

    polynomials: tuple[OrthogonalPolynomialTerm, ...] = Field(min_length=1)
    variable: str = Field(min_length=1, max_length=64)
    is_quasi_definite: bool
    is_positive_definite: bool


class ThreeTermRecurrence(StrictModel):
    """Three-term recurrence coefficients: p_{k+1} = (x - alpha_k) p_k - beta_k p_{k-1}."""

    alpha: tuple[CanonicalRational, ...]
    beta: tuple[CanonicalRational, ...]
    variable: str = Field(min_length=1, max_length=64)


class ChristoffelDarbouxKernel(StrictModel):
    """Christoffel-Darboux kernel K_m(x,y) = sum_{k=0}^m p_k(x) p_k(y) / h_k."""

    degree: int = Field(ge=0)
    numerator_x_coefficients: tuple[CanonicalRational, ...]
    numerator_y_coefficients: tuple[CanonicalRational, ...]
    variable: str = Field(min_length=1, max_length=64)


class QuadratureNode(StrictModel):
    """One node and weight of a Gaussian quadrature rule."""

    node: CanonicalRational
    weight: CanonicalRational


class GaussianQuadratureRule(StrictModel):
    """An exact Gaussian quadrature rule."""

    order: int = Field(ge=1, le=MAX_QUADRATURE_ORDER)
    nodes: tuple[QuadratureNode, ...]
    variable: str = Field(min_length=1, max_length=64)
    exactness_degree: int = Field(ge=0)


__all__ = [
    "JacobiMatrix",
    "ChristoffelDarbouxKernel",
    "GaussianQuadratureRule",
    "HankelMomentMatrix",
    "MAX_HANKEL_ORDER",
    "MAX_MOMENT_DEGREE",
    "MAX_POLYNOMIAL_DEGREE",
    "MAX_QUADRATURE_ORDER",
    "MomentFunctionalPrefix",
    "OrthogonalPolynomialFamily",
    "OrthogonalPolynomialTerm",
    "QuadratureNode",
    "ThreeTermRecurrence",
]


class JacobiMatrix(StrictModel):
    """Finite tridiagonal multiplication-by-x matrix in the monic basis."""

    alphas: tuple[CanonicalRational, ...]
    betas: tuple[CanonicalRational, ...]
    matrix: tuple[tuple[CanonicalRational, ...], ...]
    variable: str = Field(min_length=1, max_length=64)
