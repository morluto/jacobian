"""Provider-independent exact values for moment-functional operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_MOMENT_DEGREE = 64
MAX_HANKEL_ORDER = 32
MAX_POLYNOMIAL_DEGREE = 32
MAX_QUADRATURE_ORDER = 16


class MomentFunctionalPrefix(StrictModel):
    """A bounded prefix of a linear functional L(x^k) = mu_k over QQ."""

    moments: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_MOMENT_DEGREE + 1
    )
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

    @model_validator(mode="after")
    def bind_determinant_and_rank_to_entries(self) -> Self:
        side = self.order + 1
        if len(self.entries) != side or any(len(row) != side for row in self.entries):
            raise ValueError(
                f"entries must form a {side}x{side} matrix for order {self.order}"
            )
        from jacobian.math.moments_orthogonal.operations import (
            _rational_det,
            _rational_rank,
        )

        matrix = [[value.as_fraction() for value in row] for row in self.entries]
        if self.determinant.as_fraction() != _rational_det(matrix):
            raise ValueError("determinant must match the retained entries")
        if self.rank != _rational_rank(matrix):
            raise ValueError("rank must match the retained entries")
        return self


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

    @model_validator(mode="after")
    def require_canonical_family(self) -> Self:
        for index, term in enumerate(self.polynomials):
            if term.degree != index:
                raise ValueError(
                    f"polynomial at position {index} declares degree {term.degree}; "
                    "families must be contiguous from degree 0"
                )
            if len(term.coefficients) != term.degree + 1:
                raise ValueError(
                    f"p_{term.degree} must carry exactly degree+1 coefficients"
                )
            if term.coefficients[-1].as_fraction() != 1:
                raise ValueError(f"p_{term.degree} must be monic")
            if term.squared_norm.as_fraction() == 0:
                raise ValueError(
                    f"p_{index} has zero squared norm; a quasi-definite "
                    "family requires nonzero norms"
                )
        return self


class ThreeTermRecurrence(StrictModel):
    """Three-term recurrence coefficients: p_{k+1} = (x - alpha_k) p_k - beta_k p_{k-1}."""

    alpha: tuple[CanonicalRational, ...]
    beta: tuple[CanonicalRational, ...]
    variable: str = Field(min_length=1, max_length=64)


class ChristoffelDarbouxKernel(StrictModel):
    """Christoffel-Darboux kernel K_m(x,y) = sum_{k=0}^m p_k(x) p_k(y) / h_k.

    The kernel is carried as the full bivariate coefficient matrix:
    ``coefficients[i][j]`` is the exact coefficient of ``x^i y^j``.
    """

    degree: int = Field(ge=0)
    coefficients: tuple[tuple[CanonicalRational, ...], ...]
    variable: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_square_bivariate_matrix(self) -> Self:
        side = self.degree + 1
        if len(self.coefficients) != side or any(
            len(row) != side for row in self.coefficients
        ):
            raise ValueError(
                f"kernel coefficients must form a {side}x{side} matrix for "
                f"degree {self.degree}"
            )
        return self


class QuadratureNode(StrictModel):
    """One node and weight of a Gaussian quadrature rule."""

    node: CanonicalRational
    weight: CanonicalRational


class GaussianQuadratureRule(StrictModel):
    """An exact Gaussian quadrature rule bound to its source prefix."""

    order: int = Field(ge=1, le=MAX_QUADRATURE_ORDER)
    nodes: tuple[QuadratureNode, ...]
    variable: str = Field(min_length=1, max_length=64)
    exactness_degree: int = Field(ge=0)
    prefix: MomentFunctionalPrefix

    @model_validator(mode="after")
    def bind_rule_to_source(self) -> Self:
        if len(self.nodes) != self.order:
            raise ValueError(
                f"order {self.order} rules carry exactly {self.order} nodes"
            )
        expected_degree = 2 * self.order - 1
        if self.exactness_degree != expected_degree:
            raise ValueError(
                f"exactness degree must be {expected_degree} for order {self.order}"
            )
        # Pure-kernel replay: rebuild nodes and weights from the retained
        # prefix without constructing another result model.

        from jacobian.math.moments_orthogonal.operations import (
            _to_fraction,
        )

        p_n = [
            _to_fraction(c) for c in _family_p_n(self.prefix, self.order).coefficients
        ]
        moments = [_to_fraction(m) for m in self.prefix.moments]
        nodes_frac, weights = _construct_quadrature_rule(p_n, moments, self.order)
        if tuple(self.nodes) != tuple(
            QuadratureNode(
                node=CanonicalRational.from_fraction(node),
                weight=CanonicalRational.from_fraction(weight),
            )
            for node, weight in zip(nodes_frac, weights, strict=True)
        ):
            raise ValueError(
                "quadrature nodes must be the exact rule of the retained moment prefix"
            )
        return self


def _family_p_n(prefix, order: int):
    from jacobian.math.moments_orthogonal._models import OrthogonalPolynomialRequest
    from jacobian.math.moments_orthogonal.operations import (
        compute_orthogonal_polynomials,
    )

    family = compute_orthogonal_polynomials(
        OrthogonalPolynomialRequest(prefix=prefix, max_degree=order)
    )
    return family.polynomials[order]


def _construct_quadrature_rule(p_n, moments, order):
    from jacobian.math.moments_orthogonal.operations import (
        _construct_quadrature_rule as _kernel,
    )

    return _kernel(p_n, moments, order)


__all__ = [
    "MAX_HANKEL_ORDER",
    "MAX_MOMENT_DEGREE",
    "MAX_POLYNOMIAL_DEGREE",
    "MAX_QUADRATURE_ORDER",
    "ChristoffelDarbouxKernel",
    "GaussianQuadratureRule",
    "HankelMomentMatrix",
    "JacobiMatrix",
    "MomentFunctionalPrefix",
    "OrthogonalPolynomialFamily",
    "OrthogonalPolynomialTerm",
    "QuadratureNode",
    "ThreeTermRecurrence",
]


class JacobiMatrix(StrictModel):
    """Finite tridiagonal multiplication-by-x matrix in the monic basis.

    With x p_k = p_{k+1} + alpha_k p_k + beta_k p_{k-1}, the subdiagonal
    carries the monic normalization 1 and the superdiagonal carries
    beta_{i+1}; ``betas`` keeps the recurrence convention with an unused
    placeholder first.
    """

    alphas: tuple[CanonicalRational, ...]
    betas: tuple[CanonicalRational, ...]
    matrix: tuple[tuple[CanonicalRational, ...], ...]
    variable: str = Field(min_length=1, max_length=64)
