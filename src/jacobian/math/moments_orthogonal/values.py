"""Provider-independent exact values for moment-functional operations."""

from __future__ import annotations

from fractions import Fraction
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

        # Hankel structure: entries depend only on i+j.
        for i in range(side):
            for j in range(side):
                reference = self.entries[i][j]
                if any(
                    self.entries[a][b] != reference
                    for a in range(side)
                    for b in range(side)
                    if a + b == i + j
                ):
                    raise ValueError(
                        "entries along each anti-diagonal of a Hankel "
                        "matrix must be equal"
                    )
        matrix = [[value.as_fraction() for value in row] for row in self.entries]
        if self.determinant.as_fraction() != _rational_det(matrix):
            raise ValueError("determinant must match the retained entries")
        if self.rank != _rational_rank(matrix):
            raise ValueError("rank must match the retained entries")
        return self


class OrthogonalPolynomialTerm(StrictModel):
    """One monic orthogonal polynomial p_k(x) with its squared norm."""

    degree: int = Field(ge=0, le=MAX_POLYNOMIAL_DEGREE)
    # Schema-visible cap matching the family's degree bound: a degree-k
    # monic term carries exactly k+1 coefficients, so no admitted term can
    # exceed MAX_POLYNOMIAL_DEGREE + 1 entries.
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_POLYNOMIAL_DEGREE + 1
    )
    squared_norm: CanonicalRational


def _decompose_residual_in_basis(
    polynomials: tuple[OrthogonalPolynomialTerm, ...], k: int
) -> list[Fraction]:
    """Decompose ``R_k = x*p_k - p_{k+1}`` in the monic basis p_0..p_k.

    Every p_j is monic of exact degree j, so greedy division from the top
    degree downward yields the exact basis components; a nonzero remainder
    after division is impossible for a complete decomposition and raises.
    """
    from fractions import Fraction

    p_k = polynomials[k]
    p_next = polynomials[k + 1]
    shifted = [Fraction(0), *[c.as_fraction() for c in p_k.coefficients]]
    next_coeffs = [c.as_fraction() for c in p_next.coefficients]
    size = len(shifted)
    remainder = [
        shifted[i] - (next_coeffs[i] if i < len(next_coeffs) else Fraction(0))
        for i in range(size)
    ]
    components = [Fraction(0)] * size
    for j in range(size - 1, -1, -1):
        if remainder[j] == 0:
            continue
        if j >= k + 1:
            raise ValueError(
                f"residual x*p_{k} - p_{k + 1} leaves the span of the retained family"
            )
        basis = [c.as_fraction() for c in polynomials[j].coefficients]
        components[j] = remainder[j]
        for power, coefficient in enumerate(basis):
            remainder[power] -= components[j] * coefficient
    if any(coefficient != 0 for coefficient in remainder[: k + 1]):
        raise ValueError(
            f"p_{k + 1} does not satisfy the three-term recurrence against p_{k}"
        )
    return components


def _require_three_term_consistency(family: OrthogonalPolynomialFamily) -> None:
    """R_k must lie in span{p_{k-1}, p_k} with h_k = beta_k * h_{k-1}."""
    polynomials = family.polynomials
    for k in range(len(polynomials) - 1):
        components = _decompose_residual_in_basis(polynomials, k)
        lowest_free = max(k - 1, 0)
        if any(component != 0 for component in components[:lowest_free]):
            raise ValueError(
                f"p_{k + 1} does not satisfy the three-term recurrence "
                f"against p_{k}: the residual has a nonzero component "
                "below p_{k-1}"
            )
        if k >= 1:
            h_k = polynomials[k].squared_norm.as_fraction()
            h_prev = polynomials[k - 1].squared_norm.as_fraction()
            # Claimed orthogonality forces h_k = beta_k * h_{k-1}: reduce
            # h_k = <p_k, p_k> modulo the recurrence identities and the
            # vanishing cross products. Unlike a ratio, this multiplicative
            # form stays meaningful for degenerate norms - a vanishing
            # h_{k-1} forces h_k = 0 and a nonzero h_k with vanishing
            # h_{k-1} is unrealizable by any linear functional - so no
            # degenerate case escapes the norm relation.
            beta_k = components[k - 1]
            if h_k != beta_k * h_prev:
                raise ValueError(
                    "squared norms disagree with the three-term recurrence: "
                    f"beta_{k} must satisfy h_{k} = beta_{k} * h_{k - 1}"
                )


class OrthogonalPolynomialFamily(StrictModel):
    """A family of monic orthogonal polynomials p_0,...,p_n."""

    polynomials: tuple[OrthogonalPolynomialTerm, ...] = Field(
        min_length=1, max_length=MAX_POLYNOMIAL_DEGREE + 1
    )
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
        # Three-term consistency: R_k = x*p_k - p_{k+1} must lie in the
        # span of {p_{k-1}, p_k}. Because every p_j is monic, R_k is exactly
        # decomposable in the p_0,...,p_k basis by greedy division; nonzero
        # components below p_{k-1} break the recurrence. The component on
        # p_{k-1} IS beta_k, so it must also equal the norm ratio
        # h_k / h_{k-1}.
        _require_three_term_consistency(self)
        # Definiteness classifications are derived from the retained norms,
        # never free labels: quasi-definite means every norm nonzero, and
        # positive-definite means every norm positive.
        norms = [term.squared_norm.as_fraction() for term in self.polynomials]
        if self.is_quasi_definite != all(norm != 0 for norm in norms):
            raise ValueError(
                "is_quasi_definite must equal every squared norm being nonzero"
            )
        if self.is_positive_definite != all(norm > 0 for norm in norms):
            raise ValueError(
                "is_positive_definite must equal every squared norm being positive"
            )
        return self


class ThreeTermRecurrence(StrictModel):
    """Three-term recurrence coefficients: p_{k+1} = (x - alpha_k) p_k - beta_k p_{k-1}."""

    alpha: tuple[CanonicalRational, ...]
    beta: tuple[CanonicalRational, ...]
    variable: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_recurrence_dimensions(self) -> Self:
        """One beta per family term: len(beta) == len(alpha) + 1, with the
        unused zero placeholder at index 0."""
        if len(self.beta) != len(self.alpha) + 1:
            raise ValueError(
                "beta must carry exactly len(alpha) + 1 entries: one "
                "placeholder plus one ratio per recurrence step"
            )
        if self.beta[0].as_fraction() != 0:
            raise ValueError("beta[0] is the unused placeholder and must be zero")
        return self


class ChristoffelDarbouxKernel(StrictModel):
    """Christoffel-Darboux kernel K_m(x,y) = sum_{k=0}^m p_k(x) p_k(y) / h_k.

    The kernel is carried as the full bivariate coefficient matrix:
    ``coefficients[i][j]`` is the exact coefficient of ``x^i y^j``, bound
    to the family whose polynomials define the sum.
    """

    degree: int = Field(ge=0)
    coefficients: tuple[tuple[CanonicalRational, ...], ...]
    variable: str = Field(min_length=1, max_length=64)
    family: OrthogonalPolynomialFamily

    @model_validator(mode="after")
    def bind_kernel_to_family(self) -> Self:
        from jacobian.math.moments_orthogonal.operations import (
            _kernel_coefficient_matrix,
        )

        side = self.degree + 1
        if len(self.coefficients) != side or any(
            len(row) != side for row in self.coefficients
        ):
            raise ValueError(
                f"kernel coefficients must form a {side}x{side} matrix for "
                f"degree {self.degree}"
            )
        if self.variable != self.family.variable:
            raise ValueError("kernel variable must match the defining family")
        if self.degree >= len(self.family.polynomials):
            raise ValueError("kernel degree exceeds the retained family")
        # Exact replay of the defining sum against the retained family;
        # every kernel is symmetric by construction.
        expected = _kernel_coefficient_matrix(self.family.polynomials, self.degree)
        actual = tuple(
            tuple(value.as_fraction() for value in row) for row in self.coefficients
        )
        if actual != tuple(tuple(row) for row in expected):
            raise ValueError(
                "kernel coefficients must be the exact Christoffel-Darboux "
                "sum of the retained family"
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
        if len(self.prefix.moments) < 2 * self.order:
            raise ValueError(
                f"order {self.order} rules need at least {2 * self.order} "
                "source moments through mu_(2*order-1); a shorter retained "
                "prefix cannot establish the advertised exactness degree"
            )
        if len(self.nodes) != self.order:
            raise ValueError(
                f"order {self.order} rules carry exactly {self.order} nodes"
            )
        expected_degree = 2 * self.order - 1
        if self.exactness_degree != expected_degree:
            raise ValueError(
                f"exactness degree must be {expected_degree} for order {self.order}"
            )
        if self.variable != self.prefix.variable:
            raise ValueError(
                "quadrature variable must match the retained moment prefix"
            )
        # Pure-kernel replay: rebuild nodes and weights from the retained
        # prefix without constructing another result model.

        from jacobian.math.moments_orthogonal.operations import (
            _to_fraction,
        )

        p_n = _family_p_n_coefficients(self.prefix, self.order)
        moments = [_to_fraction(m) for m in self.prefix.moments]
        nodes_frac, weights = _construct_quadrature_rule(p_n, moments, self.order)
        if weights is None:
            raise ValueError("Vandermonde system is singular")
        # The operation's contract carries strictly positive weights only;
        # a payload replaying to a nonpositive rule is not an exact result
        # of this operation.
        if any(weight <= 0 for weight in weights):
            raise ValueError(
                "replayed quadrature rules must carry strictly positive weights"
            )
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


def _family_p_n_coefficients(
    prefix: MomentFunctionalPrefix, order: int
) -> list[Fraction]:
    """The monic p_order of the retained prefix, as exact fractions.

    Gaussian construction divides by norms only through p_{order-1}, so a
    vanishing terminal norm (finite-support measure) stays admissible;
    only the polynomial's coefficients are needed for the node replay.
    """
    from jacobian.math.moments_orthogonal.operations import (
        _construct_monic_orthogonal_polynomial,
    )

    return _construct_monic_orthogonal_polynomial(
        [m.as_fraction() for m in prefix.moments], order
    )


def _construct_quadrature_rule(
    p_n: list[Fraction], moments: list[Fraction], order: int
) -> tuple[list[Fraction], list[Fraction] | None]:
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

    @model_validator(mode="after")
    def bind_matrix_to_coefficients(self) -> Self:
        size = len(self.alphas)
        if len(self.betas) != size:
            raise ValueError(
                "betas must carry one entry per alpha: an unused zero "
                "placeholder followed by one norm ratio per recurrence step"
            )
        if self.betas and self.betas[0].as_fraction() != 0:
            raise ValueError("betas[0] is the unused placeholder and must be zero")
        if len(self.matrix) != size or any(len(row) != size for row in self.matrix):
            raise ValueError("matrix must be a square size x size array")
        for i in range(size):
            if self.matrix[i][i] != self.alphas[i]:
                raise ValueError(f"matrix diagonal must carry alpha_{i}")
            if i + 1 < size:
                if self.matrix[i + 1][i] != CanonicalRational.from_integer_ratio(1, 1):
                    raise ValueError("the monic subdiagonal must be exactly 1")
                if self.betas[i + 1] != self.matrix[i][i + 1]:
                    raise ValueError("matrix superdiagonal must carry beta_{i+1}")
        return self

    @model_validator(mode="after")
    def require_tridiagonal_band(self) -> Self:
        """Every entry outside the tridiagonal band must vanish; otherwise
        the value would claim a tridiagonal operator it does not carry."""
        if any(
            entry.as_fraction() != 0
            for i, row in enumerate(self.matrix)
            for j, entry in enumerate(row)
            if abs(i - j) > 1
        ):
            raise ValueError("entries outside the tridiagonal band must be zero")
        return self
