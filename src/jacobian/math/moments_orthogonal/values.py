"""Provider-independent exact values for moment-functional operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"moments_orthogonal.{reason}", message)


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
            raise _validation_error(
                "value_invariant",
                f"entries must form a {side}x{side} matrix for order {self.order}",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        order: int,
        entries: tuple[tuple[CanonicalRational, ...], ...],
        determinant: CanonicalRational,
        rank: int,
        variable: str,
    ) -> Self:
        """Build a trusted matrix result without replaying exact elimination."""
        return cls(
            order=order,
            entries=entries,
            determinant=determinant,
            rank=rank,
            variable=variable,
        )


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
                raise _validation_error(
                    "value_invariant",
                    f"polynomial at position {index} declares degree {term.degree}; "
                    "families must be contiguous from degree 0",
                )
            if len(term.coefficients) != term.degree + 1:
                raise _validation_error(
                    "value_invariant",
                    f"p_{term.degree} must carry exactly degree+1 coefficients",
                )
            if term.coefficients[-1].as_fraction() != 1:
                raise _validation_error(
                    "value_invariant", f"p_{term.degree} must be monic"
                )
        # Definiteness classifications are derived from the retained norms,
        # never free labels: quasi-definite means every norm nonzero, and
        # positive-definite means every norm positive.
        norms = [term.squared_norm.as_fraction() for term in self.polynomials]
        if self.is_quasi_definite != all(norm != 0 for norm in norms):
            raise _validation_error(
                "value_invariant",
                "is_quasi_definite must equal every squared norm being nonzero",
            )
        if self.is_positive_definite != all(norm > 0 for norm in norms):
            raise _validation_error(
                "value_invariant",
                "is_positive_definite must equal every squared norm being positive",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        polynomials: tuple[OrthogonalPolynomialTerm, ...],
        variable: str,
        is_quasi_definite: bool,
        is_positive_definite: bool,
    ) -> Self:
        return cls(
            polynomials=polynomials,
            variable=variable,
            is_quasi_definite=is_quasi_definite,
            is_positive_definite=is_positive_definite,
        )


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
            raise _validation_error(
                "value_invariant",
                "beta must carry exactly len(alpha) + 1 entries: one "
                "placeholder plus one ratio per recurrence step",
            )
        if self.beta[0].as_fraction() != 0:
            raise _validation_error(
                "value_invariant", "beta[0] is the unused placeholder and must be zero"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        alpha: tuple[CanonicalRational, ...],
        beta: tuple[CanonicalRational, ...],
        variable: str,
    ) -> Self:
        return cls(alpha=alpha, beta=beta, variable=variable)


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
        side = self.degree + 1
        if len(self.coefficients) != side or any(
            len(row) != side for row in self.coefficients
        ):
            raise _validation_error(
                "value_invariant",
                f"kernel coefficients must form a {side}x{side} matrix for "
                f"degree {self.degree}",
            )
        if self.variable != self.family.variable:
            raise _validation_error(
                "value_invariant", "kernel variable must match the defining family"
            )
        if self.degree >= len(self.family.polynomials):
            raise _validation_error(
                "value_invariant", "kernel degree exceeds the retained family"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        degree: int,
        coefficients: tuple[tuple[CanonicalRational, ...], ...],
        variable: str,
        family: OrthogonalPolynomialFamily,
    ) -> Self:
        return cls(
            degree=degree,
            coefficients=coefficients,
            variable=variable,
            family=family,
        )


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
            raise _validation_error(
                "value_invariant",
                f"order {self.order} rules need at least {2 * self.order} "
                "source moments through mu_(2*order-1); a shorter retained "
                "prefix cannot establish the advertised exactness degree",
            )
        if len(self.nodes) != self.order:
            raise _validation_error(
                "value_invariant",
                f"order {self.order} rules carry exactly {self.order} nodes",
            )
        expected_degree = 2 * self.order - 1
        if self.exactness_degree != expected_degree:
            raise _validation_error(
                "value_invariant",
                f"exactness degree must be {expected_degree} for order {self.order}",
            )
        if self.variable != self.prefix.variable:
            raise _validation_error(
                "value_invariant",
                "quadrature variable must match the retained moment prefix",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        order: int,
        nodes: tuple[QuadratureNode, ...],
        variable: str,
        prefix: MomentFunctionalPrefix,
    ) -> Self:
        return cls(
            order=order,
            nodes=nodes,
            variable=variable,
            exactness_degree=2 * order - 1,
            prefix=prefix,
        )


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
            raise _validation_error(
                "value_invariant",
                "betas must carry one entry per alpha: an unused zero "
                "placeholder followed by one norm ratio per recurrence step",
            )
        if self.betas and self.betas[0].as_fraction() != 0:
            raise _validation_error(
                "value_invariant", "betas[0] is the unused placeholder and must be zero"
            )
        if len(self.matrix) != size or any(len(row) != size for row in self.matrix):
            raise _validation_error(
                "value_invariant", "matrix must be a square size x size array"
            )
        for i in range(size):
            if self.matrix[i][i] != self.alphas[i]:
                raise _validation_error(
                    "value_invariant", f"matrix diagonal must carry alpha_{i}"
                )
            if i + 1 < size:
                if self.matrix[i + 1][i] != CanonicalRational.from_integer_ratio(1, 1):
                    raise _validation_error(
                        "value_invariant", "the monic subdiagonal must be exactly 1"
                    )
                if self.betas[i + 1] != self.matrix[i][i + 1]:
                    raise _validation_error(
                        "value_invariant", "matrix superdiagonal must carry beta_{i+1}"
                    )
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
            raise _validation_error(
                "value_invariant", "entries outside the tridiagonal band must be zero"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        alphas: tuple[CanonicalRational, ...],
        betas: tuple[CanonicalRational, ...],
        matrix: tuple[tuple[CanonicalRational, ...], ...],
        variable: str,
    ) -> Self:
        return cls(
            alphas=alphas,
            betas=betas,
            matrix=matrix,
            variable=variable,
        )
