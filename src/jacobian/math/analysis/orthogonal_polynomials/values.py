"""Provider-independent exact values for moment-functional operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.matrices.values import RationalMatrix


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"moments_orthogonal.{reason}", message)


MAX_MOMENT_DEGREE = 128
MAX_HANKEL_ORDER = 64
MAX_POLYNOMIAL_DEGREE = 32
MAX_QUADRATURE_ORDER = 16
MAX_VARIABLE_LENGTH = 64


class MomentFunctionalPrefix(StrictModel):
    """A bounded prefix of a linear functional L(x^k) = mu_k over QQ."""

    moments: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_MOMENT_DEGREE + 1
    )
    variable: str = Field(min_length=1, max_length=MAX_VARIABLE_LENGTH)

    @property
    def degree(self) -> int:
        return len(self.moments) - 1


class HankelMomentMatrix(StrictModel):
    """A source-bound ordinary or shifted exact Hankel matrix."""

    prefix: MomentFunctionalPrefix
    shift: Literal[0, 1]
    row_axis: tuple[int, ...] = Field(min_length=1, max_length=MAX_HANKEL_ORDER + 1)
    column_axis: tuple[int, ...] = Field(min_length=1, max_length=MAX_HANKEL_ORDER + 1)
    matrix: RationalMatrix
    determinant: CanonicalRational
    rank: int = Field(ge=0)

    @model_validator(mode="after")
    def require_axes_match_matrix(self) -> Self:
        """Validate carrier shape and axes without replaying claims."""
        if self.row_axis != tuple(range(self.matrix.row_count)):
            raise _validation_error(
                "value_invariant",
                "row axis must be the ordered axis of the canonical matrix",
            )
        if self.column_axis != tuple(range(self.matrix.column_count)):
            raise _validation_error(
                "value_invariant",
                "column axis must be the ordered axis of the canonical matrix",
            )
        if self.matrix.row_count != self.matrix.column_count:
            raise _validation_error("value_invariant", "Hankel matrix must be square")
        return self

    @property
    def order(self) -> int:
        return self.matrix.row_count - 1

    @classmethod
    def _from_kernel(
        cls,
        *,
        prefix: MomentFunctionalPrefix,
        order: int,
        shift: Literal[0, 1],
        matrix: RationalMatrix,
        determinant: CanonicalRational,
        rank: int,
    ) -> Self:
        """Build a trusted result without replaying exact elimination."""
        return cls.model_construct(
            prefix=prefix,
            shift=shift,
            row_axis=tuple(range(order + 1)),
            column_axis=tuple(range(order + 1)),
            matrix=matrix,
            determinant=determinant,
            rank=rank,
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
    variable: str = Field(min_length=1, max_length=MAX_VARIABLE_LENGTH)
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
        return cls.model_construct(
            polynomials=polynomials,
            variable=variable,
            is_quasi_definite=is_quasi_definite,
            is_positive_definite=is_positive_definite,
        )


class ThreeTermRecurrence(StrictModel):
    """Three-term recurrence coefficients: p_{k+1} = (x - alpha_k) p_k - beta_k p_{k-1}."""

    alpha: tuple[CanonicalRational, ...]
    beta: tuple[CanonicalRational, ...]
    variable: str = Field(min_length=1, max_length=MAX_VARIABLE_LENGTH)

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
        return cls.model_construct(alpha=alpha, beta=beta, variable=variable)


class ChristoffelDarbouxKernel(StrictModel):
    """Christoffel-Darboux kernel K_m(x,y) = sum_{k=0}^m p_k(x) p_k(y) / h_k.

    The kernel is carried as the full bivariate coefficient matrix:
    ``coefficients[i][j]`` is the exact coefficient of ``x^i y^j``, bound
    to the family whose polynomials define the sum.
    """

    degree: int = Field(ge=0)
    coefficients: tuple[tuple[CanonicalRational, ...], ...]
    variable: str = Field(min_length=1, max_length=MAX_VARIABLE_LENGTH)
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
        return cls.model_construct(
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
    variable: str = Field(min_length=1, max_length=MAX_VARIABLE_LENGTH)
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
        return cls.model_construct(
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
    """A source-bound finite Jacobi matrix in the monic polynomial basis."""

    family: OrthogonalPolynomialFamily
    recurrence: ThreeTermRecurrence
    row_axis: tuple[int, ...] = Field(
        min_length=0, max_length=MAX_POLYNOMIAL_DEGREE + 1
    )
    column_axis: tuple[int, ...] = Field(
        min_length=0, max_length=MAX_POLYNOMIAL_DEGREE + 1
    )
    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_axes_match_matrix(self) -> Self:
        """Validate only source dimensions and ordered matrix axes."""
        size = len(self.recurrence.alpha)
        if self.row_axis != tuple(range(size)):
            raise _validation_error(
                "value_invariant", "row axis must be the ordered axis of the matrix"
            )
        if self.column_axis != tuple(range(size)):
            raise _validation_error(
                "value_invariant", "column axis must be the ordered axis of the matrix"
            )
        if self.matrix.row_count != size or self.matrix.column_count != size:
            raise _validation_error(
                "value_invariant", "matrix dimensions must match the recurrence"
            )
        return self

    @property
    def alphas(self) -> tuple[CanonicalRational, ...]:
        """Compatibility view of the retained recurrence coefficients."""
        return self.recurrence.alpha

    @property
    def betas(self) -> tuple[CanonicalRational, ...]:
        """Compatibility view of the retained recurrence coefficients."""
        return self.recurrence.beta

    @property
    def variable(self) -> str:
        """Compatibility view of the recurrence variable."""
        return self.recurrence.variable

    @classmethod
    def _from_kernel(
        cls,
        *,
        family: OrthogonalPolynomialFamily,
        recurrence: ThreeTermRecurrence,
        matrix: RationalMatrix,
    ) -> Self:
        return cls.model_construct(
            family=family,
            recurrence=recurrence,
            row_axis=tuple(range(matrix.row_count)),
            column_axis=tuple(range(matrix.column_count)),
            matrix=matrix,
        )
