"""Typed wire contracts for the quadratic-form evaluation operation."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.number_theory.quadratic_forms.general.values import (
    MAX_QUADRATIC_EVALUATION_DIGITS,
    MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS,
    MAX_QUADRATIC_EVALUATION_TERM_DIGITS,
    RationalCoordinateVector,
    RationalQuadraticForm,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"quadratic_form.{reason}", message)


class EvaluationRequest(StrictModel):
    """Evaluate one rational quadratic form at an axis-matched rational vector.

    Admission runs in ``require_evaluation_budget`` before any arithmetic:
    per-entry digit bounds are stated on the nested value fields, and the
    form and vector field descriptions publish the total-support and
    aggregate-denominator envelopes a schema-valid request must satisfy.
    """

    form: RationalQuadraticForm = Field(
        description=(
            "Form on its declared axis; admission additionally caps the "
            "total materialized support (diagonal coefficients plus cross "
            f"terms) at {MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS} terms."
        ),
    )
    vector: RationalCoordinateVector = Field(
        description=(
            "Vector whose axis equals the form axis; over the active "
            "monomials (nonzero coefficient at nonzero coordinates) the "
            "aggregate denominator digits d -- active coefficient-"
            "denominator digits plus twice the touched coordinate-denominator "
            f"digits -- must satisfy d + {MAX_QUADRATIC_EVALUATION_TERM_DIGITS} "
            f"+ len(str(t)) <= {MAX_QUADRATIC_EVALUATION_DIGITS}, where t is "
            "the active term count."
        ),
    )

    @model_validator(mode="after")
    def require_shared_axis(self) -> Self:
        if self.vector.axis != self.form.axis:
            raise _validation_error(
                "axis_mismatch", "vector axis must equal the quadratic-form axis"
            )
        return self


class EvaluationResult(StrictModel):
    """A source-bound exact value of ``Q(vector)``."""

    form: RationalQuadraticForm
    vector: RationalCoordinateVector
    value: CanonicalRational

    @model_validator(mode="after")
    def require_exact_source_bound_evaluation(self) -> Self:
        try:
            require_bounded_rational(
                self.value,
                max_digits=MAX_QUADRATIC_EVALUATION_DIGITS,
                label="quadratic-form evaluation",
            )
        except ValueError as error:
            raise _validation_error("evaluation_budget", str(error)) from error
        return self

    @classmethod
    def _from_kernel(cls, request: EvaluationRequest, *, value: Fraction) -> Self:
        """Build one result after the admitted rational kernel established it."""

        return cls.model_construct(
            form=request.form,
            vector=request.vector,
            value=CanonicalRational.from_fraction(value),
        )


MAX_COEFFICIENT_MATRIX_AXIS = 128


class CoefficientMatrixRequest(StrictModel):
    """Return the symmetric matrix of one rational quadratic form.

    Admission runs in ``require_coefficient_matrix_budget`` before any
    arithmetic: the form dimension ``n`` must satisfy ``n <= 128``, so the
    dense ``n x n`` result stays within the exact-output envelope. Per-entry
    coefficient bounds are stated on the nested form value fields.
    """

    form: RationalQuadraticForm = Field(
        description=(
            "Form on its declared axis; admission additionally requires the "
            f"axis length to stay within {MAX_COEFFICIENT_MATRIX_AXIS} so the "
            "dense symmetric result fits the output envelope."
        ),
    )


class CoefficientMatrixResult(StrictModel):
    """The symmetric matrix ``A`` with ``Q(x) = x^T A x`` for a source form.

    The form axis is the shared row and column axis: row/column ``i`` means
    variable ``axis[i]``. The half-polar convention is fixed: ``A[i][i]`` is
    the ``x_i^2`` polynomial coefficient ``a_i`` and, for ``i != j``,
    ``A[i][j] = A[j][i] = c_ij / 2`` where ``c_ij`` is the ``x_i*x_j``
    polynomial coefficient (absent cross terms contribute zero). An odd
    cross-term coefficient therefore yields half-integral matrix entries.
    """

    form: RationalQuadraticForm
    matrix: RationalMatrix
    convention: Literal["Q_X_EQUALS_X_TRANSPOSE_A_X"] = "Q_X_EQUALS_X_TRANSPOSE_A_X"

    @model_validator(mode="after")
    def require_matrix_shape_matches_axis(self) -> Self:
        dimension = len(self.form.axis)
        if self.matrix.row_count != dimension or self.matrix.column_count != dimension:
            raise _validation_error(
                "matrix_shape",
                "coefficient matrix must be square on the quadratic-form axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: CoefficientMatrixRequest,
        *,
        matrix: RationalMatrix,
    ) -> Self:
        """Build one result after the admitted kernel established its entries."""

        return cls.model_construct(
            form=request.form,
            matrix=matrix,
            convention="Q_X_EQUALS_X_TRANSPOSE_A_X",
        )


__all__ = [
    "CoefficientMatrixRequest",
    "CoefficientMatrixResult",
    "EvaluationRequest",
    "EvaluationResult",
]
