"""Typed wire contracts for the quadratic-form evaluation operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.quadratic_forms.values import (
    MAX_QUADRATIC_EVALUATION_DIGITS,
    MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS,
    MAX_QUADRATIC_EVALUATION_TERM_DIGITS,
    RationalCoordinateVector,
    RationalQuadraticForm,
    evaluate_rational_quadratic_form,
    require_evaluation_budget,
)


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
            raise ValueError("vector axis must equal the quadratic-form axis")
        require_evaluation_budget(self.form, self.vector)
        return self


class EvaluationResult(StrictModel):
    """A source-bound exact value of ``Q(vector)``."""

    form: RationalQuadraticForm
    vector: RationalCoordinateVector
    value: CanonicalRational

    @model_validator(mode="after")
    def require_exact_source_bound_evaluation(self) -> Self:
        require_evaluation_budget(self.form, self.vector)
        require_bounded_rational(
            self.value,
            max_digits=MAX_QUADRATIC_EVALUATION_DIGITS,
            label="quadratic-form evaluation",
        )
        expected = evaluate_rational_quadratic_form(self.form, self.vector)
        if self.value.as_fraction() != expected:
            raise ValueError("value must equal the exact quadratic-form evaluation")
        return self


__all__ = ["EvaluationRequest", "EvaluationResult"]
