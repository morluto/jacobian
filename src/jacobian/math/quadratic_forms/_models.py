"""Typed wire contracts for the quadratic-form evaluation operation."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.quadratic_forms.values import (
    MAX_QUADRATIC_EVALUATION_DIGITS,
    RationalCoordinateVector,
    RationalQuadraticForm,
    evaluate_rational_quadratic_form,
    require_evaluation_budget,
)


class EvaluationRequest(StrictModel):
    """Evaluate one rational quadratic form at an axis-matched rational vector."""

    form: RationalQuadraticForm
    vector: RationalCoordinateVector

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
