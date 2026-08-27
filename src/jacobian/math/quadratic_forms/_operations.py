"""Private MCP adapters for exact quadratic-form evaluation."""

from pydantic import ValidationError

from jacobian.math.quadratic_forms._models import EvaluationRequest, EvaluationResult
from jacobian.math.quadratic_forms.values import evaluate_rational_quadratic_form


def evaluate_form(request: EvaluationRequest) -> EvaluationResult:
    """Evaluate the request's form exactly with direct rational arithmetic."""

    return EvaluationResult._from_kernel(
        request,
        value=evaluate_rational_quadratic_form(request.form, request.vector),
    )


def _verify_evaluation_result(result: EvaluationResult) -> bool:
    """Check an independently supplied result inside the request envelope."""

    try:
        request = EvaluationRequest(form=result.form, vector=result.vector)
    except ValidationError:
        return False
    return result.value.as_fraction() == evaluate_rational_quadratic_form(
        request.form, request.vector
    )


__all__ = ["evaluate_form"]
