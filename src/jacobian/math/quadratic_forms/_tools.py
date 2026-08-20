"""Quadratic form operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.quadratic_forms._models import (
    DiscriminantRequest,
    DiscriminantResult,
    EvaluationRequest,
    EvaluationResult,
    SignatureRequest,
    SignatureResult,
)
from jacobian.math.quadratic_forms._operations import (
    compute_discriminant,
    compute_signature,
    evaluate_form,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version="1",
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_FORM_2D = {"matrix": [[1, 0], [0, 1]]}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "quadratic_form.evaluate.compute",
        "Evaluate a quadratic form q(x) = x^T A x",
        "Compute the exact integer value q(x) = x^T A x for an "
        "integral quadratic form (symmetric matrix) and integer vector.",
        EvaluationRequest,
        EvaluationResult,
        evaluate_form,
        "algebra",
        "quadratic-form",
        "exact",
        examples=(
            example(
                "identity_2d_at_3_4",
                "Evaluate x^T I x at (3, 4); "
                "the matrix must be symmetric and the vector length must match.",
                {"form": _FORM_2D, "vector": [3, 4]},
            ),
        ),
    ),
    _op(
        "quadratic_form.discriminant.compute",
        "Compute the discriminant det(A) of a quadratic form",
        "Compute the exact determinant of the symmetric matrix "
        "representing the quadratic form, using SymPy for exact "
        "integer matrix computation.",
        DiscriminantRequest,
        DiscriminantResult,
        compute_discriminant,
        "algebra",
        "quadratic-form",
        "exact",
        examples=(
            example(
                "identity_2d_discriminant",
                "Compute det(I_2) = 1; the matrix must be symmetric.",
                {"form": _FORM_2D},
            ),
        ),
    ),
    _op(
        "quadratic_form.signature.compute",
        "Compute the signature/inertia of a quadratic form",
        "Compute the inertia (n_positive, n_negative, n_zero) of a "
        "quadratic form using SymPy eigenvalue computation, with "
        "definiteness classification.",
        SignatureRequest,
        SignatureResult,
        compute_signature,
        "algebra",
        "quadratic-form",
        "exact",
        examples=(
            example(
                "identity_2d_signature",
                "Compute the signature of I_2; the matrix must be symmetric.",
                {"form": _FORM_2D},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
