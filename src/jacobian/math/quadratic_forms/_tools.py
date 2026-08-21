"""Quadratic form operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.quadratic_forms._models import (
    DirectSumRequest,
    DirectSumResult,
    DiscriminantRequest,
    DiscriminantResult,
    EvaluationRequest,
    EvaluationResult,
    RepresentationNumbersRequest,
    RepresentationNumbersResult,
    ScalingRequest,
    ScalingResult,
    SignatureRequest,
    SignatureResult,
    ThetaSeriesPrefixRequest,
    ThetaSeriesPrefixResult,
)
from jacobian.math.quadratic_forms._operations import (
    compute_direct_sum,
    compute_discriminant,
    compute_representation_numbers,
    compute_scaling,
    compute_signature,
    compute_theta_series_prefix,
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


_FORM_2D = {"matrix": [["1", "0"], ["0", "1"]]}

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
                {"form": _FORM_2D, "vector": ["3", "4"]},
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
    _op(
        "quadratic_form.representation_numbers.compute",
        "Compute representation numbers r(0), ..., r(bound)",
        "Compute the exact representation numbers r(n) for n = 0, 1, ..., bound by brute-force enumeration over a bounded integer box. The form must be positive-definite for finite counts.",
        RepresentationNumbersRequest,
        RepresentationNumbersResult,
        compute_representation_numbers,
        "quadratic-form",
        "exact",
        examples=(
            example(
                "rep_numbers_identity_2d_bound_2",
                "Representation numbers of I_2 up to 2 (positive-definite).",
                {"form": {"matrix": [["1", "0"], ["0", "1"]]}, "bound": 2},
            ),
        ),
    ),
    _op(
        "quadratic_form.theta_series_prefix.compute",
        "Compute the theta series prefix",
        "Compute the theta series prefix coefficients r(0), ..., r(bound) where r(n) is the number of representations of n by the quadratic form.",
        ThetaSeriesPrefixRequest,
        ThetaSeriesPrefixResult,
        compute_theta_series_prefix,
        "quadratic-form",
        "exact",
        examples=(
            example(
                "theta_prefix_identity_2d_bound_2",
                "Theta prefix of I_2 up to 2.",
                {"form": {"matrix": [["1", "0"], ["0", "1"]]}, "bound": 2},
            ),
        ),
    ),
    _op(
        "quadratic_form.scale.compute",
        "Scale a quadratic form by an integer factor",
        "Scale the symmetric matrix A by an integer factor, returning factor * A. Factor and entries must keep result within digit bounds.",
        ScalingRequest,
        ScalingResult,
        compute_scaling,
        "quadratic-form",
        "exact",
        examples=(
            example(
                "scale-i2-by-2",
                "Scale 2D identity by 2.",
                {"form": {"matrix": [["1", "0"], ["0", "1"]]}, "factor": 2},
            ),
        ),
    ),
    _op(
        "quadratic_form.direct_sum.compute",
        "Compute the direct sum of two quadratic forms",
        "Compute the block diagonal direct sum A ⊕ B of two quadratic forms. Combined dimension must not exceed 10.",
        DirectSumRequest,
        DirectSumResult,
        compute_direct_sum,
        "quadratic-form",
        "exact",
        examples=(
            example(
                "direct-sum-i1-i1",
                "Direct sum of two 1D forms [[1]] oplus [[1]].",
                {"form1": {"matrix": [["1"]]}, "form2": {"matrix": [["1"]]}},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
