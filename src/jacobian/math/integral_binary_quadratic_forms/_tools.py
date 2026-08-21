"""Integral binary quadratic form operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.integral_binary_quadratic_forms._models import (
    BinaryQuadraticFormCheckRequest,
    BinaryQuadraticFormCheckResult,
    BinaryQuadraticFormEvaluateRequest,
    BinaryQuadraticFormEvaluateResult,
    BinaryQuadraticFormProperEquivRequest,
    BinaryQuadraticFormReducedClassesRequest,
    BinaryQuadraticFormReduceRequest,
    ProperEquivalenceResult,
    ReducedBinaryQuadraticFormResult,
    ReducedClassesResult,
)
from jacobian.math.integral_binary_quadratic_forms._operations import (
    compute_check,
    compute_evaluate,
    compute_proper_equivalence,
    compute_reduce,
    compute_reduced_classes,
)


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "number_theory.binary_quadratic_form.check",
        "Check a binary quadratic form",
        "Check if integer coefficients (a,b,c) form a primitive positive-definite "
        "binary quadratic form Q(x,y) = a*x^2 + b*x*y + c*y^2 with negative "
        "discriminant D = b^2 - 4ac.",
        BinaryQuadraticFormCheckRequest,
        BinaryQuadraticFormCheckResult,
        compute_check,
        "number-theory",
        "exact",
        examples=(
            example(
                "check_1_1_1",
                "Check [1,1,1]: discriminant -3, primitive positive definite.",
                {"a": 1, "b": 1, "c": 1},
            ),
        ),
    ),
    _op(
        "number_theory.binary_quadratic_form.evaluate",
        "Evaluate a binary quadratic form",
        "Evaluate Q(x,y) = a*x^2 + b*x*y + c*y^2 at an integer pair (x,y) "
        "and determine primitive status.",
        BinaryQuadraticFormEvaluateRequest,
        BinaryQuadraticFormEvaluateResult,
        compute_evaluate,
        "number-theory",
        "exact",
        examples=(
            example(
                "evaluate_111_at_1_0",
                "Evaluate [1,1,1] at (1,0): Q=1, primitive.",
                {"a": 1, "b": 1, "c": 1, "x": 1, "y": 0},
            ),
        ),
    ),
    _op(
        "number_theory.binary_quadratic_form.reduce",
        "Gauss-reduce a binary quadratic form",
        "Reduce a primitive positive-definite binary quadratic form to its "
        "canonical Gauss-reduced representative under SL_2(Z), returning the "
        "reduced form, the unimodular transformation witness, and the reduction "
        "step ledger.",
        BinaryQuadraticFormReduceRequest,
        ReducedBinaryQuadraticFormResult,
        compute_reduce,
        "number-theory",
        "exact",
        examples=(
            example(
                "reduce_5_3_1",
                "Reduce [5,3,1]: discriminant -11.",
                {"a": 5, "b": 3, "c": 1},
            ),
        ),
    ),
    _op(
        "number_theory.binary_quadratic_form.proper_equivalence.decide",
        "Decide proper equivalence of two binary quadratic forms",
        "Decide whether two primitive positive-definite binary quadratic forms "
        "are properly equivalent (SL_2(Z)) by comparing their canonical reduced "
        "representatives.",
        BinaryQuadraticFormProperEquivRequest,
        ProperEquivalenceResult,
        compute_proper_equivalence,
        "number-theory",
        "exact",
        examples=(
            example(
                "equiv_1_1_1_and_1_1_1",
                "Decide if [1,1,1] and [1,1,1] are properly equivalent.",
                {
                    "form1": [1, 1, 1],
                    "form2": [1, 1, 1],
                },
            ),
        ),
    ),
    _op(
        "number_theory.binary_quadratic_form.reduced_classes.compute",
        "Enumerate reduced classes of a discriminant",
        "Enumerate all reduced primitive positive-definite binary quadratic forms "
        "of a given negative discriminant D, returning the complete class set and "
        "class number h(D).",
        BinaryQuadraticFormReducedClassesRequest,
        ReducedClassesResult,
        compute_reduced_classes,
        "number-theory",
        "exact",
        examples=(
            example(
                "classes_disc_neg_3",
                "Enumerate reduced classes of D=-3.",
                {"discriminant": -3},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
