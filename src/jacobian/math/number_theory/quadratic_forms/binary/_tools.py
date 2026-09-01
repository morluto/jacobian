"""Integral binary quadratic form operation declarations."""

from collections.abc import Callable
from math import gcd
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.number_theory.quadratic_forms.binary import operations as native
from jacobian.math.number_theory.quadratic_forms.binary._models import (
    BinaryQuadraticFormCheckRequest,
    BinaryQuadraticFormCheckResult,
    BinaryQuadraticFormClassComposeRequest,
    BinaryQuadraticFormClassCompositionResult,
    BinaryQuadraticFormEvaluateRequest,
    BinaryQuadraticFormEvaluateResult,
    BinaryQuadraticFormProperEquivRequest,
    BinaryQuadraticFormReducedClassesRequest,
    BinaryQuadraticFormReduceRequest,
    BinaryQuadraticFormRepresentationsRequest,
    BinaryQuadraticFormRepresentationsResult,
    ProperEquivalenceResult,
    ReducedBinaryQuadraticFormResult,
    ReducedClassesResult,
)


def _admit[ResultT](
    operation: Callable[[], ResultT], location: tuple[str, ...]
) -> ResultT:
    try:
        return operation()
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc


def compute_check(
    request: BinaryQuadraticFormCheckRequest,
) -> BinaryQuadraticFormCheckResult:
    return native.check(request.a, request.b, request.c)


def compute_evaluate(
    request: BinaryQuadraticFormEvaluateRequest,
) -> BinaryQuadraticFormEvaluateResult:
    value = _admit(
        lambda: native.evaluate(request.form, request.x, request.y), ("x", "y")
    )
    return BinaryQuadraticFormEvaluateResult._from_kernel(
        form=request.form,
        x=request.x,
        y=request.y,
        value=value,
        primitive=gcd(request.x, request.y) == 1,
    )


def compute_reduce(
    request: BinaryQuadraticFormReduceRequest,
) -> ReducedBinaryQuadraticFormResult:
    reduced, matrix = native.reduction(request.form)
    return ReducedBinaryQuadraticFormResult._from_kernel(
        form=request.form, reduced_form=reduced, matrix=matrix
    )


def compute_proper_equivalence(
    request: BinaryQuadraticFormProperEquivRequest,
) -> ProperEquivalenceResult:
    return native.proper_equivalence(request.first, request.second)


def compute_reduced_classes(
    request: BinaryQuadraticFormReducedClassesRequest,
) -> ReducedClassesResult:
    classes = _admit(
        lambda: native.reduced_classes(request.discriminant), ("discriminant",)
    )
    return ReducedClassesResult._from_kernel(
        discriminant=request.discriminant, classes=classes
    )


def compute_representations(
    request: BinaryQuadraticFormRepresentationsRequest,
) -> BinaryQuadraticFormRepresentationsResult:
    representations = _admit(
        lambda: native.representations(request.form, request.target), ("target",)
    )
    return BinaryQuadraticFormRepresentationsResult._from_kernel(
        form=request.form,
        target=request.target,
        representations=representations,
    )


def compute_class_compose(
    request: BinaryQuadraticFormClassComposeRequest,
) -> BinaryQuadraticFormClassCompositionResult:
    if request.first.discriminant != request.second.discriminant:
        raise OperationDomainValidationError(
            location=("first", "second"),
            code="integral_binary_quadratic_form.class_discriminant_mismatch",
            message="proper form classes must have the same discriminant",
        )
    return _admit(
        lambda: native.compose_classes(request.first, request.second),
        ("first", "second"),
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="number_theory.binary_quadratic_form.class_compose.compute",
        title="Compose proper binary quadratic-form classes",
        description="Compute the exact proper-class product of two canonical Gauss-reduced "
        "primitive positive-definite binary quadratic forms of one discriminant. "
        "Return a direct composed form, its bilinear composition map, and the "
        "canonical reduced product with an SL_2(Z) reduction witness.",
        request_type=BinaryQuadraticFormClassComposeRequest,
        result_type=BinaryQuadraticFormClassCompositionResult,
        run=compute_class_compose,
        tags=("number-theory", "quadratic-forms", "exact"),
        examples=(
            OperationExample(
                name="compose_classes_discriminant_neg_23",
                description="Compose the class [2,-1,3] with itself at discriminant -23; "
                "both inputs must be canonical reduced representatives of the same "
                "discriminant.",
                input={
                    "first": {"representative": {"a": 2, "b": -1, "c": 3}},
                    "second": {"representative": {"a": 2, "b": -1, "c": 3}},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.binary_quadratic_form.check",
        title="Check a binary quadratic form",
        description="Check if integer coefficients (a,b,c) form a primitive positive-definite "
        "binary quadratic form Q(x,y) = a*x^2 + b*x*y + c*y^2 with negative "
        "discriminant D = b^2 - 4ac.",
        request_type=BinaryQuadraticFormCheckRequest,
        result_type=BinaryQuadraticFormCheckResult,
        run=compute_check,
        tags=("number-theory", "exact"),
        examples=(
            OperationExample(
                name="check_1_1_1",
                description="Check [1,1,1]: discriminant -3, primitive positive definite.",
                input={"a": 1, "b": 1, "c": 1},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.binary_quadratic_form.evaluate",
        title="Evaluate a binary quadratic form",
        description="Evaluate Q(x,y) = a*x^2 + b*x*y + c*y^2 at an integer pair (x,y) "
        "and determine primitive status.",
        request_type=BinaryQuadraticFormEvaluateRequest,
        result_type=BinaryQuadraticFormEvaluateResult,
        run=compute_evaluate,
        tags=("number-theory", "exact"),
        examples=(
            OperationExample(
                name="evaluate_111_at_1_0",
                description="Evaluate [1,1,1] at (1,0): Q=1, primitive.",
                input={"form": {"a": 1, "b": 1, "c": 1}, "x": 1, "y": 0},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.binary_quadratic_form.reduce",
        title="Gauss-reduce a binary quadratic form",
        description="Reduce a primitive positive-definite binary quadratic form to its "
        "canonical Gauss-reduced representative under SL_2(Z), returning the "
        "reduced form and the unimodular transformation witness.",
        request_type=BinaryQuadraticFormReduceRequest,
        result_type=ReducedBinaryQuadraticFormResult,
        run=compute_reduce,
        tags=("number-theory", "exact"),
        examples=(
            OperationExample(
                name="reduce_5_3_1",
                description="Reduce [5,3,1]: discriminant -11.",
                input={"form": {"a": 5, "b": 3, "c": 1}},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.binary_quadratic_form.proper_equivalence.decide",
        title="Decide proper equivalence of two binary quadratic forms",
        description="Decide whether two primitive positive-definite binary quadratic forms "
        "are properly equivalent (SL_2(Z)) by comparing their canonical reduced "
        "representatives.",
        request_type=BinaryQuadraticFormProperEquivRequest,
        result_type=ProperEquivalenceResult,
        run=compute_proper_equivalence,
        tags=("number-theory", "exact"),
        examples=(
            OperationExample(
                name="equiv_1_1_1_and_1_1_1",
                description="Decide if [1,1,1] and [1,1,1] are properly equivalent.",
                input={
                    "first": {"a": 1, "b": 1, "c": 1},
                    "second": {"a": 1, "b": 1, "c": 1},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.binary_quadratic_form.representations.compute",
        title="Enumerate complete binary quadratic-form representations",
        description="Return every ordered signed integer pair (x,y) satisfying "
        "Q(x,y)=n for one primitive positive-definite integral binary quadratic "
        "form. The result distinguishes raw and primitive representation counts "
        "and exhausts the exact discriminant-derived y-coordinate range.",
        request_type=BinaryQuadraticFormRepresentationsRequest,
        result_type=BinaryQuadraticFormRepresentationsResult,
        run=compute_representations,
        tags=("number-theory", "quadratic-forms", "exact", "bounded-search"),
        examples=(
            OperationExample(
                name="representations_x_squared_plus_y_squared_of_5",
                description="For D=-4, the admission expression "
                "2*floor_sqrt(4*a*n/(-D))+1 is 5; enumerate the eight "
                "ordered signed representations of 5 by x^2+y^2.",
                input={"form": {"a": 1, "b": 0, "c": 1}, "target": 5},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.binary_quadratic_form.reduced_classes.compute",
        title="Enumerate reduced classes of a discriminant",
        description="Enumerate all reduced primitive positive-definite binary quadratic forms "
        "of a given negative discriminant D, returning the complete class set and "
        "class number h(D).",
        request_type=BinaryQuadraticFormReducedClassesRequest,
        result_type=ReducedClassesResult,
        run=compute_reduced_classes,
        tags=("number-theory", "exact"),
        examples=(
            OperationExample(
                name="classes_disc_neg_3",
                description="For D=-3 the reduced-scan envelope A*(A+2) with "
                "A=floor_sqrt((-D)//3)+1 is 8; enumerate the single reduced "
                "class of discriminant -3.",
                input={"discriminant": -3},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
