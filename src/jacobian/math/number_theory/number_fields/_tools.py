"""Number field operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.number_theory.number_fields._discriminant_process import (
    compute_nf_discriminant,
)
from jacobian.math.number_theory.number_fields._models import (
    NumberFieldDiscriminantResult,
    NumberFieldEmbeddingsRequest,
    NumberFieldRequest,
)
from jacobian.math.number_theory.number_fields.operations import (
    NumberFieldEmbeddingAdmissionError,
    embeddings,
)
from jacobian.math.number_theory.number_fields.values import (
    NumberFieldEmbeddingProfile,
)


def nf_operation[RequestT: StrictModel, ResultT: StrictModel](
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
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


def _compute_embeddings(
    request: NumberFieldEmbeddingsRequest,
) -> NumberFieldEmbeddingProfile:
    try:
        return embeddings(request.field)
    except NumberFieldEmbeddingAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("field",),
            code=f"number_field.embeddings.{exc.reason}",
            message=str(exc),
        ) from exc


TOOLS: tuple[MathTool[Any, Any], ...] = (
    nf_operation(
        "number_field.discriminant.compute",
        "Compute the discriminant of a number field",
        "Compute the field discriminant of one canonical SimpleNumberFieldPresentation in an isolated SymPy worker, or return UNKNOWN if its bounded execution cannot establish a result.",
        NumberFieldRequest,
        NumberFieldDiscriminantResult,
        compute_nf_discriminant,
        "number-field",
        "discriminant",
        "exact",
        examples=(
            example(
                "quadratic_disc",
                "Discriminant of x^2-2.",
                {
                    "field": {
                        "domain": "QQ",
                        "coefficients_descending": ["1", "0", "-2"],
                    }
                },
            ),
        ),
    ),
    nf_operation(
        "number_field.embeddings.compute",
        "Compute every exact embedding of a simple number field",
        "Return all real and complex embeddings of one bounded primitive irreducible ZZ presentation of QQ(alpha), ordered by exact roots, with certified rational isolation, signature, conjugate-pair grouping, and defining-polynomial discriminant. Degree is at most 8, coefficients have at most 256 digits, and a request-deadline-bound one-shot worker performs recognition, exact signature, conditional elimination, and one all-root isolation pass within the admitted 32,768-bit refinement, 2,097,152-bit resultant-storage, and 10,485,760-byte result envelopes.",
        NumberFieldEmbeddingsRequest,
        NumberFieldEmbeddingProfile,
        _compute_embeddings,
        "number-field",
        "embedding",
        "algebraic-number",
        "exact",
        examples=(
            example(
                "gaussian_field",
                "Both embeddings of QQ(i), grouped as one conjugate pair.",
                {
                    "field": {
                        "domain": "QQ",
                        "coefficients_descending": ["1", "0", "1"],
                    }
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
