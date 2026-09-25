"""Exact projection onto one word-degree component."""

from jacobian._exact import canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._models import FreeAlgebraPolynomial
from jacobian.math.free_algebras.homogeneous_component._models import (
    MAX_HOMOGENEOUS_COMPONENT_OUTPUT_CELLS,
    MAX_HOMOGENEOUS_COMPONENT_WORK,
    FreeAlgebraHomogeneousComponent,
    FreeAlgebraHomogeneousComponentRequest,
)


def _reject_resource(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=location,
        code=f"free_algebra.homogeneous_component.{code}",
        message=message,
    )


def homogeneous_component(
    request: FreeAlgebraHomogeneousComponentRequest,
) -> FreeAlgebraHomogeneousComponent:
    """Return the exact sum of terms whose words have the requested length.

    Canonical polynomial order is already degree-first, so selecting a degree
    preserves the canonical subsequence and requires no term sorting or
    coefficient arithmetic.
    """

    try:
        admitted = FreeAlgebraHomogeneousComponentRequest.model_validate(
            request.model_dump()
        )
        polynomial = FreeAlgebraPolynomial.model_validate(
            admitted.polynomial.model_dump()
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="free_algebra.homogeneous_component.shape",
            message="the homogeneous-component request is not canonical",
        ) from exc

    term_count = len(polynomial.terms)
    selected_count = 0
    selected_word_count = 0
    for term in polynomial.terms:
        if len(term.word) == admitted.degree:
            selected_count += 1
            selected_word_count += len(term.word)
    work = 2 * term_count + selected_word_count
    if work > MAX_HOMOGENEOUS_COMPONENT_WORK:
        _reject_resource(
            ("polynomial", "terms"),
            "work_bound",
            "homogeneous-component scan exceeds the admitted work bound",
        )

    output_cells = 64 + sum(12 * len(letter) + 2 for letter in polynomial.alphabet)
    for term in polynomial.terms:
        if len(term.word) == admitted.degree:
            output_cells += (
                32
                + sum(12 * len(letter) + 2 for letter in term.word)
                + 2 * canonical_rational_component_digits(term.coefficient)
            )
    if output_cells > MAX_HOMOGENEOUS_COMPONENT_OUTPUT_CELLS:
        _reject_resource(
            ("polynomial", "terms"),
            "output_bound",
            "homogeneous component exceeds the admitted output allocation",
        )

    terms = tuple(
        term for term in polynomial.terms if len(term.word) == admitted.degree
    )
    component_polynomial = FreeAlgebraPolynomial.model_construct(
        alphabet=polynomial.alphabet,
        terms=terms,
    )
    return FreeAlgebraHomogeneousComponent.model_construct(
        degree=admitted.degree,
        polynomial=component_polynomial,
    )


__all__ = ["homogeneous_component"]
