"""Exact projection onto one word-degree component."""

from jacobian._exact import canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH,
    FreeAlgebraPolynomial,
)
from jacobian.math.free_algebras.homogeneous_component._models import (
    MAX_HOMOGENEOUS_COMPONENT_OUTPUT_CELLS,
    MAX_HOMOGENEOUS_COMPONENT_WORK,
    FreeAlgebraHomogeneousComponent,
)


def _reject_resource(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=location,
        code=f"free_algebra.homogeneous_component.{code}",
        message=message,
    )


def homogeneous_component(
    polynomial: FreeAlgebraPolynomial,
    degree: int,
) -> FreeAlgebraHomogeneousComponent:
    """Return the exact sum of terms whose words have the requested length.

    Canonical polynomial order is already degree-first, so selecting a degree
    preserves the canonical subsequence and requires no term sorting or
    coefficient arithmetic.
    """

    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
        raise OperationDomainValidationError(
            location=("degree",),
            code="free_algebra.homogeneous_component.degree",
            message="degree must be a nonnegative integer",
        )
    if degree > MAX_FREE_ALGEBRA_RESULT_WORD_LENGTH:
        _reject_resource(
            ("degree",),
            "degree_bound",
            "homogeneous-component degree exceeds the 64-letter value envelope",
        )

    term_count = len(polynomial.terms)
    selected_word_count = 0
    for term in polynomial.terms:
        if len(term.word) == degree:
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
        if len(term.word) == degree:
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

    terms = tuple(term for term in polynomial.terms if len(term.word) == degree)
    component_polynomial = FreeAlgebraPolynomial.model_construct(
        alphabet=polynomial.alphabet,
        terms=terms,
    )
    return FreeAlgebraHomogeneousComponent.model_construct(
        degree=degree,
        polynomial=component_polynomial,
    )


__all__ = ["homogeneous_component"]
