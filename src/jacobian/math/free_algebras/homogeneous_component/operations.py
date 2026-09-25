"""Exact projection onto one word-degree component."""

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._models import FreeAlgebraPolynomial
from jacobian.math.free_algebras.homogeneous_component._models import (
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
    # Match the request and result fields' exact-integer wire envelope so a
    # native call and a wire call classify an oversized degree the same way.
    # The bit-length shortcut avoids materializing 10**32768 for ordinary
    # degrees; only a value beyond it is measured exactly.
    if degree.bit_length() > 3 * MAX_CANONICAL_INTEGER_DIGITS and (
        decimal_digit_width(degree) > MAX_CANONICAL_INTEGER_DIGITS
    ):
        raise OperationDomainValidationError(
            location=("degree",),
            code="free_algebra.homogeneous_component.degree",
            message=(
                "degree must have at most "
                f"{MAX_CANONICAL_INTEGER_DIGITS} decimal digits"
            ),
        )
    if not isinstance(polynomial, FreeAlgebraPolynomial):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="free_algebra.homogeneous_component.shape",
            message="the free-algebra polynomial is not canonical",
        )
    try:
        if len(polynomial.terms) > MAX_HOMOGENEOUS_COMPONENT_WORK // 2:
            _reject_resource(
                ("polynomial", "terms"),
                "work_bound",
                "polynomial exceeds homogeneous-component scan bound",
            )
        polynomial = FreeAlgebraPolynomial.model_validate(polynomial.model_dump())
    except OperationResourceAdmissionError:
        raise
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="free_algebra.homogeneous_component.shape",
            message="the free-algebra polynomial is not canonical",
        ) from exc

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
