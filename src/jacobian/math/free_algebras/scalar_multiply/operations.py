"""Admitted exact scalar action on sparse noncommutative polynomials."""

from jacobian._exact import (
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
)
from jacobian.math.free_algebras.operations import _admit_polynomial
from jacobian.math.free_algebras.scalar_multiply._models import (
    MAX_SCALAR_MULTIPLY_COEFFICIENT_DIGITS,
    MAX_SCALAR_MULTIPLY_INTERMEDIATE_CELLS,
    MAX_SCALAR_MULTIPLY_OUTPUT_CELLS,
    MAX_SCALAR_MULTIPLY_SCALAR_DIGITS,
    MAX_SCALAR_MULTIPLY_WORK,
)


def _reject_resource(
    code: str, message: str, *, location: tuple[str, ...] = ("polynomial",)
) -> None:
    raise OperationResourceAdmissionError(
        location=location,
        code=f"free_algebra.{code}",
        message=message,
    )


def _admit_scalar(value: CanonicalRational) -> CanonicalRational:
    try:
        return CanonicalRational.model_validate(value.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("scalar",),
            code="free_algebra.scalar_shape",
            message="scalar must be a canonical exact rational",
        ) from exc


def scalar_multiply(
    polynomial: FreeAlgebraPolynomial,
    scalar: CanonicalRational,
) -> FreeAlgebraPolynomial:
    """Multiply every coefficient by an exact rational, preserving word support."""

    source = _admit_polynomial(polynomial, label="polynomial")
    scalar_value = _admit_scalar(scalar)
    scalar_fraction = scalar_value.as_fraction()
    if not source.terms or not scalar_fraction:
        return FreeAlgebraPolynomial(alphabet=source.alphabet, terms=())

    scalar_digits = canonical_rational_component_digits(scalar_value)
    if scalar_digits > MAX_SCALAR_MULTIPLY_SCALAR_DIGITS:
        _reject_resource(
            "scalar_multiply_coefficient_growth",
            "a nonzero polynomial coefficient cannot cancel enough digits from "
            f"a {scalar_digits}-digit scalar to fit the "
            f"{MAX_SCALAR_MULTIPLY_COEFFICIENT_DIGITS}-digit result bound",
            location=("scalar",),
        )

    source_digits = max(
        canonical_rational_component_digits(term.coefficient) for term in source.terms
    )
    term_count = len(source.terms)
    maximum_word_length = max((len(term.word) for term in source.terms), default=0)
    work_bound = term_count * (scalar_digits + source_digits) ** 2
    work_bound += (
        2 * term_count * max(1, term_count.bit_length()) * max(1, maximum_word_length)
    )
    if work_bound > MAX_SCALAR_MULTIPLY_WORK:
        _reject_resource(
            "scalar_multiply_work_budget",
            "exact coefficient products and canonical support checks exceed the "
            f"{MAX_SCALAR_MULTIPLY_WORK}-unit work bound",
        )

    output_cells = len(source.alphabet) + 64
    output_cells += sum(
        64
        + sum(len(letter) for letter in term.word)
        + 2 * (MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS + 1)
        for term in source.terms
    )
    if output_cells > MAX_SCALAR_MULTIPLY_OUTPUT_CELLS:
        _reject_resource(
            "scalar_multiply_output_cells",
            "the canonical scaled polynomial exceeds the admitted "
            f"{MAX_SCALAR_MULTIPLY_OUTPUT_CELLS}-cell output bound",
        )

    intermediate_cells = len(source.alphabet) + 64
    intermediate_cells += sum(
        64
        + sum(len(letter) for letter in term.word)
        + 2 * (scalar_digits + source_digits + 1)
        for term in source.terms
    )
    if intermediate_cells > MAX_SCALAR_MULTIPLY_INTERMEDIATE_CELLS:
        _reject_resource(
            "scalar_multiply_intermediate_cells",
            "exact coefficient products exceed the admitted "
            f"{MAX_SCALAR_MULTIPLY_INTERMEDIATE_CELLS}-cell intermediate bound",
        )

    coefficients = tuple(
        CanonicalRational.from_fraction(
            scalar_fraction * term.coefficient.as_fraction()
        )
        for term in source.terms
    )
    if any(
        canonical_rational_component_digits(coefficient)
        > MAX_SCALAR_MULTIPLY_COEFFICIENT_DIGITS
        for coefficient in coefficients
    ):
        _reject_resource(
            "scalar_multiply_coefficient_growth",
            "an exact scaled coefficient exceeds the "
            f"{MAX_SCALAR_MULTIPLY_COEFFICIENT_DIGITS}-digit result bound",
        )

    terms = tuple(
        FreeAlgebraTerm(coefficient=coefficient, word=term.word)
        for term, coefficient in zip(source.terms, coefficients, strict=True)
    )
    return FreeAlgebraPolynomial(alphabet=source.alphabet, terms=terms)
