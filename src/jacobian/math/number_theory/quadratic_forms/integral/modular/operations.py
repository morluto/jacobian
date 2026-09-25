"""Coefficient reduction and exact evaluation over finite residue rings."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.integral._models import (
    MAX_INTEGRAL_QUADRATIC_FORM_TERMS,
    IntegralQuadraticCrossTerm,
    IntegralQuadraticForm,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular._models import (
    MAX_MODULAR_QUADRATIC_FORM_AXIS,
    MAX_MODULAR_QUADRATIC_FORM_INTEGER_DIGITS,
    MAX_MODULAR_QUADRATIC_FORM_TERMS,
    MAX_MODULAR_QUADRATIC_RESULT_DIGITS,
    ModularCoordinateVector,
    ModularEvaluationRequest,
    ModularInteger,
    ModularQuadraticCrossTerm,
    ModularQuadraticPolynomial,
    ModularQuadraticReduction,
    ModularReductionRequest,
)

MAX_MODULAR_REDUCTION_WORK = 1_100_000
MAX_MODULAR_EVALUATION_WORK = 2_000_000


def _domain_error(reason: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("form",),
        code=f"quadratic_form.modular.{reason}",
        message=message,
    )


def _resource_error(reason: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("form",),
        code=f"quadratic_form.modular.{reason}",
        message=message,
    )


def _digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


def _check_integral_form(form: IntegralQuadraticForm) -> tuple[int, ...]:
    if not isinstance(form, IntegralQuadraticForm):
        raise _domain_error("form_type", "expected a canonical integral quadratic form")
    if (
        not isinstance(form.axis, tuple)
        or not isinstance(form.diagonal_coefficients, tuple)
        or not isinstance(form.cross_terms, tuple)
    ):
        raise _domain_error("form_shape", "source form containers must be tuples")
    n = len(form.axis)
    if n > MAX_MODULAR_QUADRATIC_FORM_AXIS:
        raise _resource_error(
            "axis_bound", "modular reduction supports at most 128 axes"
        )
    if (
        form.domain != "ZZ"
        or any(
            not isinstance(label, str) or not label or len(label) > 128
            for label in form.axis
        )
        or len(set(form.axis)) != n
        or len(form.diagonal_coefficients) != n
        or n + len(form.cross_terms) > MAX_INTEGRAL_QUADRATIC_FORM_TERMS
    ):
        raise _domain_error(
            "form_shape", "source form must be a canonical ZZ polynomial"
        )
    coefficients: list[int] = []
    for coefficient in form.diagonal_coefficients:
        if (
            not isinstance(coefficient, int)
            or isinstance(coefficient, bool)
            or abs(coefficient) >= 10**MAX_MODULAR_QUADRATIC_FORM_INTEGER_DIGITS
        ):
            raise _domain_error(
                "coefficient_bound", "source coefficients must have at most 256 digits"
            )
        coefficients.append(coefficient)
    positions: list[tuple[int, int]] = []
    for term in form.cross_terms:
        if not isinstance(term, IntegralQuadraticCrossTerm):
            raise _domain_error(
                "cross_term_type", "source cross terms must be canonical"
            )
        if (
            not isinstance(term.left, int)
            or isinstance(term.left, bool)
            or not isinstance(term.right, int)
            or isinstance(term.right, bool)
            or term.left < 0
            or term.left >= term.right
            or term.right >= n
            or not isinstance(term.coefficient, int)
            or isinstance(term.coefficient, bool)
            or term.coefficient == 0
            or abs(term.coefficient) >= 10**MAX_MODULAR_QUADRATIC_FORM_INTEGER_DIGITS
        ):
            raise _domain_error(
                "cross_term_shape", "source cross terms must be canonical"
            )
        positions.append((term.left, term.right))
        coefficients.append(term.coefficient)
    if positions != sorted(set(positions)):
        raise _domain_error("cross_term_order", "source cross terms must be ordered")
    return tuple(coefficients)


def _admit_modulus(modulus: int) -> int:
    if not isinstance(modulus, int) or isinstance(modulus, bool) or modulus < 1:
        raise _domain_error("modulus", "modulus must be a positive integer")
    if modulus.bit_length() > 851:
        raise _resource_error(
            "modulus_digits", "modulus is limited to 256 decimal digits"
        )
    digits = _digits(modulus)
    if digits > MAX_MODULAR_QUADRATIC_FORM_INTEGER_DIGITS:
        raise _resource_error(
            "modulus_digits", "modulus is limited to 256 decimal digits"
        )
    return digits


def reduce_integral_form_modulus(
    request: ModularReductionRequest,
) -> ModularQuadraticReduction:
    """Reduce polynomial coefficients canonically into the ring ``Z/mZ``."""

    if not isinstance(request, ModularReductionRequest):
        raise _domain_error(
            "request_type", "expected a typed modular reduction request"
        )
    return _reduce_integral_form_modulus_value(request.form, request.modulus)


def _reduce_integral_form_modulus_value(
    form: IntegralQuadraticForm, modulus: int
) -> ModularQuadraticPolynomial:
    """Reduce canonical values after the owning caller has parsed its request."""

    coefficients = _check_integral_form(form)
    modulus_digits = _admit_modulus(modulus)
    support = len(coefficients)
    work = sum(_digits(coefficient) + modulus_digits for coefficient in coefficients)
    if work > MAX_MODULAR_REDUCTION_WORK:
        raise _resource_error(
            "work_bound", "coefficient reduction exceeds its digit-work bound"
        )
    label_size = sum(len(label) for label in form.axis)
    # Bound the complete returned value, including retained source coefficients
    # and the axis/support structures duplicated in source and target.
    source_coefficient_digits = sum(
        _digits(coefficient) for coefficient in coefficients
    )
    result_digits = (
        modulus_digits
        + 2 * label_size
        + source_coefficient_digits
        + support * modulus_digits
        + 16 * support
    )
    if result_digits > MAX_MODULAR_QUADRATIC_RESULT_DIGITS:
        raise _resource_error(
            "result_bound", "modular polynomial exceeds its exact-result bound"
        )

    diagonal = tuple(value % modulus for value in form.diagonal_coefficients)
    cross_terms = tuple(
        ModularQuadraticCrossTerm(
            left=term.left,
            right=term.right,
            coefficient=residue,
        )
        for term in form.cross_terms
        if (residue := term.coefficient % modulus) != 0
    )
    target = ModularQuadraticPolynomial(
        modulus=modulus,
        axis=form.axis,
        diagonal_residues=diagonal,
        cross_terms=cross_terms,
    )
    return ModularQuadraticReduction(source=request.form, target=target)


def _check_modular_values(
    polynomial: ModularQuadraticPolynomial,
    vector: ModularCoordinateVector,
) -> tuple[int, int]:
    if not isinstance(polynomial, ModularQuadraticPolynomial):
        raise _domain_error(
            "polynomial_type", "expected a canonical modular quadratic polynomial"
        )
    if not isinstance(vector, ModularCoordinateVector):
        raise _domain_error(
            "vector_type", "expected a canonical modular coordinate vector"
        )
    modulus_digits = _admit_modulus(polynomial.modulus)
    if vector.modulus != polynomial.modulus:
        raise _domain_error(
            "parent_mismatch", "polynomial and vector moduli must agree"
        )
    n = len(polynomial.axis)
    if (
        n > MAX_MODULAR_QUADRATIC_FORM_AXIS
        or len(polynomial.diagonal_residues) != n
        or vector.axis != polynomial.axis
        or len(vector.coordinates) != n
        or len(set(polynomial.axis)) != n
        or any(
            not 0 <= value < polynomial.modulus
            for value in polynomial.diagonal_residues
        )
        or any(not 0 <= value < polynomial.modulus for value in vector.coordinates)
    ):
        raise _domain_error(
            "target_shape", "polynomial and vector must use one canonical target space"
        )
    if not isinstance(polynomial.cross_terms, tuple):
        raise _domain_error("polynomial_shape", "mixed terms must be a tuple")
    if any(
        not isinstance(term, ModularQuadraticCrossTerm)
        for term in polynomial.cross_terms
    ):
        raise _domain_error("polynomial_shape", "mixed terms must be canonical")
    positions = tuple((term.left, term.right) for term in polynomial.cross_terms)
    if (
        len(polynomial.cross_terms) + n > MAX_MODULAR_QUADRATIC_FORM_TERMS
        or positions != tuple(sorted(set(positions)))
        or any(
            not isinstance(term.left, int)
            or isinstance(term.left, bool)
            or not isinstance(term.right, int)
            or isinstance(term.right, bool)
            or not 0 <= term.left < term.right < n
            or not isinstance(term.coefficient, int)
            or isinstance(term.coefficient, bool)
            or not 0 < term.coefficient < polynomial.modulus
            for term in polynomial.cross_terms
        )
    ):
        raise _domain_error(
            "polynomial_shape", "modular polynomial support must be canonical"
        )
    return n, modulus_digits


def evaluate_modular_form(request: ModularEvaluationRequest) -> ModularInteger:
    """Evaluate a residue polynomial on a vector with the same parent and axis."""

    if not isinstance(request, ModularEvaluationRequest):
        raise _domain_error(
            "request_type", "expected a typed modular evaluation request"
        )
    polynomial, vector = request.polynomial, request.vector
    _, modulus_digits = _check_modular_values(polynomial, vector)
    term_count = sum(value != 0 for value in polynomial.diagonal_residues) + len(
        polynomial.cross_terms
    )
    if term_count * modulus_digits > MAX_MODULAR_EVALUATION_WORK:
        raise _resource_error(
            "evaluation_work_bound", "modular evaluation exceeds its work bound"
        )

    modulus = polynomial.modulus
    value = 0
    for coefficient, coordinate in zip(
        polynomial.diagonal_residues, vector.coordinates, strict=True
    ):
        if coefficient:
            value = (
                value + coefficient * (coordinate * coordinate % modulus)
            ) % modulus
    for term in polynomial.cross_terms:
        product = (
            vector.coordinates[term.left] * vector.coordinates[term.right]
        ) % modulus
        value = (value + term.coefficient * product) % modulus
    return ModularInteger(modulus=modulus, residue=value)


__all__ = ["evaluate_modular_form", "reduce_integral_form_modulus"]
