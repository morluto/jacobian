"""Bounded exhaustive fibers of finite-modulus quadratic polynomials."""

from __future__ import annotations

from itertools import product

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular._kernel import (
    evaluate_modular_polynomial_value,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular._models import (
    ModularCoordinateVector,
    ModularInteger,
    ModularQuadraticPolynomial,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular.fiber._models import (
    MAX_MODULAR_FIBER_OUTPUT_DIGITS,
    MAX_MODULAR_FIBER_OUTPUT_VECTORS,
    MAX_MODULAR_FIBER_WORK,
    ModularQuadraticFiber,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular.operations import (
    _admit_modulus,
    _check_modular_polynomial,
    _digits,
)


def _domain_error(reason: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("polynomial",),
        code=f"quadratic_form.modular.fiber.{reason}",
        message=message,
    )


def _resource_error(reason: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("polynomial",),
        code=f"quadratic_form.modular.fiber.{reason}",
        message=message,
    )


def _state_count(modulus: int, dimension: int) -> int:
    states = 1
    for _ in range(dimension):
        if states > MAX_MODULAR_FIBER_OUTPUT_VECTORS // modulus:
            raise _resource_error(
                "state_bound",
                f"complete fiber domain exceeds {MAX_MODULAR_FIBER_OUTPUT_VECTORS} vectors",
            )
        states *= modulus
    return states


def _preflight_output_digits(
    polynomial: ModularQuadraticPolynomial, states: int, modulus_digits: int
) -> None:
    axis_chars = sum(len(label) for label in polynomial.axis)
    coefficient_digits = sum(_digits(value) for value in polynomial.diagonal_residues)
    coefficient_digits += sum(
        _digits(term.coefficient) for term in polynomial.cross_terms
    )
    support = len(polynomial.diagonal_residues) + len(polynomial.cross_terms)
    retained_polynomial = (
        3 * modulus_digits + axis_chars + coefficient_digits + 16 * support + 32
    )
    # Every possible vector is charged at its largest coordinate representation,
    # including its repeated parent and axis. This is checked before enumeration.
    vector_digits = (
        modulus_digits + axis_chars + len(polynomial.axis) * (modulus_digits + 12) + 32
    )
    total = retained_polynomial + states * vector_digits
    if total > MAX_MODULAR_FIBER_OUTPUT_DIGITS:
        raise _resource_error(
            "output_bound", "complete fiber exceeds its 4000000 digit-value bound"
        )


def compute_modular_quadratic_fiber(
    polynomial: ModularQuadraticPolynomial, target: ModularInteger
) -> ModularQuadraticFiber:
    """Return every coordinate vector mapping to ``target``, in lexicographic order."""

    if not isinstance(polynomial, ModularQuadraticPolynomial):
        raise _domain_error(
            "polynomial_type", "expected a modular quadratic polynomial"
        )
    if not isinstance(target, ModularInteger):
        raise _domain_error("target_type", "expected a modular integer target")
    dimension, modulus_digits = _check_modular_polynomial(polynomial)
    _admit_modulus(target.modulus)
    if isinstance(target.residue, bool) or not isinstance(target.residue, int):
        raise _domain_error("target_residue", "target residue must be an exact integer")
    if target.modulus != polynomial.modulus or not 0 <= target.residue < target.modulus:
        raise _domain_error(
            "parent_mismatch", "polynomial and target residue moduli must agree"
        )

    # This exact empty case avoids enumerating an otherwise huge domain.
    if (
        target.residue != 0
        and not any(polynomial.diagonal_residues)
        and not polynomial.cross_terms
    ):
        return ModularQuadraticFiber(polynomial=polynomial, target=target, vectors=())

    states = _state_count(polynomial.modulus, dimension)
    support = sum(value != 0 for value in polynomial.diagonal_residues) + len(
        polynomial.cross_terms
    )
    # Charge search plus membership checks. For nonzero dimension, the state
    # bound also forces modulus <= 100000, keeping scalar operations small.
    work = 2 * states * (dimension + support + 1)
    if work > MAX_MODULAR_FIBER_WORK:
        raise _resource_error(
            "work_bound", "complete fiber search exceeds its operation bound"
        )
    _preflight_output_digits(polynomial, states, modulus_digits)

    coordinate_rows = (
        product(range(polynomial.modulus), repeat=dimension) if dimension else ((),)
    )
    vectors = tuple(
        ModularCoordinateVector(
            modulus=polynomial.modulus,
            axis=polynomial.axis,
            coordinates=coordinates,
        )
        for coordinates in coordinate_rows
        if evaluate_modular_polynomial_value(polynomial, coordinates) == target.residue
    )
    return ModularQuadraticFiber(
        polynomial=polynomial,
        target=target,
        vectors=vectors,
    )


__all__ = ["compute_modular_quadratic_fiber"]
