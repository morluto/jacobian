"""Small exact kernels shared by modular quadratic operations and values."""

from jacobian.math.number_theory.quadratic_forms.integral.modular._models import (
    ModularQuadraticPolynomial,
)


def evaluate_modular_polynomial_value(
    polynomial: ModularQuadraticPolynomial, coordinates: tuple[int, ...]
) -> int:
    """Evaluate an admitted polynomial and coordinate tuple in its residue ring."""

    modulus = polynomial.modulus
    value = 0
    for coefficient, coordinate in zip(
        polynomial.diagonal_residues, coordinates, strict=True
    ):
        if coefficient:
            value = (
                value + coefficient * (coordinate * coordinate % modulus)
            ) % modulus
    for term in polynomial.cross_terms:
        product = (coordinates[term.left] * coordinates[term.right]) % modulus
        value = (value + term.coefficient * product) % modulus
    return value


__all__ = ["evaluate_modular_polynomial_value"]
