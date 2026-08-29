"""Supported exact number-theory API."""

from jacobian.math.number_theory._friable_kernel import count_friable
from jacobian.math.number_theory._friable_models import FriableCountResult
from jacobian.math.number_theory._prime_shift_models import PrimeShiftProfileResult
from jacobian.math.number_theory.operations import (
    chinese_remainder,
    euler_totient,
    factorial_valuation,
    floor_square_root,
    is_prime,
    jacobi_symbol,
    legendre_symbol,
    mobius,
    modular_inverse,
    modular_polynomial_residue_assignments,
    modular_polynomial_residue_image,
    multiplicative_order,
    next_prime,
    nth_prime,
    previous_prime,
    prime_count,
    prime_shift_profile,
    primorial,
    quadratic_residues,
)
from jacobian.math.number_theory.ramanujan_sums import ramanujan_sum

__all__ = [
    "FriableCountResult",
    "PrimeShiftProfileResult",
    "chinese_remainder",
    "count_friable",
    "euler_totient",
    "factorial_valuation",
    "floor_square_root",
    "is_prime",
    "jacobi_symbol",
    "legendre_symbol",
    "mobius",
    "modular_inverse",
    "modular_polynomial_residue_assignments",
    "modular_polynomial_residue_image",
    "multiplicative_order",
    "next_prime",
    "nth_prime",
    "previous_prime",
    "prime_count",
    "prime_shift_profile",
    "primorial",
    "quadratic_residues",
    "ramanujan_sum",
]
