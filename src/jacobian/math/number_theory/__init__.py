"""Supported exact number-theory API."""

from jacobian.math.number_theory._divisibility_poset import divisibility_poset
from jacobian.math.number_theory._factorization_kernels import (
    verify_certified_factorization,
    verify_pratt_certificate,
    verify_primality_certificate,
)
from jacobian.math.number_theory._friable_enumerate import enumerate_friable
from jacobian.math.number_theory._friable_kernel import count_friable
from jacobian.math.number_theory._friable_models import FriableCountResult
from jacobian.math.number_theory._prime_shift_models import PrimeShiftProfileResult
from jacobian.math.number_theory._r_full_enumerate import enumerate_r_full
from jacobian.math.number_theory._r_full_enumerate_models import RFullEnumerateResult
from jacobian.math.number_theory.operations import (
    binomial_prime_valuation,
    chinese_remainder,
    contiguous_sum_profile,
    euler_totient,
    factorial_valuation,
    floor_square_root,
    is_prime,
    jacobi_symbol,
    ksigma_preimage,
    legendre_symbol,
    mobius,
    modular_inverse,
    modular_polynomial_residue_assignments,
    modular_polynomial_residue_image,
    multiplicative_order,
    next_prime,
    nth_prime,
    p_adic_interval_profile,
    periodic_congruence_union_measure,
    periodic_congruence_union_profile,
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
    "RFullEnumerateResult",
    "binomial_prime_valuation",
    "chinese_remainder",
    "contiguous_sum_profile",
    "count_friable",
    "divisibility_poset",
    "enumerate_friable",
    "enumerate_r_full",
    "euler_totient",
    "factorial_valuation",
    "floor_square_root",
    "is_prime",
    "jacobi_symbol",
    "ksigma_preimage",
    "legendre_symbol",
    "mobius",
    "modular_inverse",
    "modular_polynomial_residue_assignments",
    "modular_polynomial_residue_image",
    "multiplicative_order",
    "next_prime",
    "nth_prime",
    "p_adic_interval_profile",
    "periodic_congruence_union_measure",
    "periodic_congruence_union_profile",
    "previous_prime",
    "prime_count",
    "prime_shift_profile",
    "primorial",
    "quadratic_residues",
    "ramanujan_sum",
    "verify_certified_factorization",
    "verify_pratt_certificate",
    "verify_primality_certificate",
]
