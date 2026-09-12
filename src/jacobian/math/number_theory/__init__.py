"""Supported exact number-theory API."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math.number_theory._dickman_rho import (
        DickmanRhoAffineAxis,
        DickmanRhoAffinePiece,
        DickmanRhoPiecewiseEnclosureParameters,
        DickmanRhoPiecewiseEnclosureRequest,
        DickmanRhoPiecewiseEnclosureResult,
        DyadicCoefficientBall,
        dickman_rho_piecewise_enclosure,
    )
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
    from jacobian.math.number_theory._r_full_enumerate_models import (
        RFullEnumerateResult,
    )
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
    "DickmanRhoAffineAxis",
    "DickmanRhoAffinePiece",
    "DickmanRhoPiecewiseEnclosureParameters",
    "DickmanRhoPiecewiseEnclosureRequest",
    "DickmanRhoPiecewiseEnclosureResult",
    "DyadicCoefficientBall",
    "FriableCountResult",
    "PrimeShiftProfileResult",
    "RFullEnumerateResult",
    "binomial_prime_valuation",
    "chinese_remainder",
    "contiguous_sum_profile",
    "count_friable",
    "dickman_rho_piecewise_enclosure",
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


_OWNER_MODULES = {
    "DickmanRhoAffineAxis": "_dickman_rho",
    "DickmanRhoAffinePiece": "_dickman_rho",
    "DickmanRhoPiecewiseEnclosureParameters": "_dickman_rho",
    "DickmanRhoPiecewiseEnclosureRequest": "_dickman_rho",
    "DickmanRhoPiecewiseEnclosureResult": "_dickman_rho",
    "DyadicCoefficientBall": "_dickman_rho",
    "FriableCountResult": "_friable_models",
    "PrimeShiftProfileResult": "_prime_shift_models",
    "RFullEnumerateResult": "_r_full_enumerate_models",
    "count_friable": "_friable_kernel",
    "dickman_rho_piecewise_enclosure": "_dickman_rho",
    "divisibility_poset": "_divisibility_poset",
    "enumerate_friable": "_friable_enumerate",
    "enumerate_r_full": "_r_full_enumerate",
    "ramanujan_sum": "ramanujan_sums",
    "verify_certified_factorization": "_factorization_kernels",
    "verify_pratt_certificate": "_factorization_kernels",
    "verify_primality_certificate": "_factorization_kernels",
}


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{_OWNER_MODULES.get(name, 'operations')}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
