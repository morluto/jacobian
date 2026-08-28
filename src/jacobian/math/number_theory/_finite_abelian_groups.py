"""Thin catalog bindings for finite-Abelian product-group operations."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.math.groups.finite_abelian import (
    FiniteAbelianGroupFactorizationRequest,
    FiniteAbelianGroupFactorizationResult,
    FiniteAbelianSpectralPairRequest,
    FiniteAbelianSpectralPairResult,
    _run_finite_abelian_group_factorization,
    _run_finite_abelian_spectral_pair,
)
from jacobian.math.number_theory._support import number_theory_operation

FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION = number_theory_operation(
    "finite_abelian_group.exact_factorization.compute",
    "Exact finite abelian group factorization",
    (
        "Normalize two bounded integer-vector factors in a declared product "
        "of cyclic groups, exhaustively count every sum representation, and "
        "decide whether every group element has exactly one representation."
    ),
    FiniteAbelianGroupFactorizationRequest,
    FiniteAbelianGroupFactorizationResult,
    _run_finite_abelian_group_factorization,
    "number-theory",
    "finite-abelian-group",
    "cyclic-product",
    "factorization",
    "unique-representation",
    "coset-transversal",
    "exact",
    examples=(
        example(
            "z2_times_z4_transversal",
            "Verify eight representatives form a complete transversal.",
            {
                "moduli": [2, 4],
                "left": [
                    [0, 0],
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [1, 0],
                    [1, 1],
                    [1, 2],
                    [1, 3],
                ],
                "right": [[0, 0]],
            },
        ),
    ),
)

FINITE_ABELIAN_SPECTRAL_PAIR_OPERATION = number_theory_operation(
    "finite_abelian_group.spectral_pair.decide",
    "Decide an exact finite-Abelian spectral pair",
    (
        "Decide whether a canonical residue-tuple frequency set is a spectrum "
        "of a point set in an explicit product of cyclic groups, including "
        "empty and singleton degenerate cases. Uses "
        "the fixed positive dual pairing chi_lambda(a) = exp(2*pi*i*sum_j "
        "lambda_j*a_j/m_j), proves every required character sum zero by exact "
        "integer-polynomial reduction modulo Phi_lcm(m_j), and returns the "
        "first nonzero exact remainder on failure."
    ),
    FiniteAbelianSpectralPairRequest,
    FiniteAbelianSpectralPairResult,
    _run_finite_abelian_spectral_pair,
    "harmonic-analysis",
    "finite-abelian-group",
    "spectral-pair",
    "fourier",
    "character-orthogonality",
    "cyclotomic-polynomial",
    "exact",
    examples=(
        example(
            "z4_even_pair",
            "Decide the two-point spectral pair A={0,2}, Lambda={0,1} in Z/4.",
            {
                "source": {
                    "group": {"moduli": [4]},
                    "points": [[0], [2]],
                    "frequencies": [[0], [1]],
                }
            },
        ),
    ),
)

__all__ = [
    "FINITE_ABELIAN_GROUP_FACTORIZATION_OPERATION",
    "FINITE_ABELIAN_SPECTRAL_PAIR_OPERATION",
]
