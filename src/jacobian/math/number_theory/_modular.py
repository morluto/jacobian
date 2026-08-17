"""Modular-owned exact number-theory operations."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._discrete_logarithm import (
    DISCRETE_LOGARITHM_OPERATION,
)
from jacobian.math.number_theory._models import (
    ChineseRemainderRequest,
    ChineseRemainderResult,
    IntegerValueResult,
    JacobiSymbolRequest,
    JacobiSymbolResult,
    ModularPolynomialResidueImageRequest,
    ModularPolynomialResidueImageResult,
    ModularValueRequest,
    ModulusRequest,
    QuadraticResiduesResult,
)
from jacobian.math.number_theory._modular_operations import (
    compute_jacobi_symbol,
    compute_modular_inverse,
    compute_modular_polynomial_residue_assignments,
    compute_modular_polynomial_residue_image,
    compute_multiplicative_order,
    enumerate_quadratic_residues,
    solve_chinese_remainder,
)
from jacobian.math.number_theory._support import (
    number_theory_operation,
)

MODULAR_OPERATIONS = (
    number_theory_operation(
        "number_theory.compute.jacobi_symbol",
        "Compute Jacobi symbol",
        "Compute the Jacobi symbol (a / n) for an odd positive denominator.",
        JacobiSymbolRequest,
        JacobiSymbolResult,
        compute_jacobi_symbol,
        "number-theory",
        "modular",
        "jacobi-symbol",
        examples=(
            example(
                "jacobi_10_21",
                "Compute the Jacobi symbol (10/21).",
                {"a": "10", "n": 21},
            ),
            example(
                "jacobi_7_15",
                "Compute the Jacobi symbol (7/15); the denominator n must be odd.",
                {"a": "7", "n": 15},
            ),
        ),
    ),
    number_theory_operation(
        "modular.compute.inverse",
        "Compute modular inverse",
        "Compute the least nonnegative inverse of a value modulo m.",
        ModularValueRequest,
        IntegerValueResult,
        compute_modular_inverse,
        "number-theory",
        "modular",
        examples=(
            example(
                "inverse_3_mod_11",
                "Compute the inverse of 3 modulo 11.",
                {"value": "3", "modulus": 11},
            ),
        ),
    ),
    number_theory_operation(
        "modular.compute.multiplicative_order",
        "Compute multiplicative order",
        "Compute the multiplicative order of a unit modulo m.",
        ModularValueRequest,
        IntegerValueResult,
        compute_multiplicative_order,
        "number-theory",
        "modular",
        examples=(
            example(
                "multiplicative_order_2_mod_7",
                "Compute the multiplicative order of 2 modulo 7.",
                {"value": "2", "modulus": 7},
            ),
        ),
    ),
    number_theory_operation(
        "modular.enumerate.quadratic_residues",
        "Enumerate quadratic residues",
        "Enumerate all quadratic residues modulo m.",
        ModulusRequest,
        QuadraticResiduesResult,
        enumerate_quadratic_residues,
        "number-theory",
        "modular",
        "enumeration",
        examples=(
            example(
                "quadratic_residues_mod_10",
                "Enumerate quadratic residues modulo 10.",
                {"modulus": 10},
            ),
        ),
    ),
    number_theory_operation(
        "modular.polynomial_residue_image.compute",
        "Compute modular polynomial residue image",
        (
            "Compute the bounded image of a sparse integer polynomial over declared "
            "finite residue domains modulo m, including multiplicities and first "
            "witness assignments."
        ),
        ModularPolynomialResidueImageRequest,
        ModularPolynomialResidueImageResult,
        compute_modular_polynomial_residue_image,
        "number-theory",
        "modular",
        "polynomial",
        "residue",
        "enumeration",
        "obstruction",
        examples=(
            example(
                "cubic_residue_image_mod_7",
                "Enumerate four times x cubed modulo 7; variable names and exponent vectors must be unique and ordered, with canonical residues.",
                {
                    "modulus": 7,
                    "variables": [
                        {
                            "name": "x",
                            "residues": [0, 1, 2, 3, 4, 5, 6],
                        }
                    ],
                    "terms": [{"coefficient": "4", "exponents": [3]}],
                },
            ),
        ),
    ),
    number_theory_operation(
        "modular.polynomial_residue_image.assignments.compute",
        "Compute modular polynomial assignments",
        (
            "Compute the complete bounded assignment-to-residue table for a sparse "
            "modular polynomial, including the image summary."
        ),
        ModularPolynomialResidueImageRequest,
        ModularPolynomialResidueImageResult,
        compute_modular_polynomial_residue_assignments,
        "number-theory",
        "modular",
        "polynomial",
        "residue",
        "assignments",
        examples=(
            example(
                "cubic_assignment_ledger_mod_7",
                "Compute the assignment ledger for four times x cubed modulo 7; names and exponent vectors must be unique and ordered.",
                {
                    "modulus": 7,
                    "variables": [
                        {
                            "name": "x",
                            "residues": [0, 1, 2, 3, 4, 5, 6],
                        }
                    ],
                    "terms": [{"coefficient": "4", "exponents": [3]}],
                },
            ),
        ),
    ),
    number_theory_operation(
        "modular.solve.chinese_remainder",
        "Solve congruence system",
        "Solve a finite compatible system of integer congruences.",
        ChineseRemainderRequest,
        ChineseRemainderResult,
        solve_chinese_remainder,
        "number-theory",
        "modular",
        examples=(
            example(
                "crt_2_mod_3_3_mod_5",
                "Solve x=2 mod 3 and x=3 mod 5.",
                {"residues": [2, 3], "moduli": [3, 5]},
            ),
            example(
                "crt_three_congruences",
                "Solve three congruences; residues and moduli must have equal lengths and each residue must be canonical.",
                {"residues": [1, 4, 0], "moduli": [2, 5, 7]},
            ),
        ),
    ),
    DISCRETE_LOGARITHM_OPERATION,
)
