"""Modular-owned exact number-theory operations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory._discrete_logarithm import (
    DISCRETE_LOGARITHM_OPERATION,
)
from jacobian.math.number_theory._modular_basic_models import (
    ChineseRemainderRequest,
    ChineseRemainderResult,
    JacobiSymbolRequest,
    JacobiSymbolResult,
    ModularUnitRequest,
    ModulusRequest,
    QuadraticResiduesResult,
)
from jacobian.math.number_theory._modular_models import (
    ModularPolynomialResidueImageRequest,
    ModularPolynomialResidueImageResult,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue
from jacobian.math.number_theory.operations import (
    chinese_remainder,
    jacobi_symbol,
    modular_inverse,
    modular_polynomial_residue_assignments,
    modular_polynomial_residue_image,
    multiplicative_order,
    quadratic_residues,
)


def _run_jacobi_symbol(request: JacobiSymbolRequest) -> JacobiSymbolResult:
    return jacobi_symbol(request.a, request.n)


def _run_modular_inverse(request: ModularUnitRequest) -> IntegerValue:
    return modular_inverse(request.value, request.modulus)


def _run_multiplicative_order(request: ModularUnitRequest) -> IntegerValue:
    return multiplicative_order(request.value, request.modulus)


def _run_quadratic_residues(request: ModulusRequest) -> QuadraticResiduesResult:
    return quadratic_residues(request.modulus)


def _run_modular_polynomial_residue_image(
    request: ModularPolynomialResidueImageRequest,
) -> ModularPolynomialResidueImageResult:
    return modular_polynomial_residue_image(
        request.modulus, request.variables, request.terms
    )


def _run_modular_polynomial_residue_assignments(
    request: ModularPolynomialResidueImageRequest,
) -> ModularPolynomialResidueImageResult:
    return modular_polynomial_residue_assignments(
        request.modulus, request.variables, request.terms
    )


def _run_chinese_remainder(request: ChineseRemainderRequest) -> ChineseRemainderResult:
    return chinese_remainder(request.residues, request.moduli)


MODULAR_OPERATIONS = (
    MathTool(
        operation_id="number_theory.compute.jacobi_symbol",
        title="Compute Jacobi symbol",
        description="Compute the Jacobi symbol (a / n) for an odd positive denominator.",
        request_type=JacobiSymbolRequest,
        result_type=JacobiSymbolResult,
        run=_run_jacobi_symbol,
        tags=("number-theory", "modular", "jacobi-symbol"),
        examples=(
            OperationExample(
                name="jacobi_10_21",
                description="Compute the Jacobi symbol (10/21).",
                input={"a": "10", "n": 21},
            ),
            OperationExample(
                name="jacobi_7_15",
                description="Compute the Jacobi symbol (7/15); the denominator n must be odd.",
                input={"a": "7", "n": 15},
            ),
        ),
    ),
    MathTool(
        operation_id="modular.compute.inverse",
        title="Compute modular inverse",
        description="Compute the least nonnegative inverse of a unit modulo m.",
        request_type=ModularUnitRequest,
        result_type=IntegerValue,
        run=_run_modular_inverse,
        tags=("number-theory", "modular"),
        examples=(
            OperationExample(
                name="inverse_3_mod_11",
                description="Compute the inverse of 3 modulo 11.",
                input={"value": "3", "modulus": 11},
            ),
        ),
    ),
    MathTool(
        operation_id="modular.compute.multiplicative_order",
        title="Compute multiplicative order",
        description="Compute the multiplicative order of a unit modulo m.",
        request_type=ModularUnitRequest,
        result_type=IntegerValue,
        run=_run_multiplicative_order,
        tags=("number-theory", "modular"),
        examples=(
            OperationExample(
                name="multiplicative_order_2_mod_7",
                description="Compute the multiplicative order of 2 modulo 7.",
                input={"value": "2", "modulus": 7},
            ),
        ),
    ),
    MathTool(
        operation_id="modular.enumerate.quadratic_residues",
        title="Enumerate quadratic residues",
        description="Enumerate all quadratic residues modulo m.",
        request_type=ModulusRequest,
        result_type=QuadraticResiduesResult,
        run=_run_quadratic_residues,
        tags=("number-theory", "modular", "enumeration"),
        examples=(
            OperationExample(
                name="quadratic_residues_mod_10",
                description="Enumerate quadratic residues modulo 10.",
                input={"modulus": 10},
            ),
        ),
    ),
    MathTool(
        operation_id="modular.polynomial_residue_image.compute",
        title="Compute modular polynomial residue image",
        description=(
            "Compute the bounded image of a sparse integer polynomial over declared "
            "finite residue domains modulo m, including multiplicities and first "
            "witness assignments."
        ),
        request_type=ModularPolynomialResidueImageRequest,
        result_type=ModularPolynomialResidueImageResult,
        run=_run_modular_polynomial_residue_image,
        tags=(
            "number-theory",
            "modular",
            "polynomial",
            "residue",
            "enumeration",
            "obstruction",
        ),
        examples=(
            OperationExample(
                name="cubic_residue_image_mod_7",
                description="Enumerate four times x cubed modulo 7; variable names and exponent vectors must be unique and ordered, with canonical residues.",
                input={
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
    MathTool(
        operation_id="modular.polynomial_residue_image.assignments.compute",
        title="Compute modular polynomial assignments",
        description=(
            "Compute the complete bounded assignment-to-residue table for a sparse "
            "modular polynomial, including the image summary."
        ),
        request_type=ModularPolynomialResidueImageRequest,
        result_type=ModularPolynomialResidueImageResult,
        run=_run_modular_polynomial_residue_assignments,
        tags=("number-theory", "modular", "polynomial", "residue", "assignments"),
        examples=(
            OperationExample(
                name="cubic_assignment_ledger_mod_7",
                description="Compute the assignment ledger for four times x cubed modulo 7; names and exponent vectors must be unique and ordered.",
                input={
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
    MathTool(
        operation_id="modular.solve.chinese_remainder",
        title="Solve congruence system",
        description="Solve a finite compatible system of integer congruences. Admission "
        "bounds the system's combined modulus (its LCM) to the 256-digit "
        "exact result width rather than each modulus alone.",
        request_type=ChineseRemainderRequest,
        result_type=ChineseRemainderResult,
        run=_run_chinese_remainder,
        tags=("number-theory", "modular"),
        examples=(
            OperationExample(
                name="crt_2_mod_3_3_mod_5",
                description="Solve x=2 mod 3 and x=3 mod 5.",
                input={"residues": [2, 3], "moduli": [3, 5]},
            ),
            OperationExample(
                name="crt_three_congruences",
                description="Solve three congruences; residues and moduli must have equal lengths and each residue must be canonical.",
                input={"residues": [1, 4, 0], "moduli": [2, 5, 7]},
            ),
        ),
    ),
    DISCRETE_LOGARITHM_OPERATION,
)
