"""Complete fibers over small finite residue rings."""

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular import (
    ModularInteger,
    ModularQuadraticFiber,
    ModularQuadraticFiberRequest,
    ModularQuadraticPolynomial,
    compute_modular_quadratic_fiber,
    evaluate_modular_form,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular._models import (
    ModularCoordinateVector,
    ModularEvaluationRequest,
    ModularQuadraticCrossTerm,
)


def _direct_value(x: int, y: int, modulus: int) -> int:
    """Independent scalar expression, retaining the mixed monomial explicitly."""
    return (2 * x * x + 3 * x * y + 5 * y * y) % modulus


def test_mixed_composite_modulus_fibers_match_direct_exhaustive_oracle() -> None:
    polynomial = ModularQuadraticPolynomial(
        modulus=6,
        axis=("x", "y"),
        diagonal_residues=(2, 5),
        cross_terms=({"left": 0, "right": 1, "coefficient": 3},),
    )
    for target_value in range(6):
        target = ModularInteger(modulus=6, residue=target_value)
        result = compute_modular_quadratic_fiber(polynomial, target)
        expected = tuple(
            (x, y)
            for x, y in product(range(6), repeat=2)
            if _direct_value(x, y, 6) == target_value
        )
        assert tuple(vector.coordinates for vector in result.vectors) == expected
        assert result.polynomial == polynomial and result.target == target
        for vector in result.vectors:
            assert (
                evaluate_modular_form(
                    ModularEvaluationRequest(polynomial=polynomial, vector=vector)
                ).residue
                == target_value
            )


def test_empty_fiber_and_zero_ring_degenerate_domains() -> None:
    no_roots = ModularQuadraticPolynomial(
        modulus=4, axis=("x",), diagonal_residues=(1,), cross_terms=()
    )
    empty = compute_modular_quadratic_fiber(
        no_roots, ModularInteger(modulus=4, residue=3)
    )
    assert empty.vectors == ()

    zero_ring = ModularQuadraticPolynomial(
        modulus=1, axis=("x", "y"), diagonal_residues=(0, 0), cross_terms=()
    )
    one_vector = compute_modular_quadratic_fiber(
        zero_ring, ModularInteger(modulus=1, residue=0)
    )
    assert tuple(vector.coordinates for vector in one_vector.vectors) == ((0, 0),)

    zero_dimensional = ModularQuadraticPolynomial(
        modulus=10**100, axis=(), diagonal_residues=(), cross_terms=()
    )
    assert (
        len(
            compute_modular_quadratic_fiber(
                zero_dimensional, ModularInteger(modulus=10**100, residue=0)
            ).vectors
        )
        == 1
    )
    assert (
        compute_modular_quadratic_fiber(
            zero_dimensional, ModularInteger(modulus=10**100, residue=1)
        ).vectors
        == ()
    )


def test_constant_nonzero_target_is_presolved_before_domain_admission() -> None:
    modulus = 10**100
    polynomial = ModularQuadraticPolynomial(
        modulus=modulus,
        axis=tuple(f"x{i}" for i in range(128)),
        diagonal_residues=(0,) * 128,
        cross_terms=(),
    )
    result = compute_modular_quadratic_fiber(
        polynomial, ModularInteger(modulus=modulus, residue=1)
    )
    assert result.vectors == ()


def test_domain_work_is_admitted_before_enumeration() -> None:
    polynomial = ModularQuadraticPolynomial(
        modulus=3,
        axis=tuple(f"x{i}" for i in range(11)),
        diagonal_residues=(0,) * 11,
        cross_terms=(),
    )
    with pytest.raises(OperationResourceAdmissionError, match="100000 vectors"):
        compute_modular_quadratic_fiber(
            polynomial, ModularInteger(modulus=3, residue=0)
        )


def test_worst_case_output_size_is_admitted_before_enumeration() -> None:
    polynomial = ModularQuadraticPolynomial(
        modulus=100_000,
        axis=("x",),
        diagonal_residues=(1,),
        cross_terms=(),
    )
    with pytest.raises(OperationResourceAdmissionError, match="digit-value bound"):
        compute_modular_quadratic_fiber(
            polynomial, ModularInteger(modulus=100_000, residue=0)
        )


def test_digit_work_is_admitted_before_enumeration() -> None:
    axis = tuple(f"x{i}" for i in range(16))
    cross_terms = tuple(
        {"left": left, "right": right, "coefficient": 1}
        for left in range(16)
        for right in range(left + 1, 16)
    )
    polynomial = ModularQuadraticPolynomial(
        modulus=2,
        axis=axis,
        diagonal_residues=(1,) * 16,
        cross_terms=cross_terms,
    )
    with pytest.raises(OperationResourceAdmissionError, match="operation bound"):
        compute_modular_quadratic_fiber(
            polynomial, ModularInteger(modulus=2, residue=0)
        )


def test_request_parent_and_result_json_round_trip() -> None:
    polynomial = ModularQuadraticPolynomial(
        modulus=3,
        axis=("x", "y"),
        diagonal_residues=(1, 1),
        cross_terms=({"left": 0, "right": 1, "coefficient": 1},),
    )
    with pytest.raises(ValidationError, match="moduli must agree"):
        ModularQuadraticFiberRequest(
            polynomial=polynomial, target=ModularInteger(modulus=5, residue=0)
        )
    fiber = compute_modular_quadratic_fiber(
        polynomial, ModularInteger(modulus=3, residue=0)
    )
    data = fiber.model_dump_json()
    decoded = type(fiber).model_validate_json(data)
    assert decoded == fiber
    assert tuple(vector.coordinates for vector in decoded.vectors) == (
        (0, 0),
        (1, 1),
        (2, 2),
    )


def test_fiber_values_preserve_structural_vectors_and_reject_forged_cross_indices() -> (
    None
):
    polynomial = ModularQuadraticPolynomial(
        modulus=5, axis=("x",), diagonal_residues=(1,), cross_terms=()
    )
    bad_vector = ModularCoordinateVector(modulus=5, axis=("x",), coordinates=(1,))
    # Result decoding checks structure only; the membership claim belongs to
    # the operation that computed the complete fiber, not generic validation.
    decoded = ModularQuadraticFiber(
        polynomial=polynomial,
        target=ModularInteger(modulus=5, residue=0),
        vectors=(bad_vector,),
    )
    assert decoded.vectors == (bad_vector,)

    forged_term = ModularQuadraticCrossTerm.model_construct(
        left=-1, right=0, coefficient=1
    )
    forged_polynomial = ModularQuadraticPolynomial.model_construct(
        domain="Z_MOD_N",
        modulus=5,
        axis=("x", "y"),
        diagonal_residues=(0, 0),
        cross_terms=(forged_term,),
    )
    with pytest.raises(
        OperationDomainValidationError, match="support must be canonical"
    ):
        compute_modular_quadratic_fiber(
            forged_polynomial, ModularInteger(modulus=5, residue=0)
        )
