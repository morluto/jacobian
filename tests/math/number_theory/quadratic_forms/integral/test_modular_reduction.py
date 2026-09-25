"""Finite-ring reduction and evaluation of integral quadratic polynomials."""

import json
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.math.number_theory.quadratic_forms.integral._models import (
    IntegralQuadraticCrossTerm,
    IntegralQuadraticForm,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular import (
    ModularCoordinateVector,
    ModularEvaluationRequest,
    ModularQuadraticPolynomial,
    ModularReductionRequest,
    evaluate_modular_form,
    reduce_integral_form_modulus,
)


def test_composite_modulus_evaluation_matches_exhaustive_independent_oracle() -> None:
    form = IntegralQuadraticForm(
        axis=("x", "y"),
        diagonal_coefficients=(2, -1),
        cross_terms=(IntegralQuadraticCrossTerm(left=0, right=1, coefficient=3),),
    )
    reduced = reduce_integral_form_modulus(
        ModularReductionRequest(form=form, modulus=6)
    )
    assert reduced.diagonal_residues == (2, 5)
    assert reduced.cross_terms[0].coefficient == 3

    for x, y in product(range(6), repeat=2):
        vector = ModularCoordinateVector(modulus=6, axis=("x", "y"), coordinates=(x, y))
        actual = evaluate_modular_form(
            ModularEvaluationRequest(polynomial=reduced, vector=vector)
        )
        expected = (2 * x**2 + 3 * x * y - y**2) % 6
        assert (actual.modulus, actual.residue) == (6, expected)


def test_zero_ring_and_zero_mixed_residue_are_canonical() -> None:
    form = IntegralQuadraticForm(
        axis=("x", "y"),
        diagonal_coefficients=(7, -4),
        cross_terms=(IntegralQuadraticCrossTerm(left=0, right=1, coefficient=6),),
    )
    reduced = reduce_integral_form_modulus(
        ModularReductionRequest(form=form, modulus=1)
    )
    assert reduced.diagonal_residues == (0, 0)
    assert reduced.cross_terms == ()
    vector = ModularCoordinateVector(modulus=1, axis=("x", "y"), coordinates=(0, 0))
    assert (
        evaluate_modular_form(
            ModularEvaluationRequest(polynomial=reduced, vector=vector)
        ).residue
        == 0
    )


def test_target_parent_axis_and_canonical_residues_are_enforced() -> None:
    polynomial = ModularQuadraticPolynomial(
        modulus=6,
        axis=("x",),
        diagonal_residues=(2,),
        cross_terms=(),
    )
    with pytest.raises(ValidationError, match="polynomial and vector moduli"):
        ModularEvaluationRequest(
            polynomial=polynomial,
            vector=ModularCoordinateVector(modulus=5, axis=("x",), coordinates=(1,)),
        )
    with pytest.raises(ValidationError, match="axes must agree"):
        ModularEvaluationRequest(
            polynomial=polynomial,
            vector=ModularCoordinateVector(modulus=6, axis=("y",), coordinates=(1,)),
        )
    with pytest.raises(ValidationError, match="diagonal coefficients"):
        ModularQuadraticPolynomial(
            modulus=6, axis=("x",), diagonal_residues=(6,), cross_terms=()
        )


def test_serialized_values_round_trip_and_compose() -> None:
    polynomial_data = {
        "modulus": "11",
        "axis": ["a", "b"],
        "diagonal_residues": ["3", "9"],
        "cross_terms": [{"left": 0, "right": 1, "coefficient": "4"}],
    }
    vector_data = {"modulus": "11", "axis": ["a", "b"], "coordinates": ["7", "2"]}
    polynomial = ModularQuadraticPolynomial.model_validate_json(
        json.dumps(polynomial_data)
    )
    vector = ModularCoordinateVector.model_validate_json(json.dumps(vector_data))
    assert (
        evaluate_modular_form(
            ModularEvaluationRequest(polynomial=polynomial, vector=vector)
        ).residue
        == (3 * 49 + 9 * 4 + 4 * 14) % 11
    )


def test_zero_form_at_maximum_axis_is_admitted() -> None:
    axis = tuple(f"x{i}" for i in range(128))
    form = IntegralQuadraticForm(axis=axis, diagonal_coefficients=(0,) * len(axis))
    reduced = reduce_integral_form_modulus(
        ModularReductionRequest(form=form, modulus=97)
    )
    vector = ModularCoordinateVector(
        modulus=97, axis=axis, coordinates=(0,) * len(axis)
    )
    assert (
        evaluate_modular_form(
            ModularEvaluationRequest(polynomial=reduced, vector=vector)
        ).residue
        == 0
    )
