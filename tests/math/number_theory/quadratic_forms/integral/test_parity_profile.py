"""Exact mod-two profiles of integral quadratic forms."""

from __future__ import annotations

from itertools import combinations, product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import MathTool
from jacobian.math.number_theory.quadratic_forms.integral._models import (
    MAX_INTEGRAL_QUADRATIC_FORM_TERMS,
    IntegralQuadraticCrossTerm,
    IntegralQuadraticForm,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular import (
    ModularCoordinateVector,
    ModularEvaluationRequest,
    evaluate_modular_form,
)
from jacobian.math.number_theory.quadratic_forms.integral.parity import (
    ParityProfile,
    ParityProfileRequest,
    parity_profile,
)
from jacobian.math.number_theory.quadratic_forms.integral.parity._tools import TOOLS


def _form(
    axis: tuple[str, ...], coefficients: tuple[int, ...]
) -> IntegralQuadraticForm:
    diagonal = coefficients[: len(axis)]
    cross_pairs = tuple(combinations(range(len(axis)), 2))
    cross_coefficients = coefficients[len(axis) :]
    return IntegralQuadraticForm(
        axis=axis,
        diagonal_coefficients=diagonal,
        cross_terms=tuple(
            IntegralQuadraticCrossTerm(left=left, right=right, coefficient=value)
            for (left, right), value in zip(
                cross_pairs, cross_coefficients, strict=True
            )
            if value
        ),
    )


def _evaluate_directly(form: IntegralQuadraticForm, vector: tuple[int, ...]) -> int:
    value = sum(
        coefficient * coordinate**2
        for coefficient, coordinate in zip(
            form.diagonal_coefficients, vector, strict=True
        )
    )
    value += sum(
        term.coefficient * vector[term.left] * vector[term.right]
        for term in form.cross_terms
    )
    return value


def test_profiles_all_three_variable_forms_against_direct_f2_evaluation() -> None:
    axis = ("u", "v", "w")
    term_count = len(axis) + len(tuple(combinations(range(len(axis)), 2)))
    for coefficients in product((0, 1), repeat=term_count):
        form = _form(axis, coefficients)
        profile = parity_profile(form)
        polynomial = profile.polynomial

        assert polynomial.modulus == 2
        assert polynomial.axis == axis
        assert polynomial.diagonal_residues == tuple(
            _evaluate_directly(
                form,
                tuple(int(index == coordinate) for index in range(len(axis))),
            )
            % 2
            for coordinate in range(len(axis))
        )
        cross_residues = {
            (term.left, term.right): term.coefficient for term in polynomial.cross_terms
        }
        for left, right in combinations(range(len(axis)), 2):
            left_basis = tuple(int(index == left) for index in range(len(axis)))
            right_basis = tuple(int(index == right) for index in range(len(axis)))
            sum_basis = tuple(
                a + b for a, b in zip(left_basis, right_basis, strict=True)
            )
            polar_value = (
                _evaluate_directly(form, sum_basis)
                - _evaluate_directly(form, left_basis)
                - _evaluate_directly(form, right_basis)
            ) % 2
            assert cross_residues.get((left, right), 0) == polar_value

        assert profile.all_basis_norms_even is all(
            residue == 0 for residue in polynomial.diagonal_residues
        )
        for vector_coordinates in product((0, 1), repeat=len(axis)):
            reconstructed = sum(
                residue * coordinate**2
                for residue, coordinate in zip(
                    polynomial.diagonal_residues,
                    vector_coordinates,
                    strict=True,
                )
            )
            reconstructed += sum(
                coefficient * vector_coordinates[left] * vector_coordinates[right]
                for (left, right), coefficient in cross_residues.items()
            )
            assert reconstructed % 2 == _evaluate_directly(form, vector_coordinates) % 2


def test_signed_coefficients_and_zero_dimensional_convention() -> None:
    signed_form = IntegralQuadraticForm(
        axis=("x", "y"),
        diagonal_coefficients=(-3, 4),
        cross_terms=(IntegralQuadraticCrossTerm(left=0, right=1, coefficient=-5),),
    )
    profile = parity_profile(signed_form)
    assert profile.polynomial.diagonal_residues == (1, 0)
    assert tuple(
        (term.left, term.right, term.coefficient)
        for term in profile.polynomial.cross_terms
    ) == ((0, 1, 1),)
    assert not profile.all_basis_norms_even

    empty = parity_profile(IntegralQuadraticForm(axis=(), diagonal_coefficients=()))
    assert empty.polynomial.axis == ()
    assert empty.polynomial.diagonal_residues == ()
    assert empty.polynomial.cross_terms == ()
    assert empty.all_basis_norms_even


def test_profile_json_round_trip_and_derived_flag_validation() -> None:
    form = IntegralQuadraticForm(
        axis=("x", "y"),
        diagonal_coefficients=(2, 4),
        cross_terms=(IntegralQuadraticCrossTerm(left=0, right=1, coefficient=3),),
    )
    profile = parity_profile(form)
    round_tripped = ParityProfile.model_validate_json(profile.model_dump_json())
    assert round_tripped == profile
    assert round_tripped.all_basis_norms_even
    vector = ModularCoordinateVector(modulus=2, axis=("x", "y"), coordinates=(1, 1))
    actual = evaluate_modular_form(
        ModularEvaluationRequest(polynomial=round_tripped.polynomial, vector=vector)
    ).residue
    assert actual == 1
    assert actual == _evaluate_directly(form, (1, 1)) % 2
    with pytest.raises(ValidationError, match="must agree with the diagonal"):
        ParityProfile(
            polynomial=profile.polynomial,
            all_basis_norms_even=False,
        )


def test_full_source_support_boundary_has_admitted_complete_profile() -> None:
    axis = tuple(f"x{index}" for index in range(128))
    cross_pairs = tuple(combinations(range(len(axis)), 2))[
        : MAX_INTEGRAL_QUADRATIC_FORM_TERMS - len(axis)
    ]
    form = IntegralQuadraticForm(
        axis=axis,
        diagonal_coefficients=(0,) * len(axis),
        cross_terms=tuple(
            IntegralQuadraticCrossTerm(left=left, right=right, coefficient=1)
            for left, right in cross_pairs
        ),
    )
    profile = parity_profile(form)

    assert len(form.diagonal_coefficients) + len(form.cross_terms) == (
        MAX_INTEGRAL_QUADRATIC_FORM_TERMS
    )
    assert profile.polynomial.axis == axis
    assert profile.polynomial.diagonal_residues == (0,) * len(axis)
    assert len(profile.polynomial.cross_terms) == len(cross_pairs)
    assert profile.all_basis_norms_even


def test_profile_tool_is_declared_with_the_expected_typed_example() -> None:
    (tool,) = TOOLS
    assert isinstance(tool, MathTool)
    assert tool.operation_id == "quadratic_form.parity_profile.compute"
    assert tool.request_type is ParityProfileRequest
    assert tool.result_type is ParityProfile
    assert len(tool.examples) == 1
