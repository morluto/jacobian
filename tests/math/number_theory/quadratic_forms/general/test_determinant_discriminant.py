from __future__ import annotations

import json
from fractions import Fraction
from itertools import permutations
from math import lcm

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.quadratic_forms.general.determinant_discriminant._models import (
    MAX_POLAR_DETERMINANT_AXIS,
    DeterminantDiscriminantRequest,
)
from jacobian.math.number_theory.quadratic_forms.general.determinant_discriminant._tools import (
    TOOLS,
)
from jacobian.math.number_theory.quadratic_forms.general.determinant_discriminant.operations import (
    polar_gram_determinant_discriminant,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def _r(value: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value, denominator)


def _form(
    diagonal: tuple[CanonicalRational, ...],
    cross: tuple[tuple[int, int, CanonicalRational], ...] = (),
) -> RationalQuadraticForm:
    return RationalQuadraticForm(
        axis=tuple(f"x{index}" for index in range(len(diagonal))),
        diagonal_coefficients=diagonal,
        cross_terms=tuple(
            QuadraticCrossTerm(left=left, right=right, coefficient=value)
            for left, right, value in cross
        ),
    )


def _permutation_determinant(
    matrix: tuple[tuple[Fraction, ...], ...],
) -> Fraction:
    """Independent Leibniz oracle for small dimensions."""

    total = Fraction()
    n = len(matrix)
    for permutation in permutations(range(n)):
        inversions = sum(
            permutation[i] > permutation[j] for i in range(n) for j in range(i + 1, n)
        )
        term = Fraction((-1) ** inversions)
        for row, column in enumerate(permutation):
            term *= matrix[row][column]
        total += term
    return total


def _polar_gram(form: RationalQuadraticForm) -> tuple[tuple[Fraction, ...], ...]:
    n = len(form.axis)
    entries = [[Fraction() for _ in range(n)] for _ in range(n)]
    for index, coefficient in enumerate(form.diagonal_coefficients):
        entries[index][index] = 2 * coefficient.as_fraction()
    for term in form.cross_terms:
        value = term.coefficient.as_fraction()
        entries[term.left][term.right] = value
        entries[term.right][term.left] = value
    return tuple(tuple(row) for row in entries)


def test_signed_discriminant_matches_independent_binary_polynomial_oracle() -> None:
    # Q=x^2+xy+y^2 has full-polar G=[[2,1],[1,2]], det(G)=3;
    # the classical binary polynomial discriminant is 1^2-4*1*1=-3.
    form = _form((_r(1), _r(1)), ((0, 1, _r(1)),))
    result = polar_gram_determinant_discriminant(
        DeterminantDiscriminantRequest(form=form)
    )

    assert _permutation_determinant(_polar_gram(form)) == 3
    assert result.polar_gram_determinant == _r(3)
    assert result.signed_discriminant == _r(-3)
    assert result.convention == "SIGNED_FULL_POLAR_GRAM_V1"
    assert "quadratic_form.determinant_discriminant.compute" in {
        tool.operation_id for tool in TOOLS
    }


@pytest.mark.parametrize(
    ("form", "determinant", "signed"),
    (
        (_form(()), 1, 1),
        (_form((_r(7),)), 14, 14),
        (_form((_r(1), _r(0))), 0, 0),
        (_form((_r(0), _r(0)), ((0, 1, _r(1)),)), -1, 1),
        (_form((_r(2), _r(3)), ((0, 1, _r(5)),)), -1, 1),
    ),
)
def test_empty_degenerate_and_odd_cross_cases(
    form: RationalQuadraticForm, determinant: int, signed: int
) -> None:
    result = polar_gram_determinant_discriminant(
        DeterminantDiscriminantRequest(form=form)
    )
    assert result.polar_gram_determinant == _r(determinant)
    assert result.signed_discriminant == _r(signed)


def test_positive_dimensional_all_zero_form_has_zero_determinant() -> None:
    form = _form((_r(0), _r(0), _r(0)))
    result = polar_gram_determinant_discriminant(
        DeterminantDiscriminantRequest(form=form)
    )

    assert result.polar_gram_determinant == _r(0)
    assert result.signed_discriminant == _r(0)


def test_determinant_transforms_by_square_under_rational_basis_change() -> None:
    # Q=x^2+xy+y^2; P=diag(2,1) gives Q(2u,v)=4u^2+2uv+v^2.
    # This explicit Q' obeys det(G_Q')=det(P)^2*det(G_Q)=4*3.
    source = _form((_r(1), _r(1)), ((0, 1, _r(1)),))
    changed = _form((_r(4), _r(1)), ((0, 1, _r(2)),))
    source_result = polar_gram_determinant_discriminant(
        DeterminantDiscriminantRequest(form=source)
    )
    changed_result = polar_gram_determinant_discriminant(
        DeterminantDiscriminantRequest(form=changed)
    )
    assert source_result.polar_gram_determinant == _r(3)
    assert changed_result.polar_gram_determinant == _r(12)
    assert changed_result.polar_gram_determinant.as_fraction() == (
        Fraction(4) * source_result.polar_gram_determinant.as_fraction()
    )


def test_fractional_form_uses_full_polar_gram_not_coefficient_matrix() -> None:
    # For Q=2x^2+3xy+5y^2, det of coefficient matrix [[2,3/2],[3/2,5]]
    # is 31/4, while det of full polar Gram [[4,3],[3,10]] is 31.
    form = _form((_r(2), _r(5)), ((0, 1, _r(3)),))
    result = polar_gram_determinant_discriminant(
        DeterminantDiscriminantRequest(form=form)
    )
    assert _permutation_determinant(_polar_gram(form)) == 31
    assert result.polar_gram_determinant == _r(31)


def test_fractional_row_clearing_matches_independent_determinant() -> None:
    # G=[[1,1/3],[1/3,4/5]], so det(G)=31/45.
    form = _form((_r(1, 2), _r(2, 5)), ((0, 1, _r(1, 3)),))
    result = polar_gram_determinant_discriminant(
        DeterminantDiscriminantRequest(form=form)
    )
    expected = _permutation_determinant(_polar_gram(form))
    assert expected == Fraction(31, 45)
    assert result.polar_gram_determinant == CanonicalRational.from_fraction(expected)
    assert result.signed_discriminant == CanonicalRational.from_fraction(-expected)


def test_three_dimensional_bareiss_with_row_pivot_matches_permutation_oracle() -> None:
    # G=[[0,1,2],[1,1,3],[2,3,5]] requires a row pivot; its determinant is 3.
    form = _form(
        (_r(0), _r(1, 2), _r(5, 2)),
        ((0, 1, _r(1)), (0, 2, _r(2)), (1, 2, _r(3))),
    )
    expected = _permutation_determinant(_polar_gram(form))
    assert expected == 3
    result = polar_gram_determinant_discriminant(
        DeterminantDiscriminantRequest(form=form)
    )
    assert result.polar_gram_determinant == _r(3)
    assert result.signed_discriminant == _r(-3)


def test_catalog_tool_runs_its_example_and_result_survives_json_round_trip() -> None:
    operation = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "quadratic_form.determinant_discriminant.compute"
    )
    example = operation.examples[0]
    request = operation.request_type.model_validate_json(
        json.dumps(example.input), strict=True
    )
    result = operation.run(request)
    restored = operation.result_type.model_validate_json(result.model_dump_json())
    assert restored.polar_gram_determinant == _r(3)
    assert restored.signed_discriminant == _r(-3)


def test_result_rejects_signed_scalar_inconsistent_with_dimension() -> None:
    form = _form((_r(1),))
    result = polar_gram_determinant_discriminant(
        DeterminantDiscriminantRequest(form=form)
    )
    malformed = json.loads(result.model_dump_json())
    malformed["signed_discriminant"] = {"num": "99", "den": "1"}

    with pytest.raises(ValidationError, match="dimension-derived sign"):
        type(result).model_validate_json(json.dumps(malformed), strict=True)


@pytest.mark.parametrize(
    "bad_request",
    (
        object(),
        DeterminantDiscriminantRequest.model_construct(form=1),
        DeterminantDiscriminantRequest.model_construct(
            form=RationalQuadraticForm.model_construct(axis=("x",), domain="QQ")
        ),
    ),
)
def test_native_request_revalidates_constructed_model_internals(
    bad_request: object,
) -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    with pytest.raises(OperationDomainValidationError):
        polar_gram_determinant_discriminant(bad_request)  # type: ignore[arg-type]


def test_axis_boundary_is_accepted_and_next_dimension_rejected() -> None:
    accepted = _form(tuple(_r(1) for _ in range(MAX_POLAR_DETERMINANT_AXIS)))
    result = polar_gram_determinant_discriminant(
        DeterminantDiscriminantRequest(form=accepted)
    )
    assert result.polar_gram_determinant == _r(2**MAX_POLAR_DETERMINANT_AXIS)

    rejected = _form(tuple(_r(1) for _ in range(MAX_POLAR_DETERMINANT_AXIS + 1)))
    with pytest.raises(OperationResourceAdmissionError) as error:
        polar_gram_determinant_discriminant(
            DeterminantDiscriminantRequest(form=rejected)
        )
    assert error.value.errors()[0]["type"] == "quadratic_form.determinant_axis_bound"


def test_height_admission_rejects_before_fraction_free_kernel() -> None:
    dimension = 16
    # These denominators are pairwise coprime and each row sees all 16.
    # LCM row clearing therefore gives an output-height bound above the result
    # envelope even though every individual input coefficient is bounded.
    base = 10**240 * lcm(*range(1, dimension))
    denominators = tuple(1 + index * base for index in range(1, dimension + 1))
    form = _form(
        tuple(_r(1, denominators[-1]) for _ in range(dimension)),
        tuple(
            (
                left,
                right,
                _r(
                    1,
                    denominators[
                        (2 * left) % (dimension - 1)
                        if right == dimension - 1
                        else (left + right) % (dimension - 1)
                    ],
                ),
            )
            for left in range(dimension)
            for right in range(left + 1, dimension)
        ),
    )
    request = DeterminantDiscriminantRequest(form=form)
    with pytest.raises(OperationResourceAdmissionError) as error:
        polar_gram_determinant_discriminant(request)
    assert error.value.errors()[0]["type"] in {
        "quadratic_form.determinant_result_height_bound",
        "quadratic_form.determinant_intermediate_height_bound",
    }
