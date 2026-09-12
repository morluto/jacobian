from __future__ import annotations

import json
import time
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS, CanonicalRational
from jacobian._execution import current_request_execution, request_execution
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.differential_forms import (
    FormComponent,
    PolynomialDifferentialForm,
    wedge,
)
from jacobian.math.polynomials.differential_forms._tools import WedgeRequest
from jacobian.math.polynomials.differential_forms.values import (
    MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS,
    MAX_DIFFERENTIAL_FORM_EXPONENT,
    MAX_DIFFERENTIAL_FORM_TERMS,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

R = CanonicalRational


def _poly_on_axis(
    variables: tuple[str, ...],
    *terms: tuple[int | Fraction, tuple[int, ...]],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=R.from_fraction(Fraction(coefficient)),
                    exponents=exponents,
                )
                for coefficient, exponents in terms
            )
        ),
    )


def _poly(*terms: tuple[int | Fraction, tuple[int, int]]) -> RationalPolynomial:
    return _poly_on_axis(("x", "y"), *terms)


def _form(
    degree: int, *components: tuple[tuple[int, ...], RationalPolynomial]
) -> PolynomialDifferentialForm:
    return PolynomialDifferentialForm(
        variables=("x", "y"),
        degree=degree,
        components=tuple(
            FormComponent(indices=indices, coefficient=coefficient)
            for indices, coefficient in components
        ),
    )


def test_wedge_computes_permutation_sign_and_exact_product() -> None:
    dx = _form(1, ((0,), _poly((2, (1, 0)))))
    dy = _form(1, ((1,), _poly((3, (0, 1)))))
    result = wedge(dx, dy)
    assert result.degree == 2
    assert result.components[0].indices == (0, 1)
    assert result.components[0].coefficient.polynomial.terms[0].coefficient == R(
        num=6, den=1
    )
    assert wedge(dy, dx).components[0].coefficient.polynomial.terms[0].coefficient == R(
        num=-6, den=1
    )


def test_degree_two_sign_and_basis_cancellation() -> None:
    variables = ("x", "y", "z")
    unit = _poly_on_axis(variables, (1, (0, 0, 0)))
    left = PolynomialDifferentialForm(
        variables=variables,
        degree=2,
        components=(
            FormComponent(indices=(0, 1), coefficient=unit),
            FormComponent(indices=(0, 2), coefficient=unit),
        ),
    )
    right = PolynomialDifferentialForm(
        variables=variables,
        degree=1,
        components=(
            FormComponent(indices=(1,), coefficient=unit),
            FormComponent(indices=(2,), coefficient=unit),
        ),
    )
    assert wedge(left, right).components == ()

    dxz = PolynomialDifferentialForm(
        variables=variables,
        degree=2,
        components=(FormComponent(indices=(0, 2), coefficient=unit),),
    )
    dy = PolynomialDifferentialForm(
        variables=variables,
        degree=1,
        components=(FormComponent(indices=(1,), coefficient=unit),),
    )
    result = wedge(dxz, dy)
    assert result.components[0].indices == (0, 1, 2)
    assert result.components[0].coefficient.polynomial.terms[0].coefficient == R(
        num=-1, den=1
    )


def test_zero_forms_on_zero_dimensional_axis_and_graded_zero_label() -> None:
    scalar = PolynomialDifferentialForm(
        variables=(),
        degree=0,
        components=(
            FormComponent(
                indices=(),
                coefficient=_poly_on_axis((), (3, ())),
            ),
        ),
    )
    product = wedge(scalar, scalar)
    assert product.variables == ()
    assert product.degree == 0
    assert product.components[0].coefficient.polynomial.terms[0].coefficient == R(
        num=9, den=1
    )

    top = _form(2)
    graded_zero = wedge(top, top)
    assert graded_zero.degree == 4
    assert graded_zero.components == ()
    assert wedge(graded_zero, _form(0)).degree == 4


def test_wedge_repeated_differentials_and_overdimension_are_zero() -> None:
    dx = _form(1, ((0,), _poly((1, (0, 0)))))
    assert wedge(dx, dx).components == ()
    two_form = _form(2, ((0, 1), _poly((1, (0, 0)))))
    assert wedge(two_form, dx).degree == 3
    assert wedge(two_form, dx).components == ()


def test_structural_zero_skips_irrelevant_exponent_admission() -> None:
    high = _form(
        1,
        ((0,), _poly((1, (MAX_DIFFERENTIAL_FORM_EXPONENT, 0)))),
    )
    assert wedge(high, high).components == ()


def test_odd_self_wedge_cancels_before_exponent_admission() -> None:
    coefficient = _poly((1, (MAX_DIFFERENTIAL_FORM_EXPONENT, 0)))
    alpha = _form(1, ((0,), coefficient), ((1,), coefficient))
    product = wedge(alpha, alpha)
    assert product.degree == 2
    assert product.components == ()


def test_proportional_odd_polynomial_wedges_cancel_before_support_cap() -> None:
    terms = tuple((1, (exponent, 0)) for exponent in range(128, -1, -1))
    polynomial = _poly(*terms)
    doubled = _poly(*((2, exponents) for _, exponents in terms))
    alpha = _form(1, ((0,), polynomial), ((1,), polynomial))
    beta = _form(1, ((0,), doubled), ((1,), doubled))
    product = wedge(alpha, beta)
    assert product.degree == 2
    assert product.components == ()


def test_signed_linear_factors_cancel_before_coefficient_cap() -> None:
    coefficient = 10**4095
    left = _form(
        0,
        ((), _poly((coefficient, (1, 0)), (coefficient, (0, 0)))),
    )
    right = _form(0, ((), _poly((1, (1, 0)), (-1, (0, 0)))))
    product = wedge(left, right)
    terms = {
        term.exponents: term.coefficient
        for term in product.components[0].coefficient.polynomial.terms
    }
    assert terms == {
        (2, 0): R(num=coefficient, den=1),
        (0, 0): R(num=-coefficient, den=1),
    }


def test_reciprocal_scalars_cancel_to_the_unit() -> None:
    coefficient = 10**4095
    left = _form(0, ((), _poly((coefficient, (0, 0)))))
    right = _form(0, ((), _poly((Fraction(1, coefficient), (0, 0)))))
    product = wedge(left, right)
    assert product.components[0].coefficient.polynomial.terms[0].coefficient == R(
        num=1, den=1
    )


def test_wedge_defers_height_cap_until_signed_monomials_cancel() -> None:
    coefficient = 10**2048 - 1
    left = _form(
        0,
        (
            (),
            _poly(
                (coefficient, (2, 4)),
                (coefficient, (1, 1)),
                (coefficient, (0, 0)),
            ),
        ),
    )
    right = _form(
        0,
        (
            (),
            _poly(
                (-coefficient, (2, 4)),
                (coefficient, (1, 3)),
                (coefficient, (0, 0)),
            ),
        ),
    )
    product = wedge(left, right)
    terms = {
        term.exponents: term.coefficient.num
        for term in product.components[0].coefficient.polynomial.terms
    }
    square = coefficient * coefficient
    assert terms[(2, 4)] == square
    assert max(len(str(abs(value))) for value in terms.values()) <= 4096


def test_proportional_odd_forms_cancel_before_exponent_admission() -> None:
    alpha = _form(
        1,
        ((0,), _poly((1, (MAX_DIFFERENTIAL_FORM_EXPONENT, 0)))),
        ((1,), _poly((1, (MAX_DIFFERENTIAL_FORM_EXPONENT, 0)))),
    )
    beta = _form(
        1,
        ((0,), _poly((2, (MAX_DIFFERENTIAL_FORM_EXPONENT, 0)))),
        ((1,), _poly((2, (MAX_DIFFERENTIAL_FORM_EXPONENT, 0)))),
    )
    product = wedge(alpha, beta)
    assert product.degree == 2
    assert product.components == ()


def test_wedge_reserves_output_support_before_convolution() -> None:
    left_terms = tuple((1, (exponent, 0)) for exponent in range(16, -1, -1))
    right_terms = tuple((1, (0, exponent)) for exponent in range(16, -1, -1))
    left = _form(1, ((0,), _poly(*left_terms)))
    right = _form(1, ((1,), _poly(*right_terms)))
    with pytest.raises(OperationResourceAdmissionError) as error:
        wedge(left, right)
    assert error.value.errors()[0]["type"] == "differential_form.wedge.output_budget"


def test_wedge_admits_collapsed_one_variable_support() -> None:
    terms = tuple((1, (exponent, 0)) for exponent in range(16, -1, -1))
    scalar = _form(0, ((), _poly(*terms)))
    product = wedge(scalar, scalar)
    exponents = tuple(
        term.exponents[0] for term in product.components[0].coefficient.polynomial.terms
    )
    assert exponents == tuple(range(32, -1, -1))


def test_wedge_admits_one_term_product_at_coefficient_bound() -> None:
    coefficient = 10**2048 - 1
    scalar = _form(0, ((), _poly((coefficient, (0, 0)))))
    product = wedge(scalar, scalar)
    assert (
        product.components[0].coefficient.polynomial.terms[0].coefficient.num
        == coefficient * coefficient
    )


def test_wedge_with_scalar_unit_preserves_full_coefficient_envelope() -> None:
    coefficient = 10**4095
    left = _form(0, ((), _poly((coefficient, (0, 0)))))
    unit = _form(0, ((), _poly((1, (0, 0)))))
    product = wedge(left, unit)
    assert (
        product.components[0].coefficient.polynomial.terms[0].coefficient.num
        == coefficient
    )


def test_wedge_with_scalar_unit_preserves_multicomponent_forms() -> None:
    variables = tuple(f"x{index}" for index in range(8))
    terms = tuple((10**255, (exponent,) + (0,) * 7) for exponent in range(255, -1, -1))
    coefficient = _poly_on_axis(variables, *terms)
    unit_poly = _poly_on_axis(variables, (1, (0,) * 8))
    alpha = PolynomialDifferentialForm(
        variables=variables,
        degree=1,
        components=tuple(
            FormComponent(indices=(index,), coefficient=coefficient)
            for index in range(7)
        ),
    )
    unit = PolynomialDifferentialForm(
        variables=variables,
        degree=0,
        components=(FormComponent(indices=(), coefficient=unit_poly),),
    )
    assert wedge(alpha, unit) == alpha
    assert wedge(unit, alpha) == alpha


def test_wedge_applies_negative_scalar_unit_in_both_orders() -> None:
    alpha = _form(1, ((0,), _poly((2, (1, 0)))), ((1,), _poly((3, (0, 1)))))
    minus_one = _form(0, ((), _poly((-1, (0, 0)))))
    negated = _form(1, ((0,), _poly((-2, (1, 0)))), ((1,), _poly((-3, (0, 1)))))
    assert wedge(alpha, minus_one) == negated
    assert wedge(minus_one, alpha) == negated
    assert wedge(minus_one, minus_one) == _form(0, ((), _poly((1, (0, 0)))))


def test_wedge_admits_coefficient_height_before_convolution() -> None:
    coefficient = 10**4_095
    left = _form(0, ((), _poly((coefficient, (0, 0)))))
    right = _form(0, ((), _poly((coefficient, (0, 0)))))
    with pytest.raises(OperationResourceAdmissionError) as error:
        wedge(left, right)
    assert error.value.errors()[0]["type"] == (
        "differential_form.wedge.coefficient_budget"
    )


def test_wedge_is_associative_and_serializable() -> None:
    dx = _form(1, ((0,), _poly((1, (1, 0)))))
    dy = _form(1, ((1,), _poly((1, (0, 1)))))
    scalar = _form(0, ((), _poly((2, (0, 0)))))
    left = wedge(wedge(scalar, dx), dy)
    right = wedge(scalar, wedge(dx, dy))
    assert left == right
    assert type(left).model_validate_json(left.model_dump_json()) == left


def test_form_rejects_unsorted_or_mismatched_components() -> None:
    with pytest.raises(ValidationError, match="component_order"):
        _form(
            1,
            ((1,), _poly((1, (0, 0)))),
            ((0,), _poly((1, (0, 0)))),
        )
    with pytest.raises(OperationDomainValidationError, match="identical"):
        wedge(
            _form(0, ((), _poly((1, (0, 0))))),
            PolynomialDifferentialForm(
                variables=("x", "z"),
                degree=0,
                components=(),
            ),
        )


def test_duplicate_differential_indices_are_rejected() -> None:
    with pytest.raises(ValidationError, match="component_indices"):
        _form(2, ((0, 0), _poly((1, (0, 0)))))


@pytest.mark.parametrize("indices", ((1, 0), (-1,)))
def test_exported_component_rejects_noncanonical_indices(
    indices: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError, match="component_indices"):
        FormComponent(indices=indices, coefficient=_poly((1, (0, 0))))


def test_overflowing_zero_form_degree_is_typed_admission() -> None:
    degree = 10**MAX_CANONICAL_INTEGER_DIGITS - 1
    zero = _form(degree)
    with pytest.raises(OperationResourceAdmissionError) as error:
        wedge(zero, zero)
    assert error.value.errors()[0]["type"] == "differential_form.wedge.degree_budget"


def test_serialized_differential_indices_have_a_schema_bound() -> None:
    variables = [f"x{index}" for index in range(MAX_POLYNOMIAL_VARIABLES)]
    payload = {
        "variables": variables,
        "degree": "1",
        "components": [
            {
                "indices": list(range(MAX_POLYNOMIAL_VARIABLES + 1)),
                "coefficient": {
                    "variables": variables,
                    "polynomial": {
                        "terms": [
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "exponents": [0] * MAX_POLYNOMIAL_VARIABLES,
                            }
                        ]
                    },
                },
            }
        ],
    }
    with pytest.raises(ValidationError, match="at most 8 items"):
        PolynomialDifferentialForm.model_validate_json(json.dumps(payload))


def test_wedge_groups_coefficient_growth_by_output_monomial() -> None:
    coefficient = 10**1000
    scalar = _form(
        0,
        ((), _poly((coefficient, (1, 0)), (coefficient, (0, 0)))),
    )
    product = wedge(scalar, scalar)
    terms = {
        term.exponents: term.coefficient.num
        for term in product.components[0].coefficient.polynomial.terms
    }
    square = coefficient * coefficient
    assert terms == {(2, 0): square, (1, 0): 2 * square, (0, 0): square}


def test_wedge_rejects_unadmitted_accumulator_growth_before_sum() -> None:
    first = 10**2500 + 7
    second = 10**2500 + 19
    left = _form(
        0,
        (
            (),
            _poly(
                (Fraction(1, second), (1, 0)),
                (Fraction(1, first), (0, 0)),
            ),
        ),
    )
    right = _form(0, ((), _poly((1, (1, 0)), (1, (0, 0)))))
    with pytest.raises(OperationResourceAdmissionError) as error:
        wedge(left, right)
    assert error.value.errors()[0]["type"] == (
        "differential_form.wedge.coefficient_budget"
    )


def test_wedge_convolution_checkpoints_during_nested_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def _observe(phase: str) -> None:
        observed.append(phase)

    monkeypatch.setattr(
        "jacobian.math.polynomials.differential_forms.operations.request_checkpoint",
        _observe,
    )
    monkeypatch.setattr(
        "jacobian.math.polynomials.differential_forms.operations._CONVOLUTION_CHECKPOINT_INTERVAL",
        4,
    )
    terms = tuple((1, (exponent, 0)) for exponent in range(8, -1, -1))
    scalar = _form(0, ((), _poly(*terms)))
    wedge(scalar, scalar)
    assert any("convolution" in phase for phase in observed)
    scalar = _form(0, ((), _poly((10**100, (0, 0)))))
    product = wedge(scalar, scalar)
    assert (
        product.components[0].coefficient.polynomial.terms[0].coefficient.num == 10**200
    )
    assert type(product).model_validate_json(product.model_dump_json()) == product
    zero = _form(16)
    squared = wedge(zero, zero)
    assert squared.degree == 32 and squared.components == ()
    assert wedge(squared, zero).degree == 48


def test_wedge_schema_publishes_coefficient_envelope() -> None:
    coefficient = FormComponent.model_json_schema()["properties"]["coefficient"]
    assert str(MAX_DIFFERENTIAL_FORM_TERMS) in coefficient["description"]
    assert str(MAX_DIFFERENTIAL_FORM_EXPONENT) in coefficient["description"]
    assert str(MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS) in coefficient["description"]
    left = WedgeRequest.model_json_schema()["properties"]["left"]
    assert str(MAX_DIFFERENTIAL_FORM_TERMS) in left["description"]


def test_native_wedge_revalidates_forged_operands() -> None:
    forged = PolynomialDifferentialForm.model_construct(
        variables=("x", "y"),
        degree=-1,
        components=(),
    )
    with pytest.raises(OperationDomainValidationError):
        wedge(forged, forged)
    with pytest.raises(OperationDomainValidationError):
        wedge(object(), _form(0))


def test_wedge_binds_owner_deadline_before_expansion() -> None:
    started = time.monotonic()
    unit = _form(0, ((), _poly((1, (0, 0)))))
    with request_execution(started):
        assert current_request_execution() is not None
        assert current_request_execution().deadline is None
        product = wedge(unit, unit)
        bound = current_request_execution().deadline
        assert product.components[0].coefficient.polynomial.terms[0].coefficient == R(
            num=1, den=1
        )
        assert bound is not None
        assert bound == started + 60.0


def test_wedge_cancels_denominator_contributions_before_lcm_cap() -> None:
    first = 10**4095 + 1
    second = 10**4095 + 3
    left = _form(
        0,
        (
            (),
            _poly(
                (Fraction(1, first), (3, 0)),
                (Fraction(1, second), (1, 0)),
                (Fraction(-1, first), (0, 0)),
            ),
        ),
    )
    right = _form(
        0,
        (
            (),
            _poly((1, (3, 0)), (1, (2, 0)), (1, (0, 0))),
        ),
    )
    product = wedge(left, right)
    terms = {
        term.exponents: term.coefficient
        for term in product.components[0].coefficient.polynomial.terms
    }
    assert terms[(3, 0)] == R.from_fraction(Fraction(1, second))
    assert max(len(str(abs(term.num))) for term in terms.values()) <= 4096
    assert max(len(str(term.den)) for term in terms.values()) <= 4096
    assert type(product).model_validate_json(product.model_dump_json()) == product


def test_proportional_odd_forms_cancel_before_product_height_cap() -> None:
    coefficient = 10**3000
    alpha = _form(
        1,
        ((0,), _poly((coefficient, (0, 0)))),
        ((1,), _poly((coefficient, (0, 0)))),
    )
    beta = _form(
        1,
        ((0,), _poly((2 * coefficient, (0, 0)))),
        ((1,), _poly((2 * coefficient, (0, 0)))),
    )
    product = wedge(alpha, beta)
    assert product.degree == 2
    assert product.components == ()
    assert type(product).model_validate_json(product.model_dump_json()) == product


def test_wedge_weights_convolution_budget_by_coefficient_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian.math.polynomials.differential_forms.operations.MAX_WEDGE_DIGIT_WORK",
        50,
    )
    left = _form(
        0,
        ((), _poly((10**4, (1, 0)), (10**4, (0, 0)))),
    )
    right = _form(
        0,
        ((), _poly((10**4, (1, 0)), (10**4, (0, 0)))),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        wedge(left, right)
    assert error.value.errors()[0]["type"] == "differential_form.wedge.term_budget"


def test_wedge_bounds_aggregate_serialized_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian.math.polynomials.differential_forms.operations.MAX_WEDGE_OUTPUT_DIGITS",
        20,
    )
    coefficient = 10**20
    left = _form(
        0,
        ((), _poly((coefficient, (1, 0)), (coefficient, (0, 0)))),
    )
    right = _form(
        0,
        ((), _poly((coefficient, (0, 1)), (coefficient, (0, 0)))),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        wedge(left, right)
    assert error.value.errors()[0]["type"] == "differential_form.wedge.output_budget"


def test_wedge_checkpoints_while_admitting_operands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def _observe(phase: str) -> None:
        observed.append(phase)

    monkeypatch.setattr(
        "jacobian.math.polynomials.differential_forms.operations.request_checkpoint",
        _observe,
    )
    monkeypatch.setattr(
        "jacobian.math.polynomials.differential_forms.operations._CONVOLUTION_CHECKPOINT_INTERVAL",
        1,
    )
    alpha = _form(1, ((0,), _poly((2, (1, 0)))), ((1,), _poly((3, (0, 1)))))
    unit = _form(0, ((), _poly((1, (0, 0)))))
    assert wedge(alpha, unit) == alpha
    assert any("operand admission" in phase for phase in observed)


def test_wedge_constructs_trusted_result_without_budget_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = _form(1, ((0,), _poly((2, (1, 0)))))
    right = _form(1, ((1,), _poly((3, (0, 1)))))

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("replayed polynomial budget")

    monkeypatch.setattr(
        "jacobian.math.polynomials.differential_forms.values.require_polynomial_budget",
        _boom,
    )
    result = wedge(left, right)
    assert result.degree == 2
    assert result.components[0].indices == (0, 1)
    assert result.components[0].coefficient.polynomial.terms[0].coefficient == R(
        num=6, den=1
    )
