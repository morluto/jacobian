"""Exact contract and reconstruction tests for Newton interpolation."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.interpolation._models import (
    DividedDifferencesRequest,
    InterpolationSamples,
    NewtonEvaluateRequest,
    NewtonFormRequest,
)
from jacobian.math.polynomials.interpolation._tools import TOOLS
from jacobian.math.polynomials.interpolation.operations import (
    divided_differences,
    evaluate_newton,
    newton_form,
)


def _q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(numerator, denominator)


def _samples(
    nodes: tuple[CanonicalRational, ...] = (_q(0), _q(1), _q(2)),
    values: tuple[CanonicalRational, ...] = (_q(1), _q(2), _q(5)),
) -> InterpolationSamples:
    return InterpolationSamples(nodes=nodes, values=values)


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "polynomial.interpolation.divided_differences.compute",
        "polynomial.interpolation.hermite.compute",
        "polynomial.interpolation.newton_form.compute",
        "polynomial.interpolation.newton_evaluate.compute",
    }


def test_divided_differences_are_canonical_rationals() -> None:
    result = divided_differences(DividedDifferencesRequest(samples=_samples()).samples)
    assert result.coefficients == (_q(1), _q(1), _q(1))


def test_newton_form_is_directly_evaluable() -> None:
    form = newton_form(NewtonFormRequest(samples=_samples()).samples)
    assert form.coefficients == (_q(1), _q(1), _q(1))
    request = NewtonEvaluateRequest(newton_form=form, evaluation_point=_q(3))
    result = evaluate_newton(request.newton_form, request.evaluation_point)
    assert result.result == _q(10)


def test_interpolation_reconstructs_every_sample() -> None:
    samples = _samples(
        nodes=(_q(0), _q(1, 2), _q(1), _q(3, 2)),
        values=(_q(1), _q(3, 2), _q(2), _q(11, 4)),
    )
    form = newton_form(NewtonFormRequest(samples=samples).samples)
    for node, expected in zip(samples.nodes, samples.values, strict=True):
        request = NewtonEvaluateRequest(newton_form=form, evaluation_point=node)
        result = evaluate_newton(request.newton_form, request.evaluation_point)
        assert result.result == expected


def test_rational_interpolation_has_exact_coefficients_and_evaluation() -> None:
    samples = _samples(
        nodes=(_q(0), _q(1, 2), _q(1)),
        values=(_q(1), _q(3, 2), _q(2)),
    )
    form = newton_form(NewtonFormRequest(samples=samples).samples)
    assert form.coefficients == (_q(1), _q(1), _q(0))
    request = NewtonEvaluateRequest(newton_form=form, evaluation_point=_q(3, 4))
    result = evaluate_newton(request.newton_form, request.evaluation_point)
    assert result.result == _q(7, 4)


def test_newton_coefficients_may_grow_beyond_input_digit_bound() -> None:
    left = 10**255 + 19
    right = 10**255 + 21
    form = newton_form(
        NewtonFormRequest(
            samples=_samples(
                nodes=(_q(0), _q(1)),
                values=(_q(1, left), _q(1, right)),
            )
        ).samples
    )

    assert len(form.coefficients[1].den) > 256


def test_equal_rational_nodes_are_rejected_before_division() -> None:
    with pytest.raises(ValidationError):
        InterpolationSamples(
            nodes=(_q(0), _q(1, 2), _q(1, 2)),
            values=(_q(1), _q(2), _q(3)),
        )


def test_samples_require_equal_lengths() -> None:
    with pytest.raises(ValidationError):
        InterpolationSamples(nodes=(_q(0), _q(1)), values=(_q(1),))


@pytest.mark.parametrize(
    "bad_rational",
    ["1/2", {"num": "1", "den": "0"}, {"num": "2", "den": "4"}],
)
def test_noncanonical_rational_inputs_are_rejected(bad_rational: object) -> None:
    with pytest.raises(ValidationError):
        InterpolationSamples.model_validate(
            {
                "nodes": [{"num": "0", "den": "1"}, bad_rational],
                "values": [
                    {"num": "1", "den": "1"},
                    {"num": "2", "den": "1"},
                ],
            }
        )
