"""Exact coordinate identities for rational covariant derivatives."""

from importlib import import_module
from itertools import product
from typing import Any

import pytest
from sympy import cancel, symbols

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.differential.metrics import RationalCoordinateMetric
from jacobian.math.geometry.differential.rational_tensor.covariant_derivative import (
    covariant_derivative,
)
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    TensorVariance,
    canonical_locus_guards,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
    sparse_rational_polynomial_to_sympy,
)

r, theta = symbols("r theta")


def tensor(
    values: list[Any],
    variance: tuple[TensorVariance, ...],
    axis: tuple[str, ...] = ("r", "theta"),
) -> RationalCoordinateTensor:
    return RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=variance,
        components=tuple(rational_function_from_sympy(value, axis) for value in values),
    )


def polar_metric() -> RationalCoordinateMetric:
    return RationalCoordinateMetric(
        tensor=tensor([1, 0, 0, r**2], ("COVARIANT", "COVARIANT"))
    )


def expressions(value: RationalCoordinateTensor) -> tuple[Any, ...]:
    return tuple(
        rational_function_to_sympy(component) for component in value.components
    )


def test_scalar_derivative_agrees_with_rational_gradient_and_retains_det_guard() -> (
    None
):
    result = covariant_derivative(polar_metric(), tensor([r**2], ()))

    assert expressions(result) == (2 * r, 0)
    assert len(result.retained_nonzero_denominators) == 1
    assert result.retained_nonzero_denominators[0].terms[0].exponents == (
        2,
        0,
    )


def test_polar_metric_is_covariantly_constant() -> None:
    result = covariant_derivative(polar_metric(), polar_metric().tensor)

    assert expressions(result) == (0,) * 8


def test_vector_and_covector_connection_signs() -> None:
    metric = polar_metric()
    vector = covariant_derivative(metric, tensor([1, 0], ("CONTRAVARIANT",)))
    covector = covariant_derivative(metric, tensor([1, 0], ("COVARIANT",)))

    assert expressions(vector) == (0, 0, 0, 1 / r)
    assert expressions(covector) == (0, 0, 0, r)


def test_mixed_tensor_formula_replays_exactly() -> None:
    metric = polar_metric()
    source = tensor([r, 0, 0, 1], ("CONTRAVARIANT", "COVARIANT"))
    result = covariant_derivative(metric, source)
    actual = expressions(result)
    gamma = (
        ((0, 0), (0, -r)),
        ((0, 1 / r), (1 / r, 0)),
    )
    source_values = expressions(source)
    expected = []
    for derivative_axis, upper, lower in product(range(2), repeat=3):
        value = source_values[upper * 2 + lower]
        expression = value.diff((r, theta)[derivative_axis])
        for replacement in range(2):
            expression += (
                gamma[upper][derivative_axis][replacement]
                * source_values[replacement * 2 + lower]
            )
            expression -= (
                gamma[replacement][derivative_axis][lower]
                * source_values[upper * 2 + replacement]
            )
        expected.append(expression)
    assert all(
        cancel(left - right) == 0 for left, right in zip(actual, expected, strict=True)
    )


def test_tensor_serialization_and_axis_mismatch() -> None:
    result = covariant_derivative(polar_metric(), tensor([r**2], ()))
    assert (
        RationalCoordinateTensor.model_validate_json(result.model_dump_json()) == result
    )
    with pytest.raises(OperationDomainValidationError, match="same coordinate axis"):
        covariant_derivative(polar_metric(), tensor([r**2], (), axis=("theta", "r")))


def test_rank_four_output_is_rejected_before_backend_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = RationalCoordinateMetric(
        tensor=tensor(
            [1 if i == j else 0 for i in range(4) for j in range(4)],
            ("COVARIANT", "COVARIANT"),
            axis=("x", "y", "z", "w"),
        )
    )
    source = tensor(
        [0] * 256,
        ("COVARIANT",) * 4,
        axis=("x", "y", "z", "w"),
    )
    operations = import_module(
        "jacobian.math.geometry.differential.rational_tensor.covariant_derivative.operations"
    )
    monkeypatch.setattr(
        operations,
        "evaluate_admitted_covariant_derivative",
        lambda *args, **kwargs: pytest.fail("backend execution must follow admission"),
    )
    with pytest.raises(OperationResourceAdmissionError, match="component"):
        covariant_derivative(metric, source)


def test_rank_eight_source_is_rejected_before_rank_nine_result() -> None:
    metric = RationalCoordinateMetric(
        tensor=tensor([1], ("COVARIANT", "COVARIANT"), axis=("x",))
    )
    source = tensor([1], ("COVARIANT",) * 8, axis=("x",))
    with pytest.raises(OperationResourceAdmissionError, match="rank-8"):
        covariant_derivative(metric, source)


def test_determinant_guards_are_capped_before_backend_expansion() -> None:
    x = symbols("x")
    axis = ("x", "y")
    metric = RationalCoordinateMetric(
        tensor=tensor(
            [x**64, 1, 1, x**64],
            ("COVARIANT", "COVARIANT"),
            axis=axis,
        )
    )
    with pytest.raises(OperationResourceAdmissionError, match="determinant locus"):
        covariant_derivative(metric, tensor([1], (), axis=axis))


def test_shared_output_denominators_do_not_double_count_locus_guards() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    inherited = canonical_locus_guards(
        tuple(
            rational_function_from_sympy(x + offset, axis).numerator
            for offset in range(1, 767)
        ),
        (rational_function_from_sympy(x + y, axis).numerator,),
        variable_count=2,
    )
    assert len(inherited) == 767
    identity = RationalCoordinateMetric(
        tensor=tensor([1, 0, 0, 1], ("COVARIANT", "COVARIANT"), axis=axis)
    )
    source = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=(),
        components=(rational_function_from_sympy(1 / (x + y), axis),),
        retained_nonzero_denominators=inherited,
    )
    result = covariant_derivative(identity, source)
    assert len(result.retained_nonzero_denominators) == 768
    assert all(
        cancel(left - right) == 0
        for left, right in zip(
            expressions(result),
            (-1 / (x + y) ** 2, -1 / (x + y) ** 2),
            strict=True,
        )
    )


def test_an_extra_distinct_output_denominator_still_exceeds_the_guard_cap() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    inherited = canonical_locus_guards(
        tuple(
            rational_function_from_sympy(x + offset, axis).numerator
            for offset in range(1, 768)
        ),
        (rational_function_from_sympy(x + y, axis).numerator,),
        variable_count=2,
    )
    assert len(inherited) == 768
    identity = RationalCoordinateMetric(
        tensor=tensor([1, 0, 0, 1], ("COVARIANT", "COVARIANT"), axis=axis)
    )
    source = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=(),
        components=(rational_function_from_sympy(1 / (x + y), axis),),
        retained_nonzero_denominators=inherited,
    )
    with pytest.raises(OperationResourceAdmissionError, match="768 guards"):
        covariant_derivative(identity, source)


def test_axis_specific_cancelled_denominators_are_counted_separately() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    denominator = rational_function_from_sympy(x**2 * y, axis).numerator
    inherited = canonical_locus_guards(
        tuple(
            rational_function_from_sympy(x + offset, axis).numerator
            for offset in range(1, 767)
        ),
        (denominator,),
        variable_count=2,
    )
    assert len(inherited) == 767
    identity = RationalCoordinateMetric(
        tensor=tensor([1, 0, 0, 1], ("COVARIANT", "COVARIANT"), axis=axis)
    )
    source = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=(),
        components=(rational_function_from_sympy(1 / (x**2 * y), axis),),
        retained_nonzero_denominators=inherited,
    )
    with pytest.raises(OperationResourceAdmissionError, match="768 guards"):
        covariant_derivative(identity, source)


def test_axis_specific_cancelled_denominators_admit_at_the_exact_cap() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    denominator = rational_function_from_sympy(x**2 * y, axis).numerator
    inherited = canonical_locus_guards(
        tuple(
            rational_function_from_sympy(x + offset, axis).numerator
            for offset in range(1, 766)
        ),
        (denominator,),
        variable_count=2,
    )
    assert len(inherited) == 766
    identity = RationalCoordinateMetric(
        tensor=tensor([1, 0, 0, 1], ("COVARIANT", "COVARIANT"), axis=axis)
    )
    source = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=(),
        components=(rational_function_from_sympy(1 / (x**2 * y), axis),),
        retained_nonzero_denominators=inherited,
    )
    result = covariant_derivative(identity, source)
    assert len(result.retained_nonzero_denominators) == 768
    assert all(
        cancel(left - right) == 0
        for left, right in zip(
            expressions(result),
            (-2 / (x**3 * y), -1 / (x**2 * y**2)),
            strict=True,
        )
    )


def test_package_exports_the_derivative_tensor_not_a_source_bound_profile() -> None:
    from jacobian.math.geometry.differential.rational_tensor import (
        covariant_derivative as package,
    )

    assert package.__all__ == ["covariant_derivative"]
    assert not hasattr(package, "RationalCovariantDerivativeProfile")


def test_non_monomial_axis_cancellations_are_counted_separately() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    denominator = rational_function_from_sympy((x + 1) * (y + 1), axis).numerator
    inherited = canonical_locus_guards(
        tuple(
            rational_function_from_sympy(x + offset, axis).numerator
            for offset in range(1, 767)
        ),
        (denominator,),
        variable_count=2,
    )
    assert len(inherited) == 767
    identity = RationalCoordinateMetric(
        tensor=tensor([1, 0, 0, 1], ("COVARIANT", "COVARIANT"), axis=axis)
    )
    source = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=(),
        components=(rational_function_from_sympy(1 / ((x + 1) * (y + 1)), axis),),
        retained_nonzero_denominators=inherited,
    )
    with pytest.raises(OperationResourceAdmissionError, match="768 guards"):
        covariant_derivative(identity, source)


def test_non_monomial_axis_cancellations_admit_at_the_exact_cap() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    denominator = rational_function_from_sympy((x + 1) * (y + 1), axis).numerator
    inherited = canonical_locus_guards(
        tuple(
            rational_function_from_sympy(x + offset, axis).numerator
            for offset in range(1, 766)
        ),
        (denominator,),
        variable_count=2,
    )
    assert len(inherited) == 766
    identity = RationalCoordinateMetric(
        tensor=tensor([1, 0, 0, 1], ("COVARIANT", "COVARIANT"), axis=axis)
    )
    source = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=(),
        components=(rational_function_from_sympy(1 / ((x + 1) * (y + 1)), axis),),
        retained_nonzero_denominators=inherited,
    )
    result = covariant_derivative(identity, source)
    assert len(result.retained_nonzero_denominators) == 768
    assert all(
        cancel(left - right) == 0
        for left, right in zip(
            expressions(result),
            (
                -1 / ((x + 1) ** 2 * (y + 1)),
                -1 / ((x + 1) * (y + 1) ** 2),
            ),
            strict=True,
        )
    )


def test_generated_determinant_nodes_share_one_guard_identity() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    inherited = canonical_locus_guards(
        tuple(
            rational_function_from_sympy(x + offset, axis).numerator
            for offset in range(1, 768)
        ),
        variable_count=2,
    )
    assert len(inherited) == 767
    source_metric = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=tuple(
                rational_function_from_sympy(value, axis) for value in (x, 1, 1, y)
            ),
            retained_nonzero_denominators=inherited,
        )
    )
    result = covariant_derivative(source_metric, source_metric.tensor)
    assert all(not component.numerator.terms for component in result.components)
    assert len(result.retained_nonzero_denominators) == 768


def test_cancelled_factor_of_a_generated_determinant_is_counted_separately() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    inherited = canonical_locus_guards(
        tuple(
            rational_function_from_sympy(x + offset, axis).numerator
            for offset in range(1, 768)
        ),
        variable_count=2,
    )
    assert len(inherited) == 767
    source_metric = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=tuple(
                rational_function_from_sympy(value, axis) for value in (x, y, y, y)
            ),
            retained_nonzero_denominators=inherited,
        )
    )
    source = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=("CONTRAVARIANT",),
        components=tuple(rational_function_from_sympy(value, axis) for value in (1, 0)),
        retained_nonzero_denominators=inherited,
    )
    with pytest.raises(OperationResourceAdmissionError, match="768 guards"):
        covariant_derivative(source_metric, source)


def test_cancelled_factor_of_a_generated_determinant_admits_at_the_exact_cap() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    inherited = canonical_locus_guards(
        tuple(
            rational_function_from_sympy(x + offset, axis).numerator
            for offset in range(1, 767)
        ),
        variable_count=2,
    )
    assert len(inherited) == 766
    source_metric = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=tuple(
                rational_function_from_sympy(value, axis) for value in (x, y, y, y)
            ),
            retained_nonzero_denominators=inherited,
        )
    )
    source = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=("CONTRAVARIANT",),
        components=tuple(rational_function_from_sympy(value, axis) for value in (1, 0)),
        retained_nonzero_denominators=inherited,
    )
    result = covariant_derivative(source_metric, source)
    assert len(result.retained_nonzero_denominators) == 768
    det = y * (x - y)
    assert any(
        cancel(sparse_rational_polynomial_to_sympy(guard, axis) - det) == 0
        for guard in result.retained_nonzero_denominators
    )
    assert any(
        cancel(sparse_rational_polynomial_to_sympy(guard, axis) - (x - y)) == 0
        for guard in result.retained_nonzero_denominators
    )


def test_inherited_determinant_factors_are_not_charged_twice() -> None:
    x, y, z, w = symbols("x y z w")
    axis = ("x", "y", "z", "w")
    factors = tuple(
        rational_function_from_sympy(symbol, axis).numerator for symbol in (x, y, z, w)
    )
    inherited = canonical_locus_guards(
        factors,
        tuple(
            rational_function_from_sympy(x + offset, axis).numerator
            for offset in range(1, 765)
        ),
        variable_count=4,
    )
    assert len(inherited) == 768
    metric = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=tuple(
                rational_function_from_sympy(value, axis)
                for value in (x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, w)
            ),
            retained_nonzero_denominators=inherited,
        )
    )
    source = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=(),
        components=(rational_function_from_sympy(1, axis),),
        retained_nonzero_denominators=inherited,
    )
    result = covariant_derivative(metric, source)
    assert len(result.retained_nonzero_denominators) == 768
    assert expressions(result) == (0, 0, 0, 0)


def test_structurally_zero_metric_uses_the_covariant_derivative_domain_code() -> None:
    metric = RationalCoordinateMetric(
        tensor=tensor([0], ("COVARIANT", "COVARIANT"), axis=("x",))
    )
    source = tensor([1], (), axis=("x",))
    with pytest.raises(OperationDomainValidationError) as rejected:
        covariant_derivative(metric, source)
    assert rejected.value.errors()[0]["type"].endswith(
        "covariant_derivative.singular_metric"
    )


def test_singular_metric_uses_the_covariant_derivative_domain_code() -> None:
    x = symbols("x")
    metric = RationalCoordinateMetric(
        tensor=tensor([1, x, x, x**2], ("COVARIANT", "COVARIANT"), axis=("x", "y"))
    )
    source = tensor([1, 0], ("COVARIANT",), axis=("x", "y"))
    with pytest.raises(OperationDomainValidationError) as rejected:
        covariant_derivative(metric, source)
    assert rejected.value.errors()[0]["type"].endswith(
        "covariant_derivative.singular_metric"
    )


@pytest.mark.parametrize("extra_terms", (0, 1))
def test_output_denominator_shared_with_inherited_locus_is_allocated_once(
    extra_terms: int,
) -> None:
    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials.values import (
        RationalPolynomialTerm,
        SparseRationalPolynomial,
    )

    axis = ("x", "y")
    x = symbols("x")
    one = CanonicalRational(num=1, den=1)
    support = tuple(
        RationalPolynomialTerm(coefficient=one, exponents=(i, j))
        for i in range(15, -1, -1)
        for j in range(15, -1, -1)
    )
    guards = tuple(
        SparseRationalPolynomial(
            terms=(
                *support[:-1],
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=offset, den=1), exponents=(0, 0)
                ),
            )
        )
        for offset in range(1, 512)
    )
    inherited = canonical_locus_guards(
        guards,
        (SparseRationalPolynomial(terms=support[: 247 + extra_terms]),),
        (rational_function_from_sympy(x, axis).numerator,),
        variable_count=2,
    )
    assert sum(len(guard.terms) for guard in inherited) == 131_064 + extra_terms
    metric = RationalCoordinateMetric(
        tensor=tensor([x, 0, 0, 1], ("COVARIANT", "COVARIANT"), axis=axis)
    )
    source = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=("CONTRAVARIANT",),
        components=tuple(rational_function_from_sympy(value, axis) for value in (1, 0)),
        retained_nonzero_denominators=inherited,
    )
    if extra_terms:
        with pytest.raises(OperationResourceAdmissionError, match="allocation bounds"):
            covariant_derivative(metric, source)
        return
    result = covariant_derivative(metric, source)
    assert result.retained_nonzero_denominators == inherited
    assert expressions(result) == (1 / (2 * x), 0, 0, 0)
