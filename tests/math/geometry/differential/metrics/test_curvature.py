"""Exact coordinate identities, independent diffgeom oracle and useful envelopes."""

import json
from fractions import Fraction
from itertools import product
from time import monotonic
from typing import Any

import pytest
from pydantic import ValidationError
from sympy import Matrix, cancel, symbols

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.differential.metrics import (
    RationalCoordinateMetric,
    RationalMetricCurvatureProfile,
    curvature_profile,
)
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    canonical_locus_guards,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
    sparse_rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.values import RationalFunction

x, y, z, w = symbols("x y z w")


def metric(
    matrix: list[Any], axis: tuple[str, ...] = ("x", "y")
) -> RationalCoordinateMetric:
    components = tuple(rational_function_from_sympy(value, axis) for value in matrix)
    return RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=components,
            retained_nonzero_denominators=canonical_locus_guards(
                component_denominators=tuple(value.denominator for value in components),
                variable_count=len(axis),
            ),
        )
    )


def expressions(components: tuple[RationalFunction, ...]) -> tuple[Any, ...]:
    return tuple(rational_function_to_sympy(value) for value in components)


def replay(result: RationalMetricCurvatureProfile) -> None:
    axis = symbols(result.metric.tensor.coordinate_axis)
    n = len(axis)
    g = Matrix(n, n, expressions(result.metric.tensor.components))
    inverse = Matrix(n, n, expressions(result.inverse_metric.components))
    gamma = expressions(result.connection.components)
    riemann = expressions(result.riemann.components)
    ricci = expressions(result.ricci.components)
    assert (g * inverse).applyfunc(cancel) == Matrix.eye(n)
    for k, i, j in product(range(n), repeat=3):
        expected = (
            sum(
                inverse[k, a]
                * (
                    g[a, j].diff(axis[i])
                    + g[a, i].diff(axis[j])
                    - g[i, j].diff(axis[a])
                )
                for a in range(n)
            )
            / 2
        )
        assert cancel(gamma[(k * n + i) * n + j] - expected) == 0
    for upper, k, i, j in product(range(n), repeat=4):
        expected = gamma[(upper * n + j) * n + k].diff(axis[i]) - gamma[
            (upper * n + i) * n + k
        ].diff(axis[j])
        expected += sum(
            gamma[(upper * n + i) * n + a] * gamma[(a * n + j) * n + k]
            - gamma[(upper * n + j) * n + a] * gamma[(a * n + i) * n + k]
            for a in range(n)
        )
        assert cancel(riemann[((upper * n + k) * n + i) * n + j] - expected) == 0
    assert all(
        cancel(
            ricci[k * n + j]
            - sum(riemann[((i * n + k) * n + i) * n + j] for i in range(n))
        )
        == 0
        for k, j in product(range(n), repeat=2)
    )
    scalar = expressions(result.scalar_curvature.components)[0]
    assert (
        cancel(
            scalar
            - sum(
                inverse[k, j] * ricci[k * n + j] for k, j in product(range(n), repeat=2)
            )
        )
        == 0
    )
    assert (
        RationalMetricCurvatureProfile.model_validate_json(result.model_dump_json())
        == result
    )


@pytest.mark.parametrize(
    ("matrix", "scalar"),
    [
        ([1, 0, 0, 1], 0),
        ([-1, 0, 0, 2], 0),
        ([1, 0, 0, x**2], 0),
        ([1 + x**2, x, x, 1], 0),
        ([4 / (1 + x * x + y * y) ** 2, 0, 0, 4 / (1 + x * x + y * y) ** 2], 2),
        ([1 / x**2, 0, 0, 1 / x**2], -2),
    ],
)
def test_known_metrics_and_all_defining_identities(
    matrix: list[Any], scalar: int
) -> None:
    result = curvature_profile(metric(matrix))
    replay(result)
    assert expressions(result.scalar_curvature.components) == (scalar,)


def test_independent_sympy_diffgeom_curved_nondiagonal_metric() -> None:
    from sympy.diffgeom import (
        CoordSystem,
        Manifold,
        Patch,
        TensorProduct,
        metric_to_Christoffel_2nd,
        metric_to_Riemann_components,
    )

    source = metric([2 + x, y, y, 3 + x])
    result = curvature_profile(source)
    chart = CoordSystem("chart", Patch("patch", Manifold("space", 2)), (x, y))
    a, b = chart.base_scalars()
    dx, dy = chart.base_oneforms()
    form = (
        (2 + a) * TensorProduct(dx, dx)
        + b * (TensorProduct(dx, dy) + TensorProduct(dy, dx))
        + (3 + a) * TensorProduct(dy, dy)
    )
    gamma = metric_to_Christoffel_2nd(form)
    riemann = metric_to_Riemann_components(form)
    for index, value in zip(
        product(range(2), repeat=3),
        expressions(result.connection.components),
        strict=True,
    ):
        assert cancel(value - gamma[index].subs({a: x, b: y})) == 0
    for index, value in zip(
        product(range(2), repeat=4), expressions(result.riemann.components), strict=True
    ):
        assert cancel(value - riemann[index].subs({a: x, b: y})) == 0
    replay(result)


def test_polar_locus_survives_zero_curvature() -> None:
    result = curvature_profile(metric([1, 0, 0, x * x]))
    assert any(value != 0 for value in expressions(result.connection.components))
    assert all(value == 0 for value in expressions(result.riemann.components))
    assert result.scalar_curvature.retained_nonzero_denominators
    # Every retained polynomial has precisely the source polar exclusion x=0.
    for guard in result.scalar_curvature.retained_nonzero_denominators:
        assert all(term.exponents[1] == 0 for term in guard.terms)
        assert len(guard.terms) == 1 and guard.terms[0].exponents[0] > 0


def test_large_constant_diagonal_four_dimensional_metric() -> None:
    source = metric(
        [10**127 if i == j else 0 for i in range(4) for j in range(4)],
        ("x", "y", "z", "w"),
    )
    result = curvature_profile(source)
    assert all(value == 0 for value in expressions(result.riemann.components))
    for i, value in enumerate(result.inverse_metric.components):
        if i % 5 == 0:
            assert value.numerator.terms[0].coefficient.as_fraction() == Fraction(
                1, 10**127
            )
        else:
            assert not value.numerator.terms
    assert not result.scalar_curvature.retained_nonzero_denominators


def test_four_dimensional_product_chart_and_high_exponent_line() -> None:
    result = curvature_profile(
        metric(
            [
                ((x, y, z, w)[i]) ** 2 if i == j else 0
                for i in range(4)
                for j in range(4)
            ],
            ("x", "y", "z", "w"),
        )
    )
    assert len(result.riemann.components) == 256
    assert all(value == 0 for value in expressions(result.riemann.components))
    line = curvature_profile(metric([x**64], ("x",)))
    assert expressions(line.connection.components) == (32 / x,)
    assert expressions(line.scalar_curvature.components) == (0,)


def test_coordinate_axis_permutation_transports_components() -> None:
    original = curvature_profile(metric([1 + x, 0, 0, 1 + y]))
    permuted = curvature_profile(metric([1 + y, 0, 0, 1 + x], ("y", "x")))
    values = expressions(original.connection.components)
    for (k, i, j), value in zip(
        product(range(2), repeat=3),
        expressions(permuted.connection.components),
        strict=True,
    ):
        assert cancel(value - values[((1 - k) * 2 + 1 - i) * 2 + 1 - j]) == 0
    assert permuted.metric.tensor.coordinate_axis == ("y", "x")


def test_shape_singularity_and_authored_nonreduced_source_rejections() -> None:
    with pytest.raises(ValidationError, match="symmetric"):
        metric([1, 1, 0, 1])
    with pytest.raises(OperationDomainValidationError, match="determinant"):
        curvature_profile(metric([1, x, x, x * x]))
    source = metric([x], ("x",)).model_dump(mode="json")
    field = source["tensor"]["components"][0]
    field["numerator"]["terms"][0]["exponents"] = [2]
    field["denominator"]["terms"][0]["exponents"] = [1]
    source["tensor"]["retained_nonzero_denominators"] = [field["denominator"]]
    with pytest.raises(OperationDomainValidationError, match="reduced canonical"):
        curvature_profile(
            RationalCoordinateMetric.model_validate_json(json.dumps(source))
        )


def test_expansion_rejection_and_earlier_deadline() -> None:
    source = metric(
        [1 if i == j else 0 for i in range(4) for j in range(4)], ("x", "y", "z", "w")
    ).model_dump(mode="json")
    for i in range(4):
        source["tensor"]["components"][5 * i]["numerator"]["terms"] = [
            {
                "exponents": [64 * int(j == k) for j in range(4)],
                "coefficient": {"num": "1", "den": "1"},
            }
            for k in range(4)
        ] + [{"exponents": [0, 0, 0, 0], "coefficient": {"num": "1", "den": "1"}}]
    with pytest.raises(OperationResourceAdmissionError):
        curvature_profile(
            RationalCoordinateMetric.model_validate_json(json.dumps(source))
        )
    with request_execution(monotonic()):
        bind_request_deadline(monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError):
            curvature_profile(metric([1], ("x",)))


def test_oversized_raw_pair_rejected_before_backend_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw source recognition remains bounded before any SymPy conversion."""
    axis = ("x", "y", "z", "w")
    exponents = sorted(product((0, 1, 2, 64), repeat=4), reverse=True)
    polynomial = {
        "terms": [
            {
                "exponents": list(exponent),
                "coefficient": {"num": 1, "den": 1},
            }
            for exponent in exponents
        ]
    }
    one = {
        "terms": [
            {
                "exponents": [0, 0, 0, 0],
                "coefficient": {"num": 1, "den": 1},
            }
        ]
    }
    raw_pair = {
        "domain": "QQ",
        "variables": list(axis),
        "numerator": polynomial,
        "denominator": polynomial,
    }
    zero = {
        "domain": "QQ",
        "variables": list(axis),
        "numerator": {"terms": []},
        "denominator": one,
    }
    source = RationalCoordinateMetric.model_validate(
        {
            "tensor": {
                "coordinate_axis": list(axis),
                "variance": ["COVARIANT", "COVARIANT"],
                "components": [
                    raw_pair if i == j else zero for i in range(4) for j in range(4)
                ],
                "retained_nonzero_denominators": [polynomial],
            }
        }
    )
    monkeypatch.setattr(
        "jacobian.math.geometry.differential.metrics.operations.recognize_canonical_rational_functions",
        lambda *args, **kwargs: pytest.fail("backend execution must follow admission"),
    )
    with pytest.raises(OperationResourceAdmissionError, match=r"allocation|work"):
        curvature_profile(source)


def test_four_dimensional_hyperbolic_metric() -> None:
    result = curvature_profile(
        metric(
            [1 / x**2 if i == j else 0 for i in range(4) for j in range(4)],
            ("x", "y", "z", "w"),
        )
    )
    assert len(result.riemann.components) == 256
    assert expressions(result.scalar_curvature.components) == (-12,)
    assert expressions(result.ricci.components) == tuple(
        -3 / x**2 if i == j else 0 for i in range(4) for j in range(4)
    )


def test_complete_locus_admitted_before_curvature_expansion() -> None:
    source = metric([x * y, 0, 0, x * y])
    guards = canonical_locus_guards(
        tuple(
            rational_function_from_sympy(x + c, ("x", "y")).numerator
            for c in range(1, 769)
        ),
        variable_count=2,
    )
    source = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=source.tensor.coordinate_axis,
            variance=source.tensor.variance,
            components=source.tensor.components,
            retained_nonzero_denominators=guards,
        )
    )
    # Shared formal denominator factors can reduce to distinct canonical
    # denominators. The inherited family already saturates the 768-guard
    # budget, so the independent xy determinant cannot be retained.
    with pytest.raises(OperationResourceAdmissionError, match="768 guards"):
        curvature_profile(source)


def test_inherited_determinant_and_inverse_guards_are_unioned_before_the_cap() -> None:
    axis = ("x",)
    guards = canonical_locus_guards(
        (
            rational_function_from_sympy(x**2, axis).numerator,
            rational_function_from_sympy(x, axis).numerator,
            *(
                rational_function_from_sympy(x + offset, axis).numerator
                for offset in range(1, 767)
            ),
        ),
        variable_count=1,
    )
    source = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=(rational_function_from_sympy(x**2, axis),),
            retained_nonzero_denominators=guards,
        )
    )
    result = curvature_profile(source)
    assert len(result.inverse_metric.retained_nonzero_denominators) == 768
    replay(result)


def test_nonmonic_determinant_unions_with_inherited_monic_guards() -> None:
    axis = ("x",)
    guards = canonical_locus_guards(
        (
            rational_function_from_sympy(x, axis).numerator,
            *(
                rational_function_from_sympy(x + offset, axis).numerator
                for offset in range(1, 768)
            ),
        ),
        variable_count=1,
    )
    source = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=(rational_function_from_sympy(2 * x, axis),),
            retained_nonzero_denominators=guards,
        )
    )
    result = curvature_profile(source)
    assert len(result.inverse_metric.retained_nonzero_denominators) == 768
    replay(result)


def test_conformal_flat_metric_counts_complete_denominator_powers() -> None:
    conformal = 1 + x**2 + y**2
    source = metric([conformal, 0, 0, conformal])
    result = curvature_profile(source)
    dens = {
        sparse_rational_polynomial_to_sympy(guard, ("x", "y")).as_expr().expand()
        for guard in result.inverse_metric.retained_nonzero_denominators
    }
    assert dens == {conformal.expand(), (conformal**2).expand(), (conformal**3).expand()}
    replay(result)

    extra = tuple(
        rational_function_from_sympy(x + offset, ("x", "y")).numerator
        for offset in range(1, 766)
    )
    admitted = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=source.tensor.coordinate_axis,
            variance=source.tensor.variance,
            components=source.tensor.components,
            retained_nonzero_denominators=canonical_locus_guards(
                extra, variable_count=2
            ),
        )
    )
    admitted_profile = curvature_profile(admitted)
    assert len(admitted_profile.inverse_metric.retained_nonzero_denominators) == 768
    replay(admitted_profile)

    saturated = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=source.tensor.coordinate_axis,
            variance=source.tensor.variance,
            components=source.tensor.components,
            retained_nonzero_denominators=canonical_locus_guards(
                extra
                + (
                    rational_function_from_sympy(x + 766, ("x", "y")).numerator,
                ),
                variable_count=2,
            ),
        )
    )
    with pytest.raises(OperationResourceAdmissionError, match="768 guards"):
        curvature_profile(saturated)
