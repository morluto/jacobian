"""Independent identities for complete fixed-node Lebesgue extrema."""

import time
from fractions import Fraction

import pytest
from sympy import Expr, Poly, Rational, Symbol, expand

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.analysis.approximation import lagrange_basis
from jacobian.math.analysis.approximation._models import RationalNodeSet
from jacobian.math.analysis.approximation.lebesgue import (
    LebesgueIntervalProfile,
    LebesgueIntervalSource,
    lebesgue_interval_profile,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.math.polynomials.values import RationalPolynomial


def source(
    nodes: tuple[int, ...], lower: int | Fraction, upper: int | Fraction
) -> LebesgueIntervalSource:
    def q(v: int | Fraction) -> CanonicalRational:
        return CanonicalRational.from_fraction(Fraction(v))

    return LebesgueIntervalSource(
        nodes=RationalNodeSet(nodes=tuple(q(v) for v in nodes)),
        interval=ClosedRationalInterval(lower=q(lower), upper=q(upper)),
    )


def expression(polynomial: RationalPolynomial) -> Expr:
    x = Symbol("x")
    return sum(
        Rational(t.coefficient.num, t.coefficient.den) * x ** t.exponents[0]
        for t in polynomial.polynomial.terms
    )


def test_complete_three_node_fixture() -> None:
    request = source((-1, 0, 1), -1, 1)
    result = lebesgue_interval_profile(request)
    x = Symbol("x")
    assert result.basis == lagrange_basis(request.nodes)
    assert [expression(c.polynomial) for c in result.cells] == [
        1 - x - x * x,
        1 + x - x * x,
    ]
    assert [c.basis_signs for c in result.cells] == [(1, 1, -1), (-1, 1, 1)]
    assert result.maximum.polynomial == (4, -5)
    assert [p.polynomial for p in result.maximizing_points] == [(2, 1), (2, -1)]
    assert all(
        c.endpoint_values == (CanonicalRational(num=1, den=1),) * 2
        for c in result.cells
    )
    assert [len(c.critical_points) for c in result.cells] == [1, 1]
    assert (
        LebesgueIntervalProfile.model_validate_json(result.model_dump_json()) == result
    )


@pytest.mark.parametrize(
    "nodes,lower,upper",
    [((3,), -5, 7), ((-1, 1), -1, 1), ((-1, 1), Fraction(-1, 3), Fraction(1, 2))],
)
def test_constant_intervals_retain_entire_maximum_locus(
    nodes: tuple[int, ...], lower: int | Fraction, upper: int | Fraction
) -> None:
    request = source(nodes, lower, upper)
    result = lebesgue_interval_profile(request)
    assert result.maximum.polynomial == (1, -1)
    assert result.maximizing_points == ()
    assert result.maximizing_intervals == (request.interval,)
    assert all(c.constant_on_cell and not c.critical_points for c in result.cells)


def test_subinterval_and_singleton_do_not_require_all_nodes() -> None:
    result = lebesgue_interval_profile(
        source((-1, 0, 1), Fraction(1, 4), Fraction(3, 4))
    )
    assert len(result.cells) == 1
    assert result.maximum.polynomial == (4, -5)
    point = source((-1, 0, 1), Fraction(1, 4), Fraction(1, 4))
    singleton = lebesgue_interval_profile(point)
    assert singleton.cells == ()
    assert singleton.maximum.polynomial == (16, -19)
    assert singleton.maximizing_intervals == (point.interval,)


def test_irrational_critical_values_and_affine_invariance() -> None:
    original = lebesgue_interval_profile(source((0, 1, 2, 3), 0, 3))
    translated = lebesgue_interval_profile(source((5, 7, 9, 11), 5, 11))
    assert original.maximum == translated.maximum
    assert original.maximum.polynomial == (27, -14, -49)
    assert original.maximum.real_root_index == 1
    assert len(original.maximizing_points) == 2
    x = Symbol("x")
    for cell in original.cells:
        derivative = Poly(expression(cell.polynomial), x).diff()
        a, b = cell.interval.lower.as_fraction(), cell.interval.upper.as_fraction()
        assert derivative.count_roots(
            Rational(a.numerator, a.denominator), Rational(b.numerator, b.denominator)
        ) == len(cell.critical_points)
        for critical in cell.critical_points:
            minimal = Poly.from_list(critical.point.polynomial, x)
            assert derivative.rem(minimal).is_zero
            value_polynomial = Poly.from_list(critical.value.polynomial, x)
            assert (
                Poly(
                    expand(
                        value_polynomial.as_expr().subs(x, expression(cell.polynomial))
                    ),
                    x,
                )
                .rem(minimal)
                .is_zero
            )


def test_32_node_exterior_query_remains_accepted() -> None:
    result = lebesgue_interval_profile(source(tuple(range(32)), 32, 33))
    assert len(result.cells) == 1
    assert not result.cells[0].critical_points
    assert result.maximizing_points[0].polynomial == (1, -33)
    # Independently evaluate product-defined cardinal polynomials at the maximum.
    expected = Fraction(0)
    for i in range(32):
        value = Fraction(1)
        for j in range(32):
            if i != j:
                value *= Fraction(33 - j, i - j)
        expected += abs(value)
    assert result.maximum.polynomial == (expected.denominator, -expected.numerator)


def test_inherited_deadline_is_not_replaced() -> None:
    request = source((-1, 0, 1), -1, 1)
    with request_execution(time.monotonic()):
        bind_request_deadline(time.monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError):
            lebesgue_interval_profile(request)


def test_sixteen_node_complete_profile_remains_accepted() -> None:
    result = lebesgue_interval_profile(source(tuple(range(16)), 0, 15))
    assert len(result.cells) == 15
    assert all(len(cell.critical_points) == 1 for cell in result.cells)
    assert len(result.maximizing_points) == 2
    assert result.maximum.real_root_index >= 0
    assert (
        LebesgueIntervalProfile.model_validate_json(result.model_dump_json()) == result
    )


def test_constant_query_retains_large_rational_endpoints() -> None:
    endpoint = 10**2000
    request = source((0,), -endpoint, endpoint)
    result = lebesgue_interval_profile(request)
    assert result.maximizing_intervals == (request.interval,)
    assert result.maximum.polynomial == (1, -1)
    assert (
        LebesgueIntervalProfile.model_validate_json(result.model_dump_json()) == result
    )


def test_rational_endpoints_are_not_capped_by_algebraic_height() -> None:
    result = lebesgue_interval_profile(
        source((-1, 0, 1), Fraction(1, 2), Fraction(1, 2) + Fraction(1, 10**2000))
    )
    assert result.cells
    assert (
        LebesgueIntervalProfile.model_validate_json(result.model_dump_json()) == result
    )


def test_unrepresentable_interior_degree_rejects_before_basis_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.analysis.approximation.lebesgue import operations

    def unexpected_expansion(*args: object) -> None:
        pytest.fail("basis expansion preceded complete source admission")

    monkeypatch.setattr(operations, "_poly_multiply", unexpected_expansion)
    with pytest.raises(OperationResourceAdmissionError, match="degree"):
        lebesgue_interval_profile(source(tuple(range(32)), 0, 31))
