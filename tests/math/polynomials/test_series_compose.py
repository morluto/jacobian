from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import pytest


def test_composition_stops_building_inner_powers_past_the_last_outer_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Horner-style power building must not outrun the nonzero outer terms.

    The kernel adds `inner_power` only when the outer coefficient is nonzero,
    so powers above the highest nonzero outer degree are computed and discarded.
    It built them anyway: with an outer series equal to `x^2` at order 512 and a
    dense 256-digit inner series, the loop reached `G^511`, whose numerator has
    about 131,000 digits - 32x the 4,096-digit result envelope admission is
    meant to enforce - and the admitted request took about 36 minutes. The
    admission path's skip-zero-powers branch already assumed the kernel stopped.
    """
    import jacobian.math.polynomials.series.operations as operations
    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials.series._models import TruncatedSeries

    def _rational(value: int) -> CanonicalRational:
        return CanonicalRational(num=value, den=1)

    order = 64
    outer = TruncatedSeries(
        variable="x",
        truncation_order=order,
        coefficients=(_rational(0), _rational(0), _rational(1))
        + (_rational(0),) * (order - 3),
    )
    inner = TruncatedSeries(
        variable="x",
        truncation_order=order,
        coefficients=(
            _rational(0),
            *(_rational(10**60 + i) for i in range(1, order)),
        ),
    )

    # an independent Frobenius-free oracle: outer is exactly x^2, so the
    # composition is the second power of the inner series
    series = [value.as_fraction() for value in inner.coefficients]
    expected = [Fraction(0)] * order
    for i, left in enumerate(series):
        if not left:
            continue
        for j, right in enumerate(series[: order - i]):
            if not right:
                continue
            expected[i + j] += left * right

    result = operations.compose(outer, inner)

    assert [value.as_fraction() for value in result.result.coefficients] == expected
    # the kernel stops after the last nonzero outer degree, so it convolves once
    calls = 0
    original_convolve = operations._cauchy_convolve

    def _counting_convolve(
        left: Sequence[Fraction], right: Sequence[Fraction], order: int
    ) -> list[Fraction]:
        nonlocal calls
        calls += 1
        return original_convolve(left, right, order)

    monkeypatch.setattr(operations, "_cauchy_convolve", _counting_convolve)
    operations.compose(outer, inner)
    # x^2 needs G^2, i.e. two convolutions, not order - 1 = 63
    assert calls == 2
