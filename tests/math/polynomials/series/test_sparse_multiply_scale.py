"""Sparse truncated convolution admits actual retained incidences."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.series import TruncatedSeries, multiply


def _series(order: int, terms: dict[int, Fraction | int]) -> TruncatedSeries:
    return TruncatedSeries(
        variable="t",
        truncation_order=order,
        coefficients=tuple(
            CanonicalRational.from_fraction(Fraction(terms.get(i, 0)))
            for i in range(order)
        ),
    )


def test_long_sparse_product_retains_axis_and_exact_cancellation() -> None:
    left = _series(4096, {0: 1, 2048: 1})
    right = _series(4096, {0: 1, 2048: -1})
    result = multiply(left, right)
    assert result.result == _series(4096, {0: 1})
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_discarded_high_degree_pairs_do_not_consume_convolution_work() -> None:
    left = _series(2048, dict.fromkeys(range(1024, 2048), 1))
    result = multiply(left, left)
    assert result.result == _series(2048, {})


def test_dense_product_still_refuses_excessive_retained_work() -> None:
    source = _series(513, dict.fromkeys(range(513), 1))
    with pytest.raises(OperationResourceAdmissionError):
        multiply(source, source)


def test_sparse_fraction_product_matches_independent_coefficients() -> None:
    left = _series(1024, {0: Fraction(2, 3), 3: Fraction(-4, 7), 1000: 5})
    right = _series(1024, {1: Fraction(3, 5), 21: Fraction(7, 11), 1001: 2})
    expected = _series(
        1024,
        {
            1: Fraction(2, 5),
            4: Fraction(-12, 35),
            21: Fraction(14, 33),
            24: Fraction(-4, 11),
            1001: Fraction(13, 3),
            1004: Fraction(-8, 7),
            1021: Fraction(35, 11),
        },
    )
    assert multiply(left, right).result == expected
    assert multiply(right, left).result == expected
