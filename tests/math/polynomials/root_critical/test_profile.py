"""Independent exact checks for root--critical distance profiles."""

from __future__ import annotations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.root_critical._models import (
    RootCriticalDistanceProfileRequest,
)
from jacobian.math.polynomials.root_critical._tools import TOOLS
from jacobian.math.polynomials.root_critical.operations import (
    root_critical_distance_profile,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(*coefficients: tuple[int, int]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("z",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=value, den=1),
                    exponents=(exponent,),
                )
                for exponent, value in sorted(coefficients, reverse=True)
                if value
            )
        ),
    )


def test_cubic_profile_has_complete_rows_and_unit_distances() -> None:
    result = root_critical_distance_profile(_polynomial((3, 1), (0, -1)))

    assert len(result.roots) == 3
    assert len(result.critical_points) == 1
    assert len(result.pairs) == 3
    assert all(row.distance_squared.polynomial == (1, -1) for row in result.pairs)
    assert all(row.kind == "POSITIVE" for row in result.pairs)
    assert all(row.isolating_interval.lower.as_fraction() == 1 for row in result.pairs)


def test_repeated_root_and_zero_distance_are_retained() -> None:
    result = root_critical_distance_profile(_polynomial((3, 1), (1, -3), (0, 2)))

    assert [root.multiplicity for root in result.roots] == [2, 1]
    assert [critical.multiplicity for critical in result.critical_points] == [1, 1]
    zero_rows = [row for row in result.pairs if row.kind == "ZERO_DISTANCE"]
    assert len(zero_rows) == 1
    assert zero_rows[0].distance_squared.polynomial == (1, 0)


def test_nonreal_roots_use_exact_nonnegative_squared_distances() -> None:
    # (z^2+1)(z-2), with derivative roots 1 and 1/3.  The expected values
    # are obtained independently from the explicit roots ±i and 2.
    result = root_critical_distance_profile(
        _polynomial((3, 1), (2, -2), (1, 1), (0, -2))
    )

    assert len(result.pairs) == 6
    values = {tuple(row.distance_squared.polynomial) for row in result.pairs}
    assert values == {(1, -1), (1, -2), (9, -10), (9, -25)}
    assert all(row.isolating_interval.lower.as_fraction() >= 0 for row in result.pairs)
    assert any(
        root.rectangle.imaginary_lower.as_fraction() < 0 for root in result.roots
    )
    assert any(
        root.rectangle.imaginary_upper.as_fraction() > 0 for root in result.roots
    )


def test_admission_rejects_incomplete_pair_budget_before_expansion() -> None:
    request = RootCriticalDistanceProfileRequest(
        polynomial=_polynomial((3, 1), (0, -1)),
        max_pair_rows=2,
    )
    with pytest.raises(OperationResourceAdmissionError):
        TOOLS[0].run(request)


def test_constant_polynomial_is_outside_profile_domain() -> None:
    with pytest.raises(OperationDomainValidationError):
        root_critical_distance_profile(_polynomial((0, 3)))


def test_operation_is_published_with_stable_id() -> None:
    assert TOOLS[0].operation_id == "polynomial.root_critical_distance_profile.compute"
