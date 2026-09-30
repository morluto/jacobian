"""Regressions for the local-series native boundaries and cell bounds.

Each repaired case is paired with a negative control showing the enclosing
bound or domain rejection is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series.newton_polygon import (
    LocalPolynomialCoefficient,
    LocalPolynomialInSeries,
)
from jacobian.math.polynomials.local_series.newton_transform import (
    NewtonTransformRequest,
    newton_transform,
)
from jacobian.math.polynomials.local_series.smooth_branch import (
    SmoothBranchFirstJetRequest,
    SmoothBranchPrefixRequest,
    smooth_branch_first_jet,
    smooth_branch_prefix,
)
from jacobian.math.polynomials.local_series.values import TruncatedLaurentWindow


def _rational(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _window(
    coefficients: Sequence[int],
    *,
    lower: int = 0,
    place: str = "FINITE",
    center: CanonicalRational | None = None,
) -> TruncatedLaurentWindow:
    return TruncatedLaurentWindow(
        variable="t",
        place=place,  # type: ignore[arg-type]
        center=_rational(0) if center is None else center,
        valuation_lower=lower,
        precision=len(coefficients),
        coefficients=tuple(_rational(value) for value in coefficients),
    )


def _forged_window(
    coefficients: Sequence[int],
    *,
    place: str = "FINITE",
    center: CanonicalRational,
) -> TruncatedLaurentWindow:
    """A nested window that bypasses its own centre validation."""
    return TruncatedLaurentWindow.model_construct(
        variable="t",
        place=place,
        center=center,
        valuation_lower=0,
        precision=len(coefficients),
        coefficients=tuple(_rational(value) for value in coefficients),
    )


def _polynomial(
    rows: Sequence[tuple[int, TruncatedLaurentWindow | None]],
) -> LocalPolynomialInSeries:
    return LocalPolynomialInSeries(
        variable="t",
        place="FINITE",
        center=_rational(0),
        coefficients=tuple(
            LocalPolynomialCoefficient(y_degree=degree, series=window)
            for degree, window in rows
        ),
    )


def _cubic() -> LocalPolynomialInSeries:
    """Modulo t^2 this is (y - (2 + 3t))(y + 1 + 7t); the root is c = 2."""
    return _polynomial(
        (
            (0, _window([-2, -17, 11])),
            (1, _window([-1, 4, -31])),
            (2, _window([1, 0, 5])),
        )
    )


def _code(error: OperationDomainValidationError) -> str:
    return str(error.errors()[0]["type"])


# --- the Newton source centre is validated before it is read ---------------


def _forged_source_with_bad_center() -> LocalPolynomialInSeries:
    bad_center = CanonicalRational.model_construct(num=0, den=0)

    def forged(*values: int) -> TruncatedLaurentWindow:
        return TruncatedLaurentWindow.model_construct(
            variable="t",
            place="FINITE",
            center=bad_center,
            valuation_lower=0,
            precision=len(values),
            coefficients=tuple(_rational(value) for value in values),
        )

    return LocalPolynomialInSeries.model_construct(
        variable="t",
        place="FINITE",
        center=bad_center,
        coefficients=(
            LocalPolynomialCoefficient(y_degree=0, series=forged(-1, 1)),
            LocalPolynomialCoefficient(y_degree=1, series=forged(1, 0)),
        ),
    )


def test_newton_rejects_a_forged_source_center() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        newton_transform(
            NewtonTransformRequest(
                polynomial=_forged_source_with_bad_center(),
                edge_index=0,
                initial_root=_rational(1),
            )
        )
    assert _code(error.value) == "local_series.newton_transform_center_canonical"


@pytest.mark.parametrize("denominator", [0, -1])
def test_newton_rejects_a_non_positive_source_center_denominator(
    denominator: int,
) -> None:
    bad_center = CanonicalRational.model_construct(num=1, den=denominator)
    source = LocalPolynomialInSeries.model_construct(
        variable="t",
        place="FINITE",
        center=bad_center,
        coefficients=(
            LocalPolynomialCoefficient(
                y_degree=1, series=_forged_window([1, 0], center=bad_center)
            ),
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        newton_transform(
            NewtonTransformRequest(
                polynomial=source, edge_index=0, initial_root=_rational(1)
            )
        )
    assert _code(error.value) == "local_series.newton_transform_center_canonical"


def test_newton_rejects_a_non_reduced_source_center() -> None:
    bad_center = CanonicalRational.model_construct(num=2, den=4)
    source = LocalPolynomialInSeries.model_construct(
        variable="t",
        place="FINITE",
        center=bad_center,
        coefficients=(
            LocalPolynomialCoefficient(
                y_degree=1, series=_forged_window([1, 0], center=bad_center)
            ),
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        newton_transform(
            NewtonTransformRequest(
                polynomial=source, edge_index=0, initial_root=_rational(1)
            )
        )
    assert _code(error.value) == "local_series.newton_transform_center_canonical"


# Negative control: a genuine non-zero centre is still refused, with its own code.
def test_newton_still_requires_a_zero_centre() -> None:
    shifted = LocalPolynomialInSeries(
        variable="t",
        place="FINITE",
        center=_rational(1),
        coefficients=(
            LocalPolynomialCoefficient(
                y_degree=0, series=_window([-1, 1], center=_rational(1))
            ),
            LocalPolynomialCoefficient(
                y_degree=1, series=_window([1, 0], center=_rational(1))
            ),
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        newton_transform(
            NewtonTransformRequest(
                polynomial=shifted, edge_index=0, initial_root=_rational(1)
            )
        )
    assert _code(error.value) == "local_series.newton_transform_origin_only"


# --- smooth-branch roots are validated before conversion --------------------


def _forged_jet_request(root: CanonicalRational) -> SmoothBranchFirstJetRequest:
    return SmoothBranchFirstJetRequest.model_construct(
        polynomial=_cubic(), initial_root=root
    )


def test_smooth_jet_rejects_a_zero_root_denominator() -> None:
    request = _forged_jet_request(CanonicalRational.model_construct(num=1, den=0))
    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_first_jet(request)
    assert _code(error.value) == "local_series.smooth_branch.root_canonical"


def test_smooth_jet_rejects_a_non_reduced_root() -> None:
    request = _forged_jet_request(CanonicalRational.model_construct(num=2, den=2))
    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_first_jet(request)
    assert _code(error.value) == "local_series.smooth_branch.root_canonical"


def test_smooth_prefix_rejects_a_zero_root_denominator() -> None:
    request = SmoothBranchPrefixRequest.model_construct(
        polynomial=_cubic(),
        initial_root=CanonicalRational.model_construct(num=1, den=0),
        precision=4,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_prefix(request)
    assert _code(error.value) == "local_series.smooth_branch.root_canonical"


# Negative controls: a genuine non-root and a non-finite place are still
# refused with their own codes.
def test_smooth_jet_still_rejects_a_non_root() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_first_jet(
            SmoothBranchFirstJetRequest(polynomial=_cubic(), initial_root=_rational(5))
        )
    assert _code(error.value) == "local_series.smooth_branch.not_root"


def test_smooth_jet_still_rejects_a_non_finite_place() -> None:
    source = LocalPolynomialInSeries(
        variable="t",
        place="INFINITY",
        center=_rational(0),
        coefficients=(
            LocalPolynomialCoefficient(
                y_degree=0, series=_window([-1, 0], place="INFINITY")
            ),
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_first_jet(
            SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(1))
        )
    assert _code(error.value) == "local_series.smooth_branch.place"


# --- exact-zero rows do not raise the enforced branch degree ----------------


def test_exact_zero_row_does_not_raise_the_branch_degree() -> None:
    source = _polynomial(
        (
            (0, _window([-1, 0])),
            (1, _window([1, 0])),
            (17, None),
        )
    )
    result = smooth_branch_first_jet(
        SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(1))
    )
    assert [value.as_fraction() for value in result.series.coefficients] == [
        Fraction(1),
        Fraction(0),
    ]


def test_exact_zero_row_leaves_a_known_answer_unchanged() -> None:
    plain = smooth_branch_first_jet(
        SmoothBranchFirstJetRequest(polynomial=_cubic(), initial_root=_rational(2))
    )
    with_zero_row = smooth_branch_first_jet(
        SmoothBranchFirstJetRequest(
            polynomial=_polynomial(
                (
                    (0, _window([-2, -17, 11])),
                    (1, _window([-1, 4, -31])),
                    (2, _window([1, 0, 5])),
                    (17, None),
                )
            ),
            initial_root=_rational(2),
        )
    )
    assert with_zero_row.series.coefficients == plain.series.coefficients


# Negative control: a genuinely degree-17 polynomial is still refused.
def test_smooth_jet_still_enforces_the_degree_budget() -> None:
    source = _polynomial(
        (
            (0, _window([-1, 0])),
            (1, _window([1, 0])),
            (17, _window([1, 0])),
        )
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        smooth_branch_first_jet(
            SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(1))
        )
    assert _code(error.value) == "local_series.smooth_branch.degree_budget"


# --- the Newton cell bound measures retained width --------------------------


def test_wide_precision_transform_is_admitted() -> None:
    """Two precision-4096 rows for y - R transform without exceeding the bound."""
    big = 10**255
    source = _polynomial(
        (
            (0, _window([-big, *([0] * 4095)])),
            (1, _window([1, *([0] * 4095)])),
        )
    )
    result = newton_transform(
        NewtonTransformRequest(
            polynomial=source, edge_index=0, initial_root=_rational(big)
        )
    )
    assert len(result.transformed_polynomial.coefficients) >= 1


# Negative control: the row and slot budgets are unchanged.
def test_newton_still_enforces_its_row_budget() -> None:
    source = _polynomial(tuple((degree, _window([1, 0])) for degree in range(200)))
    with pytest.raises(
        (OperationDomainValidationError, OperationResourceAdmissionError)
    ):
        newton_transform(
            NewtonTransformRequest(
                polynomial=source, edge_index=0, initial_root=_rational(1)
            )
        )


def test_smooth_jet_still_enforces_its_row_budget() -> None:
    source = _polynomial(tuple((degree, _window([1, 0])) for degree in range(40)))
    with pytest.raises(OperationResourceAdmissionError) as error:
        smooth_branch_first_jet(
            SmoothBranchFirstJetRequest(polynomial=source, initial_root=_rational(1))
        )
    assert _code(error.value) == "local_series.smooth_branch.row_budget"
