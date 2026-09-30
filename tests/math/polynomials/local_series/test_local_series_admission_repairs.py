"""Regressions for the local-series native boundaries and cell bounds.

Each repaired case is paired with a negative control showing the enclosing
bound or domain rejection is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import pytest
from sympy import nextprime

from jacobian._exact import CanonicalRational
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series.newton_polygon import (
    LocalPolynomialCoefficient,
    LocalPolynomialInSeries,
)
from jacobian.math.polynomials.local_series.newton_transform import (
    MAX_NEWTON_TRANSFORM_OUTPUT_CELLS,
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
    coefficients: Sequence[int | Fraction],
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


# --- an exact-zero row is admitted before the degree guard excludes it ------


def _zero_row_source(y_degree: object) -> LocalPolynomialInSeries:
    """A linear polynomial plus one exact-zero row with a forged degree."""
    return LocalPolynomialInSeries.model_construct(
        variable="t",
        place="FINITE",
        center=_rational(0),
        coefficients=(
            LocalPolynomialCoefficient(y_degree=0, series=_window([-1, 0])),
            LocalPolynomialCoefficient(y_degree=1, series=_window([1, 0])),
            LocalPolynomialCoefficient.model_construct(y_degree=y_degree, series=None),
        ),
    )


@pytest.mark.parametrize("y_degree", [32_769, -1, True])
def test_smooth_jet_rejects_a_forged_exact_zero_row_degree(y_degree: object) -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_first_jet(
            SmoothBranchFirstJetRequest.model_construct(
                polynomial=_zero_row_source(y_degree), initial_root=_rational(1)
            )
        )
    assert _code(error.value) == "local_series.smooth_branch.row_degree"


def test_smooth_prefix_rejects_a_forged_exact_zero_row_degree() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_prefix(
            SmoothBranchPrefixRequest.model_construct(
                polynomial=_zero_row_source(32_769),
                initial_root=_rational(1),
                precision=4,
            )
        )
    assert _code(error.value) == "local_series.smooth_branch.row_degree"


# Negative controls: an admitted exact-zero row still leaves the mathematical
# degree unchanged, and a genuine high-degree row is still refused.
def test_admitted_exact_zero_row_still_leaves_the_degree_unchanged() -> None:
    with_zero_row = smooth_branch_first_jet(
        SmoothBranchFirstJetRequest(
            polynomial=_zero_row_source(32_768), initial_root=_rational(1)
        )
    )
    plain = smooth_branch_first_jet(
        SmoothBranchFirstJetRequest(
            polynomial=_polynomial(((0, _window([-1, 0])), (1, _window([1, 0])))),
            initial_root=_rational(1),
        )
    )
    assert with_zero_row.series.coefficients == plain.series.coefficients


# --- forged row degrees are refused before any row is indexed --------------


# F(t,y) = -1 + (1-t)y has the unique smooth branch y = 1 + t. The forged
# sources below present coefficient windows at degrees that bypass the model's
# own unique-increasing row validator.
def _forged_degree_source(
    rows: Sequence[tuple[int, Sequence[int]]],
) -> LocalPolynomialInSeries:
    return LocalPolynomialInSeries.model_construct(
        variable="t",
        place="FINITE",
        center=_rational(0),
        coefficients=tuple(
            LocalPolynomialCoefficient(y_degree=degree, series=_window(values))
            for degree, values in rows
        ),
    )


# Declaring y - 1 and (1-t)y - 1 at the same degree is not a polynomial at all:
# read together, the constant equation is -1 + 2y, whose root is 1/2, so the
# supplied root is no root. Before the repair the later row simply overwrote
# the earlier one and answered for a polynomial nobody had declared.
DUPLICATE_DEGREE_ROWS = ((0, [-1, 0]), (1, [1, 0]), (1, [1, -1]))


@pytest.mark.parametrize(
    "rows",
    [
        DUPLICATE_DEGREE_ROWS,
        ((0, [-1, 0]), (2, [0, 0]), (1, [1, -1])),
        ((1, [1, -1]), (0, [-1, 0])),
    ],
    ids=["duplicate", "out_of_order", "reversed"],
)
def test_smooth_jet_rejects_forged_row_degrees(
    rows: Sequence[tuple[int, Sequence[int]]],
) -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_first_jet(
            SmoothBranchFirstJetRequest.model_construct(
                polynomial=_forged_degree_source(rows), initial_root=_rational(1)
            )
        )
    assert _code(error.value) == "local_series.smooth_branch.row_order"


def test_smooth_prefix_rejects_forged_row_degrees() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        smooth_branch_prefix(
            SmoothBranchPrefixRequest.model_construct(
                polynomial=_forged_degree_source(DUPLICATE_DEGREE_ROWS),
                initial_root=_rational(1),
                precision=4,
            )
        )
    assert _code(error.value) == "local_series.smooth_branch.row_order"


# Negative control: the canonical order of the same windows is a branch of the
# declared polynomial. The forged duplicate had returned these exact values
# for a row it had silently dropped.
def test_canonical_row_order_still_answers_the_declared_branch() -> None:
    result = smooth_branch_first_jet(
        SmoothBranchFirstJetRequest(
            polynomial=_polynomial(((0, _window([-1, 0])), (1, _window([1, -1])))),
            initial_root=_rational(1),
        )
    )
    assert [value.as_fraction() for value in result.series.coefficients] == [
        Fraction(1),
        Fraction(1),
    ]


# --- retained source scalars are charged for both components ---------------

# A 256-digit prime keeps every proper fraction below in lowest terms, so each
# retained coefficient really does carry a wide numerator and a wide
# denominator rather than one wide component.
WIDE_DENOMINATOR = nextprime(10**255)


def _wide_source(terms: int) -> LocalPolynomialInSeries:
    """``y`` plus a wide degree-zero window with -1 as the edge constant."""
    return _polynomial(
        (
            (
                0,
                _window(
                    [
                        Fraction(-1),
                        *(
                            Fraction(WIDE_DENOMINATOR - index, WIDE_DENOMINATOR)
                            for index in range(1, terms + 1)
                        ),
                    ]
                ),
            ),
            (1, _window([Fraction(1), Fraction(0)])),
        )
    )


def test_newton_refuses_a_retained_source_wider_than_its_cell_bound() -> None:
    source = _wide_source(2_000)
    retained = sum(
        decimal_digit_width(value.numerator) + decimal_digit_width(value.denominator)
        for row in source.coefficients
        if row.series is not None
        for coefficient in row.series.coefficients
        if (value := coefficient.as_fraction())
    )
    assert retained > MAX_NEWTON_TRANSFORM_OUTPUT_CELLS
    with pytest.raises(OperationResourceAdmissionError) as error:
        newton_transform(
            NewtonTransformRequest(
                polynomial=source, edge_index=0, initial_root=_rational(1)
            )
        )
    assert _code(error.value) == "local_series.newton_transform_output_bound"


# Negative control: the same wide shape below the bound still transforms
# exactly, charging both components without refusing valid mathematics.
def test_newton_still_transforms_wide_proper_fractions_exactly() -> None:
    source = _wide_source(3)
    result = newton_transform(
        NewtonTransformRequest(
            polynomial=source, edge_index=0, initial_root=_rational(1)
        )
    )
    rows = result.transformed_polynomial.coefficients
    constant_row, linear_row = rows
    assert constant_row.series is not None and linear_row.series is not None
    assert [value.as_fraction() for value in constant_row.series.coefficients] == [
        Fraction(0),
        Fraction(WIDE_DENOMINATOR - 1, WIDE_DENOMINATOR),
    ]
    assert [value.as_fraction() for value in linear_row.series.coefficients] == [
        Fraction(1),
        Fraction(0),
    ]
    assert result.constant_term_valuation_lower_bound == 1
