"""Tests for hyperplane arrangement operations."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.arrangements._models import (
    MAX_GENERIC_FORMULA_INDEX,
    ChamberCountRequest,
    ChamberCountResult,
    CharacteristicPolynomialRequest,
    CharacteristicPolynomialResult,
    HyperplaneArrangementRequest,
    HyperplaneArrangementResult,
    RationalHyperplane,
)
from jacobian.math.geometry.arrangements._tools import TOOLS
from jacobian.math.geometry.arrangements.operations import (
    MAX_GENERIC_FORMULA_WORK,
    arrangement,
    chamber_count,
    characteristic_polynomial,
)


def compute_arrangement(
    request: HyperplaneArrangementRequest,
) -> HyperplaneArrangementResult:
    return arrangement(request.ambient_dimension, request.hyperplanes)


def compute_characteristic_polynomial(
    request: CharacteristicPolynomialRequest,
) -> CharacteristicPolynomialResult:
    return characteristic_polynomial(
        request.ambient_dimension, request.hyperplane_count
    )


def compute_chamber_count(request: ChamberCountRequest) -> ChamberCountResult:
    return chamber_count(request.ambient_dimension, request.hyperplane_count)


def _r(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(num, den)


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "arrangement.construct",
        "arrangement.characteristic_polynomial.compute",
        "arrangement.chamber_count.compute",
    }


def test_arrangement_central() -> None:
    request = HyperplaneArrangementRequest(
        ambient_dimension=2,
        hyperplanes=(
            RationalHyperplane(coefficients=(_r(1), _r(0)), constant=_r(0)),
            RationalHyperplane(coefficients=(_r(0), _r(1)), constant=_r(0)),
        ),
    )
    result = compute_arrangement(request)
    assert result.is_central is True
    assert result.hyperplane_count == 2


def test_arrangement_noncentral() -> None:
    request = HyperplaneArrangementRequest(
        ambient_dimension=2,
        hyperplanes=(
            RationalHyperplane(coefficients=(_r(1), _r(0)), constant=_r(0)),
            RationalHyperplane(coefficients=(_r(0), _r(1)), constant=_r(1)),
        ),
    )
    result = compute_arrangement(request)
    assert result.is_central is False


# --- Issue 1: characteristic polynomial must be monic and correct ---


def test_characteristic_polynomial_generic() -> None:
    request = CharacteristicPolynomialRequest(ambient_dimension=2, hyperplane_count=2)
    result = compute_characteristic_polynomial(request)
    assert result.degree == 2
    assert len(result.coefficients) == 3


def test_characteristic_polynomial_is_monic() -> None:
    """chi(t) must always be monic of degree n."""
    for n, m in [(1, 1), (2, 2), (2, 3), (3, 4), (3, 2), (4, 6)]:
        request = CharacteristicPolynomialRequest(
            ambient_dimension=n, hyperplane_count=m
        )
        result = compute_characteristic_polynomial(request)
        assert result.degree == n
        assert result.coefficients[-1] == "1", (
            f"leading coefficient must be 1 for n={n}, m={m}"
        )


def test_characteristic_polynomial_n2_m2() -> None:
    """n=2, m=2: chi(t) = t^2 - 2t + 1 (not 4t^2 - 2t + 1)."""
    request = CharacteristicPolynomialRequest(ambient_dimension=2, hyperplane_count=2)
    result = compute_characteristic_polynomial(request)
    assert result.coefficients == ("1", "-2", "1")


# --- Issue 2: chamber count must use central formula ---


def test_chamber_count_generic() -> None:
    request = ChamberCountRequest(ambient_dimension=2, hyperplane_count=2)
    result = compute_chamber_count(request)
    assert result.chamber_count == "4"


def test_chamber_count_central_m_gt_n() -> None:
    """n=2, m=3: 6 regions (not 7)."""
    request = ChamberCountRequest(ambient_dimension=2, hyperplane_count=3)
    result = compute_chamber_count(request)
    assert result.chamber_count == "6"


def test_chamber_count_zaslavsky_consistency() -> None:
    """regions = (-1)^n * chi(-1) must hold for several (n, m) pairs."""
    for n, m in [(1, 1), (2, 2), (2, 3), (3, 4), (3, 5), (4, 6)]:
        chi_result = compute_characteristic_polynomial(
            CharacteristicPolynomialRequest(ambient_dimension=n, hyperplane_count=m)
        )
        count_result = compute_chamber_count(
            ChamberCountRequest(ambient_dimension=n, hyperplane_count=m)
        )
        coeffs = [int(c) for c in chi_result.coefficients]
        chi_neg1 = sum(v * (-1) ** i for i, v in enumerate(coeffs))
        zaslavsky = (-1) ** n * chi_neg1
        assert zaslavsky == int(count_result.chamber_count), (
            f"Zaslavsky mismatch for n={n}, m={m}: "
            f"{zaslavsky} != {count_result.chamber_count}"
        )


def test_formula_operations_exceed_materialized_arrangement_limits() -> None:
    characteristic = compute_characteristic_polynomial(
        CharacteristicPolynomialRequest(
            ambient_dimension=64,
            hyperplane_count=128,
        )
    )
    chambers = compute_chamber_count(
        ChamberCountRequest(ambient_dimension=64, hyperplane_count=128)
    )

    assert characteristic.degree == 64
    assert len(characteristic.coefficients) == 65
    assert characteristic.coefficients[-1] == "1"
    chi_at_negative_one = sum(
        int(coefficient) * (-1) ** exponent
        for exponent, coefficient in enumerate(characteristic.coefficients)
    )
    assert int(chambers.chamber_count) == chi_at_negative_one


def test_scalar_chamber_count_has_an_independent_envelope() -> None:
    result = compute_chamber_count(
        ChamberCountRequest(
            ambient_dimension=MAX_GENERIC_FORMULA_INDEX,
            hyperplane_count=10,
        )
    )

    assert result.chamber_count == "1024"


def test_characteristic_coefficients_match_direct_formula() -> None:
    from math import comb

    n, m = 32, 47
    result = compute_characteristic_polynomial(
        CharacteristicPolynomialRequest(
            ambient_dimension=n,
            hyperplane_count=m,
        )
    )
    inner = tuple((-1) ** k * comb(m - 1, k) for k in range(n))
    expected_descending = (
        inner[0],
        *(inner[index] - inner[index - 1] for index in range(1, n)),
        -inner[-1],
    )
    assert tuple(map(int, reversed(result.coefficients))) == expected_descending


def test_formula_request_rejects_index_above_schema_envelope() -> None:
    with pytest.raises(ValidationError):
        ChamberCountRequest(
            ambient_dimension=MAX_GENERIC_FORMULA_INDEX + 1,
            hyperplane_count=1,
        )


def test_characteristic_rejects_coefficient_work_before_construction() -> None:
    request = CharacteristicPolynomialRequest(
        ambient_dimension=MAX_GENERIC_FORMULA_WORK + 1,
        hyperplane_count=1,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compute_characteristic_polynomial(request)
    assert (
        error.value.errors()[0]["type"]
        == "hyperplane_arrangement.characteristic_coefficient_work_exceeded"
    )


def test_characteristic_accounts_for_zero_coefficient_tail() -> None:
    result = compute_characteristic_polynomial(
        CharacteristicPolynomialRequest(
            ambient_dimension=5_000,
            hyperplane_count=10,
        )
    )
    assert len(result.coefficients) == 5_001
    assert sum(coefficient != "0" for coefficient in result.coefficients) == 11


def test_characteristic_rejects_oversized_coefficient_formatting() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        compute_characteristic_polynomial(
            CharacteristicPolynomialRequest(
                ambient_dimension=800,
                hyperplane_count=10**15,
            )
        )
    assert (
        error.value.errors()[0]["type"]
        == "hyperplane_arrangement.characteristic_formatting_work_exceeded"
    )


def test_characteristic_rejects_aggregate_output_before_construction() -> None:
    request = CharacteristicPolynomialRequest(
        ambient_dimension=MAX_GENERIC_FORMULA_WORK,
        hyperplane_count=MAX_GENERIC_FORMULA_WORK,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compute_characteristic_polynomial(request)
    assert (
        error.value.errors()[0]["type"]
        == "hyperplane_arrangement.characteristic_result_bytes_exceeded"
    )


def test_chamber_count_uses_the_requested_binomial_prefix() -> None:
    request = ChamberCountRequest(ambient_dimension=1, hyperplane_count=200_000)
    assert compute_chamber_count(request).chamber_count == "2"


def test_characteristic_uses_the_requested_binomial_prefix() -> None:
    result = compute_characteristic_polynomial(
        CharacteristicPolynomialRequest(ambient_dimension=1, hyperplane_count=200_000)
    )
    assert result.coefficients == ("-1", "1")


def test_chamber_count_rejects_actual_output_bytes() -> None:
    request = ChamberCountRequest(
        ambient_dimension=40_000_000,
        hyperplane_count=40_000_000,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compute_chamber_count(request)
    assert (
        error.value.errors()[0]["type"]
        == "hyperplane_arrangement.chamber_result_bytes_exceeded"
    )


def test_chamber_count_closed_form_converts_within_quadratic_work() -> None:
    result = compute_chamber_count(
        ChamberCountRequest(ambient_dimension=8_000, hyperplane_count=8_000)
    )
    assert result.chamber_count == format_canonical_integer(1 << 8_000)


def test_chamber_count_rejects_quadratic_closed_form_conversion_work() -> None:
    # n = m = 10_000 yields ~3011 digits. A constant-cost chunk count is 670,
    # under the 100_000 work ledger; cumulative limb widths are 112_560.
    request = ChamberCountRequest(
        ambient_dimension=10_000,
        hyperplane_count=10_000,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compute_chamber_count(request)
    assert (
        error.value.errors()[0]["type"]
        == "hyperplane_arrangement.chamber_formatting_work_exceeded"
    )


def test_chamber_count_rejects_closed_form_integer_conversion_work() -> None:
    request = ChamberCountRequest(
        ambient_dimension=30_000_000,
        hyperplane_count=30_000_000,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compute_chamber_count(request)
    assert (
        error.value.errors()[0]["type"]
        == "hyperplane_arrangement.chamber_formatting_work_exceeded"
    )


def test_chamber_count_rejects_long_partial_sum() -> None:
    request = ChamberCountRequest(
        ambient_dimension=MAX_GENERIC_FORMULA_WORK + 1,
        hyperplane_count=MAX_GENERIC_FORMULA_WORK + 2,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compute_chamber_count(request)
    assert (
        error.value.errors()[0]["type"]
        == "hyperplane_arrangement.chamber_summation_work_exceeded"
    )


def test_chamber_count_rejects_recurrence_limb_work() -> None:
    # n = 8500, m = 8501 passes the digit and conversion guards (~90,010
    # units) but the binomial recurrence walks C(8500, 4250). Max-width
    # mul/div limb visits are about 4.8 million, over the 100,000 ledger.
    request = ChamberCountRequest(
        ambient_dimension=8_500,
        hyperplane_count=8_501,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compute_chamber_count(request)
    assert (
        error.value.errors()[0]["type"]
        == "hyperplane_arrangement.chamber_summation_work_exceeded"
    )


def test_result_models_reject_contradictory_exact_values() -> None:
    with pytest.raises(ValidationError) as polynomial_error:
        CharacteristicPolynomialResult(coefficients=("1",), degree=1)
    assert (
        polynomial_error.value.errors()[0]["type"]
        == "hyperplane_arrangement.coefficient_count_mismatch"
    )

    with pytest.raises(ValidationError) as chamber_error:
        ChamberCountResult(chamber_count="0")
    assert (
        chamber_error.value.errors()[0]["type"]
        == "hyperplane_arrangement.chamber_count_nonpositive"
    )


# --- Issue 3: validate hyperplane inputs ---


def test_rational_hyperplane_valid() -> None:
    hp = RationalHyperplane(coefficients=(_r(1), _r(0)), constant=_r(0))
    assert hp.coefficients == (_r(1), _r(0))
    assert hp.constant == _r(0)


def test_rational_hyperplane_rejects_non_rational() -> None:
    with pytest.raises(ValidationError):
        RationalHyperplane(
            coefficients=("sqrt(2)", _r(0)),  # type: ignore[arg-type]
            constant=_r(0),
        )


def test_rational_hyperplane_rejects_all_zero_coefficients() -> None:
    with pytest.raises(ValidationError):
        RationalHyperplane(coefficients=(_r(0), _r(0)), constant=_r(0))


def test_rational_hyperplane_rejects_non_rational_constant() -> None:
    with pytest.raises(ValidationError):
        RationalHyperplane(
            coefficients=(_r(1), _r(0)),
            constant="abc",  # type: ignore[arg-type]
        )


def test_rational_hyperplane_accepts_negative_rationals() -> None:
    hp = RationalHyperplane(coefficients=(_r(-1, 2), _r(3, 4)), constant=_r(5, 6))
    assert hp.coefficients == (_r(-1, 2), _r(3, 4))
    assert hp.constant == _r(5, 6)
