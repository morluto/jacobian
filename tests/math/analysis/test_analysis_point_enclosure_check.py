from __future__ import annotations

from fractions import Fraction
from importlib import import_module
from typing import Any, cast

import pytest
from pydantic import ValidationError
from tests.math.analysis._analysis_support import analysis_validation_error

from jacobian.math.analysis._models import ExactDyadic
from jacobian.math.analysis._point_enclosure import (
    MAX_POINT_CHECK_DYADIC_EXPONENT,
    MAX_POINT_CHECK_FRACTION_BITS,
    MAX_POINT_CHECK_LOG_TERMS,
    MAX_POINT_CHECK_OUTPUT_BYTES,
    ArbPointEnclosureRequest,
    ArbPointEnclosureResult,
    ClaimedPointEnclosure,
    PointEnclosureCheckRequest,
    PointEnclosureCheckResult,
    _check_point_enclosure,
    _point_check_fraction_bound_bits,
    _point_enclosure,
)
from jacobian.math.analysis._point_enclosure_check import (
    _log_range_reduction,
    _positive_atanh_enclosures,
)

_LOG_137_80_LOWER = ExactDyadic(
    mantissa="183056359489241580409096213252691826307",
    exponent=-128,
)
_LOG_137_80_UPPER = ExactDyadic(
    mantissa="183056359489241580409096213252691826313",
    exponent=-128,
)


def _claim(
    function: str,
    numerator: str,
    denominator: str,
    lower: ExactDyadic,
    upper: ExactDyadic,
    *,
    precision_bits: int = 128,
) -> dict[str, Any]:
    return {
        "function": function,
        "argument": {"num": numerator, "den": denominator},
        "precision_bits": precision_bits,
        "lower": lower.model_dump(mode="json"),
        "upper": upper.model_dump(mode="json"),
    }


def _request(
    function: str,
    numerator: str,
    denominator: str,
    lower: ExactDyadic,
    upper: ExactDyadic,
    *,
    precision_bits: int = 128,
) -> PointEnclosureCheckRequest:
    return PointEnclosureCheckRequest.model_validate(
        {
            "enclosure": _claim(
                function,
                numerator,
                denominator,
                lower,
                upper,
                precision_bits=precision_bits,
            )
        }
    )


def _run(
    function: str,
    numerator: str,
    denominator: str,
    lower: ExactDyadic,
    upper: ExactDyadic,
    *,
    precision_bits: int = 128,
) -> PointEnclosureCheckResult:
    return _check_point_enclosure(
        _request(
            function,
            numerator,
            denominator,
            lower,
            upper,
            precision_bits=precision_bits,
        )
    )


def _integer_dyadic(value: int) -> ExactDyadic:
    return _normalized_dyadic(value, 0)


def _normalized_dyadic(mantissa: int, exponent: int) -> ExactDyadic:
    if mantissa == 0:
        return ExactDyadic(mantissa="0", exponent=0)
    while mantissa % 2 == 0:
        mantissa //= 2
        exponent += 1
    return ExactDyadic(mantissa=str(mantissa), exponent=exponent)


def test_log_137_80_accepts_the_source_bound_arb_enclosure() -> None:
    result = _run(
        "LOG",
        "137",
        "80",
        _LOG_137_80_LOWER,
        _LOG_137_80_UPPER,
    )

    assert result.outcome == "ACCEPTED"
    assert result.enclosure.function == "LOG"
    assert result.enclosure.argument.as_fraction() == Fraction(137, 80)
    assert result.enclosure.lower == _LOG_137_80_LOWER
    assert result.enclosure.upper == _LOG_137_80_UPPER
    assert (
        PointEnclosureCheckResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_producer_enclosure_crosses_the_checker_boundary_unchanged() -> None:
    producer_result = _point_enclosure(
        ArbPointEnclosureRequest.model_validate(
            {
                "function": "LOG",
                "argument": {"num": "137", "den": "80"},
                "precision_bits": 128,
            }
        )
    )
    serialized = producer_result.model_dump(mode="json")
    assert serialized["status"] == "ENCLOSED"

    request = PointEnclosureCheckRequest.model_validate(
        {"enclosure": serialized["enclosure"]}
    )
    outcome = _check_point_enclosure(request)

    assert producer_result.enclosure is not None
    assert request.enclosure == producer_result.enclosure
    assert outcome.outcome == "ACCEPTED"
    assert outcome.enclosure == producer_result.enclosure
    assert (
        ArbPointEnclosureResult.model_validate_json(producer_result.model_dump_json())
        == producer_result
    )


def test_nonenclosure_outcomes_retain_the_request_source() -> None:
    nonfinite = _point_enclosure(
        ArbPointEnclosureRequest.model_validate(
            {
                "function": "LOG",
                "argument": {"num": "-1", "den": "1"},
                "precision_bits": 128,
            }
        )
    )
    exceeded = _point_enclosure(
        ArbPointEnclosureRequest.model_validate(
            {
                "function": "EXP",
                "argument": {"num": "1" + "0" * 17, "den": "1"},
                "precision_bits": 32,
            }
        )
    )

    assert nonfinite.status == "NONFINITE"
    assert exceeded.status == "OUTPUT_MAGNITUDE_EXCEEDED"
    for result in (nonfinite, exceeded):
        assert result.enclosure is None
        assert result.relative_accuracy_bits is None
        assert not result.exact

    assert nonfinite.function == "LOG"
    assert nonfinite.argument.as_fraction() == Fraction(-1, 1)
    assert nonfinite.precision_bits == 128
    assert exceeded.function == "EXP"
    assert exceeded.argument.as_fraction() == Fraction(10**17, 1)
    assert exceeded.precision_bits == 32

    serialized = [result.model_dump(mode="json") for result in (nonfinite, exceeded)]
    assert len({repr(payload) for payload in serialized}) == 2
    replayed = [
        ArbPointEnclosureResult.model_validate_json(result.model_dump_json())
        for result in (nonfinite, exceeded)
    ]
    assert [payload.model_dump(mode="json") for payload in replayed] == serialized


def test_enclosed_result_must_restate_the_retained_request_source() -> None:
    producer_result = _point_enclosure(
        ArbPointEnclosureRequest.model_validate(
            {
                "function": "SQRT",
                "argument": {"num": "2", "den": "1"},
                "precision_bits": 128,
            }
        )
    )
    assert producer_result.status == "ENCLOSED"
    serialized = producer_result.model_dump(mode="json")

    mismatched_function = {**serialized, "function": "LOG"}
    with analysis_validation_error():
        ArbPointEnclosureResult.model_validate(mismatched_function)

    mismatched_argument = {**serialized, "argument": {"num": "3", "den": "1"}}
    with analysis_validation_error():
        ArbPointEnclosureResult.model_validate(mismatched_argument)

    mismatched_precision = {**serialized, "precision_bits": 256}
    with analysis_validation_error():
        ArbPointEnclosureResult.model_validate(mismatched_precision)

    dropped_source = {
        key: value
        for key, value in serialized.items()
        if key not in {"function", "argument", "precision_bits"}
    }
    with pytest.raises(ValidationError):
        ArbPointEnclosureResult.model_validate(dropped_source)

    forged_status = {**serialized, "status": "NONFINITE"}
    with analysis_validation_error():
        ArbPointEnclosureResult.model_validate(forged_status)


def test_checker_rejects_unsupported_claim_functions_at_admission() -> None:
    claim = _claim("EXP", "1", "1", _integer_dyadic(2), _integer_dyadic(3))

    with analysis_validation_error():
        PointEnclosureCheckRequest.model_validate({"enclosure": claim})


@pytest.mark.parametrize(
    ("numerator", "denominator", "lower", "upper"),
    (
        ("1", "1", 0, 0),
        ("1", "2", -1, 0),
        ("8", "1", 2, 3),
        ("1", "8", -3, -2),
    ),
)
def test_log_range_reduction_handles_one_below_one_and_powers_of_two(
    numerator: str,
    denominator: str,
    lower: int,
    upper: int,
) -> None:
    assert (
        _run(
            "LOG",
            numerator,
            denominator,
            _integer_dyadic(lower),
            _integer_dyadic(upper),
        ).outcome
        == "ACCEPTED"
    )


@pytest.mark.parametrize(
    ("numerator", "lower", "upper"),
    (
        ("0", 0, 0),
        ("4", 2, 2),
        ("2", 1, 2),
    ),
)
def test_sqrt_accepts_zero_rational_and_irrational_roots(
    numerator: str,
    lower: int,
    upper: int,
) -> None:
    assert (
        _run(
            "SQRT",
            numerator,
            "1",
            _integer_dyadic(lower),
            _integer_dyadic(upper),
        ).outcome
        == "ACCEPTED"
    )


@pytest.mark.parametrize("function", ("LOG", "SQRT"))
def test_real_domain_failures_are_typed_rejections(function: str) -> None:
    result = _run(
        function,
        "-1",
        "1",
        _integer_dyadic(-1),
        _integer_dyadic(1),
    )

    assert result.outcome == "REJECTED"


def test_reversed_and_provably_excluding_intervals_are_typed_rejections() -> None:
    reversed_result = _run("SQRT", "2", "1", _integer_dyadic(2), _integer_dyadic(1))
    excluded_sqrt = _run("SQRT", "2", "1", _integer_dyadic(0), _integer_dyadic(1))
    excluded_log = _run(
        "LOG",
        "137",
        "80",
        _integer_dyadic(0),
        ExactDyadic(mantissa="1", exponent=-1),
    )

    assert reversed_result.outcome == "REJECTED"
    assert excluded_sqrt.outcome == "REJECTED"
    assert excluded_log.outcome == "REJECTED"


def test_log_overlapping_claim_at_the_series_cap_is_a_non_result() -> None:
    near_log_two = ExactDyadic(
        mantissa=(
            "9293584264128987901384440660653081117630633404975079641076009770"
            "5783645736330756451676239180675129869308430405969526366636736757"
            "98576024949780480621393957"
        ),
        exponent=-512,
    )

    accepted = _run("LOG", "2", "1", _integer_dyadic(0), _integer_dyadic(1))
    rejected = _run(
        "LOG",
        "2",
        "1",
        _integer_dyadic(0),
        ExactDyadic(mantissa="1", exponent=-1),
    )
    unresolved = _run("LOG", "2", "1", near_log_two, near_log_two, precision_bits=512)

    assert accepted.outcome == "ACCEPTED"
    assert rejected.outcome == "REJECTED"
    assert unresolved.outcome == "NON_RESULT"


def test_positive_atanh_tail_bound_is_exactly_the_documented_geometric_bound() -> None:
    z = Fraction(1, 3)
    lower, upper = tuple(_positive_atanh_enclosures(z))[-1]
    first_omitted_power = z ** (2 * MAX_POINT_CHECK_LOG_TERMS + 1)
    expected_tail = (
        2 * first_omitted_power / (2 * MAX_POINT_CHECK_LOG_TERMS + 1) / (1 - z * z)
    )

    assert upper - lower == expected_tail


def test_fraction_intermediates_fit_the_preflighted_bit_bound_at_the_source_limit() -> (
    None
):
    denominator = 10**127
    argument = Fraction(denominator - 1, denominator)
    exponent, reduced = _log_range_reduction(argument)
    assert exponent == -1
    z = (reduced - 1) / (reduced + 1)
    reduced_lower, reduced_upper = tuple(_positive_atanh_enclosures(z))[-1]
    log_two_lower, log_two_upper = tuple(_positive_atanh_enclosures(Fraction(1, 3)))[-1]
    lower = reduced_lower - log_two_upper
    upper = reduced_upper - log_two_lower

    endpoint_mantissa = int("9" * 1_235)
    large_endpoint = Fraction(endpoint_mantissa << MAX_POINT_CHECK_DYADIC_EXPONENT)
    small_endpoint = Fraction(endpoint_mantissa, 1 << MAX_POINT_CHECK_DYADIC_EXPONENT)
    comparison_products = (
        large_endpoint.numerator * lower.denominator,
        lower.numerator * large_endpoint.denominator,
        small_endpoint.numerator * upper.denominator,
        upper.numerator * small_endpoint.denominator,
    )

    assert (
        max(
            lower.numerator.bit_length(),
            lower.denominator.bit_length(),
            upper.numerator.bit_length(),
            upper.denominator.bit_length(),
            *(abs(value).bit_length() for value in comparison_products),
        )
        <= MAX_POINT_CHECK_FRACTION_BITS
    )
    assert (
        _point_check_fraction_bound_bits(
            _request(
                "LOG",
                str(denominator - 1),
                str(denominator),
                _integer_dyadic(-1),
                _integer_dyadic(0),
            ).enclosure.argument
        )
        <= MAX_POINT_CHECK_FRACTION_BITS
    )


def test_result_parsing_is_structural_and_checker_verifies_claims() -> None:
    result = _run(
        "LOG",
        "137",
        "80",
        _LOG_137_80_LOWER,
        _LOG_137_80_UPPER,
    )

    forged_verdict = result.model_dump(mode="json")
    forged_verdict["outcome"] = "REJECTED"
    parsed_forged_verdict = PointEnclosureCheckResult.model_validate(forged_verdict)
    assert parsed_forged_verdict.outcome == "REJECTED"
    assert (
        _check_point_enclosure(
            PointEnclosureCheckRequest(enclosure=parsed_forged_verdict.enclosure)
        ).outcome
        == "ACCEPTED"
    )

    tampered_interval = result.model_dump(mode="json")
    tampered_interval["enclosure"]["upper"] = {"mantissa": "1", "exponent": -1}
    parsed_interval = PointEnclosureCheckResult.model_validate(tampered_interval)
    assert (
        _check_point_enclosure(
            PointEnclosureCheckRequest(enclosure=parsed_interval.enclosure)
        ).outcome
        == "REJECTED"
    )

    wrong_function = result.model_dump(mode="json")
    wrong_function["enclosure"]["function"] = "SQRT"
    parsed_function = PointEnclosureCheckResult.model_validate(wrong_function)
    assert (
        _check_point_enclosure(
            PointEnclosureCheckRequest(enclosure=parsed_function.enclosure)
        ).outcome
        == "REJECTED"
    )

    wrong_argument = result.model_dump(mode="json")
    wrong_argument["enclosure"]["argument"] = {"num": "0", "den": "1"}
    parsed_argument = PointEnclosureCheckResult.model_validate(wrong_argument)
    assert (
        _check_point_enclosure(
            PointEnclosureCheckRequest(enclosure=parsed_argument.enclosure)
        ).outcome
        == "REJECTED"
    )

    oversized_source = result.model_dump(mode="json")
    oversized_source["enclosure"]["argument"] = {"num": "1" + "0" * 128, "den": "1"}
    with analysis_validation_error():
        PointEnclosureCheckResult.model_validate(oversized_source)


def test_request_accepts_exact_structural_bounds_and_result_fits_output_budget() -> (
    None
):
    largest_positive_mantissa = "9" * 1_235
    largest_negative_mantissa = "-" + "9" * 1_234
    result = _run(
        "SQRT",
        "1" + "0" * 127,
        "1",
        ExactDyadic(
            mantissa=largest_negative_mantissa,
            exponent=MAX_POINT_CHECK_DYADIC_EXPONENT,
        ),
        ExactDyadic(
            mantissa=largest_positive_mantissa,
            exponent=MAX_POINT_CHECK_DYADIC_EXPONENT,
        ),
        precision_bits=4096,
    )

    assert result.outcome == "ACCEPTED"
    assert len(result.model_dump_json().encode("utf-8")) < MAX_POINT_CHECK_OUTPUT_BYTES


def test_request_rejects_values_immediately_over_each_structural_bound() -> None:
    with analysis_validation_error():
        _request(
            "SQRT",
            "1" + "0" * 128,
            "1",
            _integer_dyadic(0),
            _integer_dyadic(1),
        )
    with analysis_validation_error():
        _request(
            "SQRT",
            "1",
            "1",
            _integer_dyadic(0),
            ExactDyadic(mantissa="1", exponent=MAX_POINT_CHECK_DYADIC_EXPONENT + 1),
        )
    with analysis_validation_error():
        _request(
            "SQRT",
            "1",
            "1",
            _integer_dyadic(0),
            _integer_dyadic(1),
            precision_bits=4097,
        )
    with analysis_validation_error():
        ExactDyadic(mantissa="9" * 1_236, exponent=0)


@pytest.mark.parametrize(
    ("function", "numerator", "denominator"),
    (
        ("LOG", 137, 80),
        ("LOG", 1, 3),
        ("LOG", 8, 1),
        ("SQRT", 2, 1),
        ("SQRT", 17, 5),
    ),
)
def test_checker_agrees_with_high_precision_mpmath_values(
    function: str,
    numerator: int,
    denominator: int,
) -> None:
    mp = cast(Any, import_module("mpmath").mp)

    dyadic_bits = 256
    with mp.workprec(1024):
        argument = mp.mpf(numerator) / denominator
        value = mp.log(argument) if function == "LOG" else mp.sqrt(argument)
        scaled = value * mp.power(2, dyadic_bits)
        lower = _normalized_dyadic(int(mp.floor(scaled)), -dyadic_bits)
        upper = _normalized_dyadic(int(mp.ceil(scaled)), -dyadic_bits)

        result = _run(
            function,
            str(numerator),
            str(denominator),
            lower,
            upper,
            precision_bits=dyadic_bits,
        )

        assert result.outcome == "ACCEPTED"
        assert mp.mpf(lower.mantissa) * mp.power(2, lower.exponent) <= value
        assert value <= mp.mpf(upper.mantissa) * mp.power(2, upper.exponent)


def test_source_payload_helper_preserves_json_types() -> None:
    request = _request("SQRT", "0", "1", _integer_dyadic(0), _integer_dyadic(0))
    payload: dict[str, Any] = request.model_dump(mode="json")

    assert payload == {
        "enclosure": {
            "function": "SQRT",
            "argument": {"num": "0", "den": "1"},
            "precision_bits": 128,
            "lower": {"mantissa": "0", "exponent": 0},
            "upper": {"mantissa": "0", "exponent": 0},
        }
    }


def test_request_schema_publishes_work_and_precision_limits() -> None:
    schema = PointEnclosureCheckRequest.model_json_schema()
    enclosure_ref = schema["properties"]["enclosure"]["$ref"]
    enclosure_name = enclosure_ref.rsplit("/", 1)[-1]
    enclosure_schema = schema["$defs"][enclosure_name]

    assert schema["point_check_log_term_bound"] == MAX_POINT_CHECK_LOG_TERMS
    assert (
        schema["point_check_fraction_intermediate_bit_bound"]
        == MAX_POINT_CHECK_FRACTION_BITS
    )
    assert schema["point_check_output_byte_bound"] == MAX_POINT_CHECK_OUTPUT_BYTES
    precision_description = enclosure_schema["properties"]["precision_bits"][
        "description"
    ]
    assert "does not promise" in precision_description
    precision_description_schema = ClaimedPointEnclosure.model_json_schema()[
        "properties"
    ]["precision_bits"]["description"]
    assert precision_description == precision_description_schema
    enclosure_description = schema["properties"]["enclosure"]["description"]
    assert "only LOG and SQRT" in enclosure_description
    assert (
        PointEnclosureCheckResult.model_json_schema()["point_check_output_byte_bound"]
        == MAX_POINT_CHECK_OUTPUT_BYTES
    )
