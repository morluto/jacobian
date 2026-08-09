"""Independent exact replay for recurrences and rational series.

This checker imports neither SymPy nor producer modules. Only passive,
artifact-bound JSON values cross the checker boundary.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from fractions import Fraction
from typing import Any

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian_checkers.bound_artifacts import bound_request as _bound_request

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_RECURRENCE_CONVENTION = "A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"
_P_RECURSIVE_CONVENTION = "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
_SERIES_CONVENTION = "ASCENDING_POWERS_OF_X"
_RESIDUAL_CONGRUENCE = "DENOMINATOR_TIMES_SERIES_MINUS_NUMERATOR_IS_ZERO_MOD_X_TO_ORDER"
_META = {
    "exactness": "EXACT_RATIONAL",
    "determinism": "DETERMINISTIC",
    "backend": "sympy",
    "backend_version": "1.14.0",
    "verification": "UNVERIFIED",
}
_MAX_RESULT_DIGITS = 32_768
_MAX_RESULT_BYTES = 10 * 1024 * 1024
_LOG10_2 = math.log10(2)


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(operation_id: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": f"independent standard-library Fraction replay accepted {operation_id}",
    }


def _integer(value: object, *, maximum: int | None = None) -> int:
    if not isinstance(value, str) or _INTEGER.fullmatch(value) is None:
        raise ValueError("integer is not canonical")
    if maximum is not None and len(value.lstrip("-")) > len(str(maximum)):
        raise ValueError("integer is outside the checker scope")
    parsed = parse_canonical_integer(value)
    if format_canonical_integer(parsed) != value or (
        maximum is not None and abs(parsed) > maximum
    ):
        raise ValueError("integer is outside the checker scope")
    return parsed


def _fraction(value: object, *, max_digits: int) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational is malformed")
    if (
        not isinstance(value["num"], str)
        or not isinstance(value["den"], str)
        or len(value["num"].lstrip("-")) > max_digits
        or len(value["den"].lstrip("-")) > max_digits
    ):
        raise ValueError("rational exceeds the checker digit scope")
    numerator = _integer(value["num"])
    denominator = _integer(value["den"])
    if denominator <= 0:
        raise ValueError("rational denominator must be positive")
    result = Fraction(numerator, denominator)
    if (result.numerator, result.denominator) != (numerator, denominator):
        raise ValueError("rational is not reduced")
    return result


def _fractions(
    value: object,
    *,
    minimum: int,
    maximum: int,
    max_digits: int,
) -> list[Fraction]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError("rational vector is outside the checker scope")
    return [_fraction(item, max_digits=max_digits) for item in value]


def _metadata(result: dict[str, Any], fields: set[str]) -> bool:
    return set(result) == fields | set(_META) and all(
        result.get(key) == value for key, value in _META.items()
    )


def _run(
    request: object,
    *,
    operation_id: str,
    witness_format: str,
    replay: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> dict[str, Any]:
    try:
        source, result = _bound_request(
            request,
            operation_id=operation_id,
            witness_format=witness_format,
        )
        if not replay(source, result):
            return _reject("declared result does not match independent Fraction replay")
        return _accept(operation_id)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


def _validate_recurrence_scope(
    source: dict[str, Any],
    raw_indices: list[Any],
) -> list[int] | None:
    if source["scope"] == "PREFIX":
        term_count = source["term_count"]
        if type(term_count) is not int or not 1 <= term_count <= 513 or raw_indices:
            return None
        return list(range(term_count))
    if source["scope"] == "INDICES":
        if source["term_count"] is not None or not raw_indices:
            return None
        if any(
            type(index) is not int or not 0 <= index <= 512 for index in raw_indices
        ) or raw_indices != sorted(set(raw_indices)):
            return None
        return raw_indices
    return None


def _validate_recurrence_values(
    result: dict[str, Any],
    requested: list[int],
    replay: list[Fraction],
    end: int,
) -> bool:
    raw_replay = result["replay_prefix"]
    if (
        type(result["replay_scope_end"]) is not int
        or result["replay_scope_end"] != end
        or not isinstance(raw_replay, list)
        or len(raw_replay) != end + 1
        or [_fraction(item, max_digits=32_768) for item in raw_replay] != replay
    ):
        return False
    values = result["values"]
    if not isinstance(values, list) or len(values) != len(requested):
        return False
    for item, index in zip(values, requested, strict=True):
        if (
            not isinstance(item, dict)
            or set(item) != {"index", "value"}
            or type(item["index"]) is not int
            or item["index"] != index
            or _fraction(item["value"], max_digits=32_768) != replay[index]
        ):
            return False
    return True


def _replay_linear_recurrence(
    source: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    if set(source) != {
        "coefficients",
        "initial_values",
        "coefficient_convention",
        "scope",
        "term_count",
        "indices",
    } or not _metadata(
        result,
        {
            "coefficient_convention",
            "scope",
            "values",
            "replay_prefix",
            "replay_scope_end",
        },
    ):
        return False
    if (
        source["coefficient_convention"] != _RECURRENCE_CONVENTION
        or result["coefficient_convention"] != _RECURRENCE_CONVENTION
        or result["scope"] != source["scope"]
    ):
        return False
    coefficients = _fractions(
        source["coefficients"],
        minimum=1,
        maximum=16,
        max_digits=64,
    )
    initial = _fractions(
        source["initial_values"],
        minimum=1,
        maximum=16,
        max_digits=64,
    )
    if len(coefficients) != len(initial):
        return False
    raw_indices = source["indices"]
    if not isinstance(raw_indices, list) or len(raw_indices) > 256:
        return False
    requested = _validate_recurrence_scope(source, raw_indices)
    if requested is None:
        return False
    end = requested[-1]
    replay = initial[: end + 1]
    while len(replay) <= end:
        replay.append(
            sum(
                (
                    coefficient * replay[len(replay) - offset]
                    for offset, coefficient in enumerate(coefficients, start=1)
                ),
                start=Fraction(),
            )
        )
    return _validate_recurrence_values(result, requested, replay, end)


def _canonical_polynomial(value: object) -> list[Fraction]:
    coefficients = _fractions(
        value,
        minimum=1,
        maximum=33,
        max_digits=64,
    )
    if len(coefficients) > 1 and coefficients[-1] == 0:
        raise ValueError("polynomial has a trailing zero coefficient")
    return coefficients


def _p_recursive_metadata_matches(
    source: dict[str, Any], result: dict[str, Any]
) -> bool:
    return set(source) == {
        "coefficient_polynomials",
        "initial_values",
        "coefficient_convention",
        "polynomial_convention",
        "scope",
        "term_count",
        "indices",
    } and _metadata(
        result,
        {
            "coefficient_convention",
            "polynomial_convention",
            "scope",
            "recurrence_order",
            "values",
            "replay_prefix",
            "residuals",
            "replay_scope_end",
        },
    )


def _p_recursive_residuals_match(
    result: dict[str, Any], *, order: int, end: int, residuals: list[Fraction]
) -> bool:
    raw_residuals = result["residuals"]
    if not isinstance(raw_residuals, list) or len(raw_residuals) != max(
        0, end - order + 1
    ):
        return False
    return all(
        isinstance(item, dict)
        and set(item) == {"index", "value"}
        and type(item["index"]) is int
        and item["index"] == index
        and _fraction(item["value"], max_digits=32_768) == residual
        and residual == 0
        for item, index, residual in zip(
            raw_residuals, range(order, end + 1), residuals, strict=True
        )
    )


def _p_recursive_replay(
    polynomials: list[list[Fraction]],
    initial: list[Fraction],
    end: int,
    requested: set[int],
) -> tuple[list[Fraction], list[Fraction]] | None:
    order = len(polynomials) - 1

    def polynomial_value(polynomial: list[Fraction], index: int) -> Fraction:
        return sum(
            (
                coefficient * index**power
                for power, coefficient in enumerate(polynomial)
            ),
            start=Fraction(),
        )

    replay = initial[: end + 1]
    minimum_size = 1_024 + sum(
        _minimum_fraction_bytes(value) * (1 + (index in requested))
        for index, value in enumerate(replay)
    )
    residuals: list[Fraction] = []
    while len(replay) <= end:
        index = len(replay)
        coefficients = [polynomial_value(item, index) for item in polynomials]
        if coefficients[0] == 0:
            return None
        next_value = (
            -sum(
                (
                    coefficients[offset] * replay[index - offset]
                    for offset in range(1, order + 1)
                ),
                start=Fraction(),
            )
            / coefficients[0]
        )
        if not _bounded_replay_fraction(next_value):
            return None
        minimum_size += _minimum_fraction_bytes(next_value) * (1 + (index in requested))
        minimum_size += 32
        if minimum_size > _MAX_RESULT_BYTES:
            return None
        replay.append(next_value)
        residuals.append(
            sum(
                (
                    coefficients[offset] * replay[index - offset]
                    for offset in range(order + 1)
                ),
                start=Fraction(),
            )
        )
    return replay, residuals


def _decimal_digits(value: int) -> int:
    if value == 0:
        return 1
    return math.floor((abs(value).bit_length() - 1) * _LOG10_2) + 1


def _minimum_fraction_bytes(value: Fraction) -> int:
    return _decimal_digits(value.numerator) + _decimal_digits(value.denominator) + 20


def _bounded_replay_fraction(value: Fraction) -> bool:
    return all(
        _decimal_digits(component) <= _MAX_RESULT_DIGITS
        for component in (value.numerator, value.denominator)
    )


def _replay_polynomial_coefficient_recurrence(
    source: dict[str, Any], result: dict[str, Any]
) -> bool:
    if not _p_recursive_metadata_matches(source, result):
        return False
    if (
        source["coefficient_convention"] != _P_RECURSIVE_CONVENTION
        or source["polynomial_convention"] != "ASCENDING_POWERS_OF_N"
        or result["coefficient_convention"] != _P_RECURSIVE_CONVENTION
        or result["polynomial_convention"] != "ASCENDING_POWERS_OF_N"
        or result["scope"] != source["scope"]
    ):
        return False
    raw_polynomials = source["coefficient_polynomials"]
    if not isinstance(raw_polynomials, list) or not 2 <= len(raw_polynomials) <= 17:
        return False
    polynomials = [_canonical_polynomial(item) for item in raw_polynomials]
    if any(len(polynomial) > 17 for polynomial in polynomials):
        return False
    initial = _fractions(source["initial_values"], minimum=1, maximum=16, max_digits=64)
    order = len(polynomials) - 1
    if (
        len(initial) != order
        or type(result["recurrence_order"]) is not int
        or not 1 <= result["recurrence_order"] <= 16
        or result["recurrence_order"] != order
    ):
        return False
    raw_indices = source["indices"]
    if not isinstance(raw_indices, list) or len(raw_indices) > 256:
        return False
    requested = _validate_recurrence_scope(source, raw_indices)
    if requested is None:
        return False
    end = requested[-1]
    replayed = _p_recursive_replay(polynomials, initial, end, set(requested))
    if replayed is None:
        return False
    replay, residuals = replayed
    if not _validate_recurrence_values(result, requested, replay, end):
        return False
    return _p_recursive_residuals_match(
        result, order=order, end=end, residuals=residuals
    )


def _replay_rational_series(
    source: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    if set(source) != {
        "numerator",
        "denominator",
        "coefficient_convention",
        "expansion_point",
        "truncation_order",
    } or not _metadata(
        result,
        {
            "coefficient_convention",
            "expansion_point",
            "truncation_order",
            "coefficients",
            "residual_congruence",
            "residual_coefficients",
        },
    ):
        return False
    order = source["truncation_order"]
    if (
        source["coefficient_convention"] != _SERIES_CONVENTION
        or source["expansion_point"] != "0"
        or type(order) is not int
        or not 1 <= order <= 512
        or result["coefficient_convention"] != _SERIES_CONVENTION
        or result["expansion_point"] != "0"
        or result["truncation_order"] != order
        or result["residual_congruence"] != _RESIDUAL_CONGRUENCE
    ):
        return False
    numerator = _canonical_polynomial(source["numerator"])
    denominator = _canonical_polynomial(source["denominator"])
    if denominator[0] == 0:
        return False
    expected: list[Fraction] = []
    for degree in range(order):
        numerator_coefficient = (
            numerator[degree] if degree < len(numerator) else Fraction()
        )
        known = sum(
            (
                denominator[offset] * expected[degree - offset]
                for offset in range(1, min(degree, len(denominator) - 1) + 1)
            ),
            start=Fraction(),
        )
        expected.append((numerator_coefficient - known) / denominator[0])
    coefficients = _fractions(
        result["coefficients"],
        minimum=order,
        maximum=order,
        max_digits=32_768,
    )
    residuals = _fractions(
        result["residual_coefficients"],
        minimum=order,
        maximum=order,
        max_digits=32_768,
    )
    independently_computed_residuals = [
        sum(
            (
                denominator[offset] * coefficients[degree - offset]
                for offset in range(min(degree, len(denominator) - 1) + 1)
            ),
            start=Fraction(),
        )
        - (numerator[degree] if degree < len(numerator) else Fraction())
        for degree in range(order)
    ]
    return (
        coefficients == expected
        and residuals == independently_computed_residuals
        and all(residual == 0 for residual in residuals)
    )


def check_linear_recurrence_evaluation(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="combinatorics.recurrence.linear.evaluate",
        witness_format="combinatorics.linear-recurrence.fraction-replay",
        replay=_replay_linear_recurrence,
    )


def check_polynomial_coefficient_recurrence_evaluation(
    request: object,
) -> dict[str, Any]:
    return _run(
        request,
        operation_id="combinatorics.recurrence.p_recursive.evaluate",
        witness_format="combinatorics.p-recursive.fraction-residual-replay",
        replay=_replay_polynomial_coefficient_recurrence,
    )


def check_rational_generating_function_coefficients(
    request: object,
) -> dict[str, Any]:
    return _run(
        request,
        operation_id="combinatorics.generating_function.coefficients.compute",
        witness_format="combinatorics.rational-series.fraction-residual-replay",
        replay=_replay_rational_series,
    )


__all__ = [
    "check_linear_recurrence_evaluation",
    "check_polynomial_coefficient_recurrence_evaluation",
    "check_rational_generating_function_coefficients",
]
