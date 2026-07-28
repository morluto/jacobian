"""Independent exact rational replay for projective line arrangements."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(detail: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_RATIONAL",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": detail,
    }


def _rational(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("malformed rational")
    numerator = value["num"]
    denominator = value["den"]
    if not isinstance(numerator, str) or not isinstance(denominator, str):
        raise ValueError("malformed rational")
    fraction = Fraction(int(numerator), int(denominator))
    if numerator != str(fraction.numerator) or denominator != str(fraction.denominator):
        raise ValueError("noncanonical rational")
    return fraction


def _primitive(
    values: tuple[Fraction, Fraction, Fraction],
) -> tuple[int, int, int]:
    denominator_lcm = 1
    for value in values:
        denominator_lcm = (
            denominator_lcm
            * value.denominator
            // gcd(denominator_lcm, value.denominator)
        )
    integers = tuple(
        value.numerator * (denominator_lcm // value.denominator) for value in values
    )
    divisor = 0
    for integer in integers:
        divisor = gcd(divisor, abs(integer))
    if divisor == 0:
        raise ValueError("zero projective coordinates")
    normalized = tuple(integer // divisor for integer in integers)
    if next(integer for integer in normalized if integer) < 0:
        normalized = tuple(-integer for integer in normalized)
    return normalized[0], normalized[1], normalized[2]


def _cross(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
) -> tuple[int, int, int]:
    return _primitive(
        (
            Fraction(left[1] * right[2] - left[2] * right[1]),
            Fraction(left[2] * right[0] - left[0] * right[2]),
            Fraction(left[0] * right[1] - left[1] * right[0]),
        )
    )


def _triple(values: tuple[int, int, int]) -> dict[str, list[str]]:
    return {"coordinates": [str(values[0]), str(values[1]), str(values[2])]}


def _expected(source: dict[str, Any]) -> dict[str, Any]:
    if set(source) != {"arrangement_schema_version", "lines"}:
        raise ValueError("malformed arrangement request")
    lines = source["lines"]
    if (
        source["arrangement_schema_version"] != "1"
        or not isinstance(lines, list)
        or not 2 <= len(lines) <= 64
    ):
        raise ValueError("arrangement lies outside checker scope")
    normalized: list[tuple[str, tuple[int, int, int]]] = []
    for line in lines:
        if not isinstance(line, dict) or set(line) != {"label", "coefficients"}:
            raise ValueError("malformed projective line")
        label = line["label"]
        coefficients = line["coefficients"]
        if (
            not isinstance(label, str)
            or not isinstance(coefficients, list)
            or len(coefficients) != 3
        ):
            raise ValueError("malformed projective line")
        normalized.append(
            (
                label,
                _primitive(
                    (
                        _rational(coefficients[0]),
                        _rational(coefficients[1]),
                        _rational(coefficients[2]),
                    )
                ),
            )
        )
    normalized.sort()
    if len({label for label, _ in normalized}) != len(normalized) or len(
        {coefficients for _, coefficients in normalized}
    ) != len(normalized):
        raise ValueError("duplicate labels or projective lines")
    points = {
        _cross(normalized[left][1], normalized[right][1])
        for left in range(len(normalized))
        for right in range(left + 1, len(normalized))
    }
    flats: list[dict[str, Any]] = []
    histogram: dict[int, int] = {}
    for point in sorted(points):
        incident = [
            label
            for label, coefficients in normalized
            if sum(
                coefficient * coordinate
                for coefficient, coordinate in zip(
                    coefficients,
                    point,
                    strict=True,
                )
            )
            == 0
        ]
        multiplicity = len(incident)
        if multiplicity < 2:
            raise ValueError("intersection lost its defining lines")
        histogram[multiplicity] = histogram.get(multiplicity, 0) + 1
        flats.append(
            {
                "point": _triple(point),
                "incident_labels": incident,
                "multiplicity": multiplicity,
                "pair_count": multiplicity * (multiplicity - 1) // 2,
            }
        )
    line_count = len(normalized)
    return {
        "result_schema_version": "1",
        "line_count": line_count,
        "normalized_lines": [
            {
                "label": label,
                "coefficients": _triple(coefficients),
            }
            for label, coefficients in normalized
        ],
        "flats": flats,
        "non_double_flats": sorted(
            flat["incident_labels"] for flat in flats if flat["multiplicity"] > 2
        ),
        "multiplicity_histogram": [
            {"multiplicity": multiplicity, "flat_count": count}
            for multiplicity, count in sorted(histogram.items())
        ],
        "pair_count_total": line_count * (line_count - 1) // 2,
        "completion": "COMPLETE",
        "arithmetic": "EXACT_INTEGER",
        "verification_capability_id": (
            "geometry.projective_line_arrangement.flats.verify"
        ),
        "verification_input_field": "result_uri",
    }


def check_projective_line_arrangement_flats(
    request: dict[str, Any],
) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id="geometry.projective_line_arrangement.flats.materialize",
            witness_format=(
                "geometry.projective-line-arrangement.flats.exhaustive-replay"
            ),
        )
        if result != _expected(source):
            return _reject(
                "stored flats do not match independent exact projective replay"
            )
        return _accept(
            "independent exact projective replay accepted every line pair, flat "
            "incidence and multiplicity"
        )
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return _reject("malformed, unsupported, or mismatched checker request")


__all__ = ["check_projective_line_arrangement_flats"]
