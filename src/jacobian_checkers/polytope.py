"""Independent replay for finite rational-polytope evidence."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_MAX_GENERATORS = 100_000
_MAX_DIMENSION = 256


def _reject(
    detail: str,
    *,
    method: str,
) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": method,
        "coverage": (
            "EXHAUSTIVE" if method == "EXHAUSTIVE_FINITE" else "NOT_APPLICABLE"
        ),
        "detail": detail,
    }


def _parse_rational(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational must contain num and den")
    numerator = value["num"]
    denominator = value["den"]
    if (
        not isinstance(numerator, str)
        or not isinstance(denominator, str)
        or _INTEGER.fullmatch(numerator) is None
        or _INTEGER.fullmatch(denominator) is None
    ):
        raise ValueError("noncanonical rational integer")
    parsed = Fraction(int(numerator), int(denominator))
    if str(parsed.numerator) != numerator or str(parsed.denominator) != denominator:
        raise ValueError("rational is not reduced and canonical")
    return parsed


def _point(payload: object) -> tuple[Fraction, ...]:
    if not isinstance(payload, dict):
        raise ValueError("point payload must be an object")
    if payload.get("point_schema_version") != "1":
        raise ValueError("unsupported point version")
    coordinates = payload.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError("point coordinates must be nonempty")
    if len(coordinates) > _MAX_DIMENSION:
        raise ValueError("point exceeds checker dimension limit")
    return tuple(_parse_rational(value) for value in coordinates)


def _generators(payload: object) -> tuple[tuple[Fraction, ...], ...]:
    if not isinstance(payload, dict):
        raise ValueError("generator payload must be an object")
    if payload.get("generator_set_schema_version") != "1":
        raise ValueError("unsupported generator-set version")
    dimension = payload.get("dimension")
    generators = payload.get("generators")
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 1
        or dimension > _MAX_DIMENSION
        or not isinstance(generators, list)
        or not generators
        or len(generators) > _MAX_GENERATORS
    ):
        raise ValueError("invalid finite generator set")
    parsed: list[tuple[Fraction, ...]] = []
    for generator in generators:
        if not isinstance(generator, dict) or set(generator) != {"values"}:
            raise ValueError("generator must contain values")
        values = generator["values"]
        if not isinstance(values, list) or len(values) != dimension:
            raise ValueError("generator dimension mismatch")
        parsed.append(tuple(_parse_rational(value) for value in values))
    return tuple(parsed)


def _claim(
    request: dict[str, Any],
    expected_predicate: str,
) -> tuple[int, str, str]:
    claim = request["claim"]["payload"]
    if not isinstance(claim, dict):
        raise ValueError("claim payload must be an object")
    if claim.get("claim_schema_version") != "1":
        raise ValueError("unsupported claim version")
    if claim.get("predicate") != expected_predicate:
        raise ValueError("unexpected polytope predicate")
    dimension = claim.get("dimension")
    point_uri = claim.get("point_uri")
    generator_set_uri = claim.get("generator_set_uri")
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or not isinstance(point_uri, str)
        or not isinstance(generator_set_uri, str)
    ):
        raise ValueError("malformed polytope claim")
    if (
        point_uri != request["candidate"]["artifact_uri"]
        or generator_set_uri != request["scope"]["artifact_uri"]
    ):
        raise ValueError("claim URIs do not bind the supplied objects")
    return dimension, point_uri, generator_set_uri


def _check_convex_witness_header(
    witness: dict[str, Any],
    expected_bindings: object,
) -> str | None:
    if witness.get("witness_format") != "polytope.convex_combination":
        return "unexpected witness format"
    if witness.get("format_version") != "1":
        return "unsupported witness version"
    if witness.get("role") != "SUPPORTS_CLAIM":
        return "witness does not support the claim"
    if witness.get("bindings") != expected_bindings:
        return "witness bindings do not match"
    return None


def _check_object_dimensions(
    point: tuple[Fraction, ...],
    generators: tuple[tuple[Fraction, ...], ...],
    dimension: int,
) -> str | None:
    if len(point) != dimension or any(
        len(generator) != dimension for generator in generators
    ):
        return "claim dimension does not match objects"
    return None


def _check_convex_weights(
    inner: dict[str, Any],
    generators: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], str | None]:
    raw_weights = inner.get("weights")
    if not isinstance(raw_weights, list) or len(raw_weights) != len(generators):
        return (), "weight count does not match generators"
    weights = tuple(_parse_rational(value) for value in raw_weights)
    if any(weight < 0 for weight in weights) or sum(weights) != 1:
        return (), "weights are not a convex combination"
    return weights, None


def check_convex_combination(request: dict[str, Any]) -> dict[str, Any]:
    """Check exact nonnegative weights reconstruct the bound point."""

    method = "DIRECT_WITNESS"
    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version", method=method)
        dimension, _, _ = _claim(request, "INSIDE_CONVEX_HULL")
        witness = request["witness"]["payload"]
        error = _check_convex_witness_header(witness, request["expected_bindings"])
        if error is not None:
            return _reject(error, method=method)
        point = _point(request["candidate"]["payload"])
        generators = _generators(request["scope"]["payload"])
        error = _check_object_dimensions(point, generators, dimension)
        if error is not None:
            return _reject(error, method=method)
        inner = witness.get("payload")
        if not isinstance(inner, dict):
            return _reject("witness payload is missing", method=method)
        weights, error = _check_convex_weights(inner, generators)
        if error is not None:
            return _reject(error, method=method)
        reconstructed = tuple(
            sum(
                weight * generator[index]
                for weight, generator in zip(
                    weights,
                    generators,
                    strict=True,
                )
            )
            for index in range(dimension)
        )
        if reconstructed != point:
            return _reject("weights do not reconstruct the point", method=method)
        declared = inner.get("reconstructed_point")
        if (
            not isinstance(declared, list)
            or tuple(_parse_rational(value) for value in declared) != point
        ):
            return _reject(
                "declared reconstruction does not match the point",
                method=method,
            )
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_RATIONAL",
            "method": method,
            "coverage": "NOT_APPLICABLE",
            "detail": "convex-combination witness replayed exactly",
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _reject("malformed convex-combination request", method=method)


def _check_separator_certificate_header(
    certificate: dict[str, Any],
    expected_bindings: object,
) -> str | None:
    if certificate.get("certificate_type") != "polytope.linear_separator":
        return "unexpected certificate format"
    if certificate.get("format_version") != "1":
        return "unsupported certificate version"
    if certificate.get("bindings") != expected_bindings:
        return "certificate bindings do not match"
    return None


def _check_separator_payload(
    payload: object,
    dimension: int,
) -> tuple[tuple[Fraction, ...], Fraction] | str:
    if not isinstance(payload, dict) or payload.get("sense") != "<=":
        return "separator must use <= sense"
    raw_coefficients = payload.get("coefficients")
    if not isinstance(raw_coefficients, list) or len(raw_coefficients) != dimension:
        return "separator dimension does not match"
    coefficients = tuple(_parse_rational(value) for value in raw_coefficients)
    if all(value == 0 for value in coefficients):
        return "separator normal cannot be zero"
    rhs = _parse_rational(payload.get("rhs"))
    return coefficients, rhs


def _check_separator_declared_values(
    payload: dict[str, Any],
    point_value: Fraction,
    max_generator: Fraction,
) -> str | None:
    if _parse_rational(payload.get("point_value")) != point_value:
        return "declared point value is incorrect"
    if _parse_rational(payload.get("max_generator_value")) != max_generator:
        return "declared maximum generator value is incorrect"
    if _parse_rational(payload.get("margin")) != point_value - max_generator:
        return "declared separator margin is incorrect"
    return None


def check_linear_separator(request: dict[str, Any]) -> dict[str, Any]:
    """Check a strict exact separator against every finite generator."""

    method = "EXHAUSTIVE_FINITE"
    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version", method=method)
        dimension, _, _ = _claim(request, "OUTSIDE_CONVEX_HULL")
        certificate = request["certificate"]["payload"]
        error = _check_separator_certificate_header(
            certificate, request["expected_bindings"]
        )
        if error is not None:
            return _reject(error, method=method)
        point = _point(request["candidate"]["payload"])
        generators = _generators(request["scope"]["payload"])
        error = _check_object_dimensions(point, generators, dimension)
        if error is not None:
            return _reject(error, method=method)
        result = _check_separator_payload(certificate.get("payload"), dimension)
        if isinstance(result, str):
            return _reject(result, method=method)
        coefficients, rhs = result
        generator_values = tuple(
            sum(
                (
                    coefficient * coordinate
                    for coefficient, coordinate in zip(
                        coefficients,
                        generator,
                        strict=True,
                    )
                ),
                Fraction(),
            )
            for generator in generators
        )
        if any(value > rhs for value in generator_values):
            return _reject("separator excludes a generator", method=method)
        point_value = sum(
            (
                coefficient * coordinate
                for coefficient, coordinate in zip(
                    coefficients,
                    point,
                    strict=True,
                )
            ),
            Fraction(),
        )
        max_generator = max(generator_values)
        if point_value <= rhs:
            return _reject("separator does not strictly exclude point", method=method)
        error = _check_separator_declared_values(
            certificate["payload"], point_value, max_generator
        )
        if error is not None:
            return _reject(error, method=method)
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_RATIONAL",
            "method": method,
            "coverage": "EXHAUSTIVE",
            "detail": "strict separator replayed against every generator",
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _reject("malformed separation-certificate request", method=method)
