"""Independent SymPy replay for finite polynomial maps and certificates.

This checker parses passive wire values itself and uses SymPy polynomial
arithmetic. It does not import the Python-FLINT producer or Jacobian's domain
contracts.
"""

from __future__ import annotations

from typing import Any

from jacobian_checkers.bound_artifacts import bound_request

_MAX_FIELD_ORDER = 4096


def _decision(*, accepted: bool, detail: str) -> dict[str, Any]:
    return {
        "accepted": accepted,
        "conclusion": "TRUE" if accepted else "UNKNOWN",
        "arithmetic": "EXACT_ALGEBRAIC",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": detail,
    }


def _presentation(value: object) -> tuple[dict[str, Any], int, tuple[int, ...]]:
    if not isinstance(value, dict) or set(value) != {
        "characteristic",
        "modulus_coefficients",
        "generator",
        "element_encoding_version",
    }:
        raise ValueError("finite-field presentation is malformed")
    prime = value["characteristic"]
    coefficients = value["modulus_coefficients"]
    if (
        type(prime) is not int
        or not isinstance(coefficients, list)
        or not 3 <= len(coefficients) <= 17
        or coefficients[-1] != 1
        or any(type(item) is not int or not 0 <= item < prime for item in coefficients)
        or not isinstance(value["generator"], str)
        or not value["generator"]
        or value["element_encoding_version"] != "power-basis-v1"
    ):
        raise ValueError("finite-field presentation is not canonical and bounded")

    from sympy import Poly, isprime, symbols

    if not isprime(prime):
        raise ValueError("finite-field characteristic is not prime")
    variable = symbols("z")
    modulus = Poly(
        sum(
            coefficient * variable**power
            for power, coefficient in enumerate(coefficients)
        ),
        variable,
        modulus=prime,
    )
    if (
        not modulus.is_irreducible
        or prime ** (len(coefficients) - 1) > _MAX_FIELD_ORDER
    ):
        raise ValueError("finite-field presentation is reducible or too large")
    return value, prime, tuple(coefficients)


def _element(
    value: object,
    *,
    presentation: dict[str, Any],
    prime: int,
    degree: int,
) -> tuple[int, ...]:
    if not isinstance(value, dict) or set(value) != {"presentation", "coordinates"}:
        raise ValueError("finite-field element is malformed")
    coordinates = value["coordinates"]
    if (
        value["presentation"] != presentation
        or not isinstance(coordinates, list)
        or len(coordinates) != degree
        or any(type(item) is not int or not 0 <= item < prime for item in coordinates)
    ):
        raise ValueError("finite-field element does not match its presentation")
    return tuple(coordinates)


def _encoded_coordinates(encoded: int, *, prime: int, degree: int) -> list[int]:
    return [(encoded // prime**power) % prime for power in range(degree)]


def _encoding(value: dict[str, Any]) -> int:
    prime = value["presentation"]["characteristic"]
    return sum(
        coordinate * prime**power
        for power, coordinate in enumerate(value["coordinates"])
    )


def _replay_table(polynomial_map: object) -> dict[str, Any]:
    if not isinstance(polynomial_map, dict) or set(polynomial_map) != {
        "domain",
        "codomain",
        "polynomial",
    }:
        raise ValueError("finite polynomial map is malformed")
    presentation, prime, modulus_coefficients = _presentation(polynomial_map["domain"])
    if polynomial_map["codomain"] != presentation:
        raise ValueError("finite polynomial map is not an exact self-map")
    raw_polynomial = polynomial_map["polynomial"]
    if not isinstance(raw_polynomial, dict) or set(raw_polynomial) != {
        "presentation",
        "variable",
        "coefficients",
    }:
        raise ValueError("finite polynomial is malformed")
    raw_coefficients = raw_polynomial["coefficients"]
    degree = len(modulus_coefficients) - 1
    if (
        raw_polynomial["presentation"] != presentation
        or not isinstance(raw_polynomial["variable"], str)
        or not raw_polynomial["variable"]
        or not isinstance(raw_coefficients, list)
        or not raw_coefficients
    ):
        raise ValueError("finite polynomial is not canonical")
    coefficients = tuple(
        _element(
            coefficient,
            presentation=presentation,
            prime=prime,
            degree=degree,
        )
        for coefficient in raw_coefficients
    )
    if len(coefficients) > 1 and not any(coefficients[-1]):
        raise ValueError("finite polynomial has a trailing zero coefficient")

    from sympy import Poly, symbols

    variable = symbols("z")
    modulus = Poly(
        sum(
            coefficient * variable**power
            for power, coefficient in enumerate(modulus_coefficients)
        ),
        variable,
        modulus=prime,
    )

    def as_polynomial(coordinates: tuple[int, ...]) -> Any:
        return Poly(
            sum(
                coefficient * variable**power
                for power, coefficient in enumerate(coordinates)
            ),
            variable,
            modulus=prime,
        )

    polynomial_coefficients = tuple(as_polynomial(value) for value in coefficients)
    entries: list[list[dict[str, Any]]] = []
    for encoded in range(prime**degree):
        source_coordinates = _encoded_coordinates(
            encoded,
            prime=prime,
            degree=degree,
        )
        source_polynomial = as_polynomial(tuple(source_coordinates))
        result = Poly(0, variable, modulus=prime)
        for coefficient in reversed(polynomial_coefficients):
            result = (result * source_polynomial + coefficient).rem(modulus)
        target_coordinates = [int(result.nth(power)) % prime for power in range(degree)]
        entries.append(
            [
                {"presentation": presentation, "coordinates": source_coordinates},
                {"presentation": presentation, "coordinates": target_coordinates},
            ]
        )
    return {"map": polynomial_map, "entries": entries}


def _verified_table(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"map", "entries"}:
        raise ValueError("finite map table is malformed")
    if value != _replay_table(value["map"]):
        raise ValueError("finite map table does not match independent replay")
    return value


def _table_request(claim: dict[str, Any]) -> dict[str, Any]:
    if set(claim) != {"table"}:
        raise ValueError("finite map-table request is malformed")
    return _verified_table(claim["table"])


def check_finite_map_table(request: dict[str, Any]) -> dict[str, Any]:
    claim, candidate = bound_request(
        request,
        operation_id="finite_field.polynomial_map.table.compute",
        witness_format="finite-field.polynomial-map-table.sympy-replay",
    )
    if set(claim) != {"polynomial_map"}:
        raise ValueError("finite map-table request is malformed")
    if candidate != _replay_table(claim["polynomial_map"]):
        return _decision(
            accepted=False,
            detail="candidate table does not match independent SymPy enumeration",
        )
    return _decision(
        accepted=True,
        detail="replayed the complete polynomial map with SymPy",
    )


def check_finite_map_fibers(request: dict[str, Any]) -> dict[str, Any]:
    claim, candidate = bound_request(
        request,
        operation_id="finite_field.polynomial_map.fibers.compute",
        witness_format="finite-field.polynomial-map-fibers.sympy-replay",
    )
    table = _table_request(claim)
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    targets: dict[tuple[int, ...], dict[str, Any]] = {}
    for source, target in table["entries"]:
        key = tuple(target["coordinates"])
        targets.setdefault(key, target)
        grouped.setdefault(key, []).append(source)
    expected = {
        "table": table,
        "fibers": [[targets[key], sources] for key, sources in grouped.items()],
    }
    if candidate != expected:
        return _decision(
            accepted=False,
            detail="candidate fibers do not partition the independently replayed map",
        )
    return _decision(accepted=True, detail="replayed the complete fiber partition")


def check_finite_map_collision(request: dict[str, Any]) -> dict[str, Any]:
    claim, candidate = bound_request(
        request,
        operation_id="finite_field.polynomial_map.collision.compute",
        witness_format="finite-field.polynomial-map-collision.sympy-replay",
    )
    table = _table_request(claim)
    seen: dict[tuple[int, ...], dict[str, Any]] = {}
    expected: dict[str, Any] | None = None
    for source, target in table["entries"]:
        key = tuple(target["coordinates"])
        if key in seen:
            expected = {
                "table": table,
                "left": seen[key],
                "right": source,
                "image": target,
            }
            break
        seen[key] = source
    if expected is None or candidate != expected:
        return _decision(
            accepted=False,
            detail="candidate collision does not match the independently replayed map",
        )
    return _decision(accepted=True, detail="replayed the exact collision")


def check_finite_map_permutation(request: dict[str, Any]) -> dict[str, Any]:
    claim, candidate = bound_request(
        request,
        operation_id="finite_field.polynomial_map.permutation.compute",
        witness_format="finite-field.polynomial-map-permutation.sympy-replay",
    )
    table = _table_request(claim)
    entries = table["entries"]
    if len({tuple(target["coordinates"]) for _, target in entries}) != len(entries):
        return _decision(accepted=False, detail="replayed map is not a permutation")
    inverse = sorted(entries, key=lambda entry: _encoding(entry[1]))
    expected = {
        "table": table,
        "inverse_entries": [[target, source] for source, target in inverse],
    }
    if candidate != expected:
        return _decision(
            accepted=False,
            detail="candidate inverse does not match the independently replayed map",
        )
    return _decision(accepted=True, detail="replayed the exact permutation inverse")
