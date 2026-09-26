"""One-shot, bounded PARI worker for rational modular-form basis prefixes."""

from __future__ import annotations

import hashlib
import math
import sys
from fractions import Fraction
from typing import Any

from jacobian._worker_protocol import encode_worker_result_frame
from jacobian.canonical import (
    format_canonical_integer,
    loads_strict_json,
)

_MAX_LEVEL = 10_000
_MAX_WEIGHT = 120
_MAX_PRECISION = 128
_MAX_DIMENSION = 32
_MAX_COEFFICIENT_DIGITS = 512
_MAX_CHARACTER_COEFFICIENT_DIGITS = 4
_MAX_CHARACTER_MODULUS = 2_048
_MAX_CHARACTER_ROOT_ORDER = 128


def _int(value: object, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            "PARI modular-form worker input is outside its admitted scalar range"
        )
    return value


def _as_fraction_pair(value: Any) -> list[str]:
    numerator = int(value.numerator())
    denominator = int(value.denominator())
    if denominator <= 0:
        raise RuntimeError("PARI returned a rational with a nonpositive denominator")
    if max(len(str(abs(numerator))), len(str(denominator))) > _MAX_COEFFICIENT_DIGITS:
        raise RuntimeError("PARI modular-form coefficient exceeds its digit envelope")
    return [format_canonical_integer(numerator), format_canonical_integer(denominator)]


def _as_cyclotomic_coordinates(pari: Any, value: Any, order: int) -> list[list[str]]:
    """Encode a PARI algebraic coefficient in its declared power basis."""
    polynomial = pari.lift(value)
    degree = sum(math.gcd(index, order) == 1 for index in range(1, order + 1))
    coefficients = []
    for index in range(degree):
        pair = _as_fraction_pair(pari.polcoef(polynomial, index))
        if (
            max(len(pair[0].lstrip("-")), len(pair[1]))
            > _MAX_CHARACTER_COEFFICIENT_DIGITS
        ):
            raise RuntimeError("PARI character coefficient exceeds its admitted height")
        coefficients.append(pair)
    return coefficients


def _character_vector(pari: Any, raw: object, field_order: object) -> tuple[Any, Any]:
    """Build PARI's standard-generator character and verify it on every unit."""

    if (
        type(field_order) is not int
        or not 1 <= field_order <= _MAX_CHARACTER_ROOT_ORDER
    ):
        raise ValueError("PARI character coefficient-field root order is invalid")
    if not isinstance(raw, dict) or set(raw) != {
        "modulus",
        "unit_residues",
        "generator_orders",
        "unit_coordinates",
        "coordinates",
    }:
        raise ValueError("PARI character request has an invalid shape")
    modulus = _int(raw["modulus"], minimum=1, maximum=_MAX_CHARACTER_MODULUS)
    units = raw["unit_residues"]
    orders = raw["generator_orders"]
    rows = raw["unit_coordinates"]
    coordinates = raw["coordinates"]
    if (
        type(units) is not list
        or type(orders) is not list
        or type(rows) is not list
        or type(coordinates) is not list
        or len(units)
        != sum(math.gcd(residue, modulus) == 1 for residue in range(modulus))
        or len(rows) != len(units)
        or len(coordinates) != len(orders)
        or any(type(order) is not int or order < 1 for order in orders)
        or any(
            type(value) is not int or not 0 <= value < order
            for value, order in zip(coordinates, orders, strict=True)
        )
    ):
        raise ValueError("PARI character request has malformed unit coordinates")
    if (
        any(type(unit) is not int for unit in units)
        or units != sorted(set(units))
        or units
        != [residue for residue in range(modulus) if math.gcd(residue, modulus) == 1]
        or any(type(row) is not list or len(row) != len(orders) for row in rows)
        or any(
            type(value) is not int or not 0 <= value < order
            for row in rows
            for value, order in zip(row, orders, strict=True)
        )
    ):
        raise ValueError("PARI character request has noncanonical unit coordinates")
    if len(units) * max(1, len(orders)) > 100_000:
        raise ValueError("PARI character agreement check exceeds its work envelope")
    character_order = (
        math.lcm(
            *(
                order // math.gcd(value, order)
                for value, order in zip(coordinates, orders, strict=True)
            )
        )
        if orders
        else 1
    )
    if field_order % character_order:
        raise ValueError(
            "PARI coefficient-field root order does not contain the character"
        )

    pari_group = pari.znstar(modulus, 1)
    # CyPari's Gen wrapper for `bid` does not expose GP member notation as
    # Python attributes.  The documented structure stores its finite group at
    # index 1, whose entries 1 and 2 are the cycle orders and actual generators.
    pari_unit_group = pari_group[1]
    pari_orders = tuple(int(value) for value in pari_unit_group[1])
    pari_generators = tuple(int(value) % modulus for value in pari_unit_group[2])
    if len(pari_orders) != len(pari_generators) or any(
        type(order) is not int or order < 2 for order in pari_orders
    ):
        raise RuntimeError("PARI returned an invalid unit-group generator presentation")
    unit_row = dict(zip(units, rows, strict=True))
    pari_character: list[int] = []
    for generator, pari_order in zip(pari_generators, pari_orders, strict=True):
        row = unit_row.get(generator)
        if row is None:
            raise RuntimeError("PARI unit-group generator is not a canonical unit")
        generator_exponent = (
            sum(
                (
                    Fraction(coordinate * unit_coordinate, order)
                    for coordinate, order, unit_coordinate in zip(
                        coordinates, orders, row, strict=True
                    )
                ),
                Fraction(0),
            )
            % 1
        )
        field_exponent = generator_exponent * field_order
        if field_exponent.denominator != 1:
            raise ValueError(
                "explicit coefficient field does not contain a character value"
            )
        pari_numerator = generator_exponent * pari_order
        if pari_numerator.denominator != 1:
            raise RuntimeError(
                "character value does not respect a PARI generator order"
            )
        pari_character.append(int(pari_numerator) % pari_order)

    pari_character_vector = pari(pari_character)
    for unit, row in zip(units, rows, strict=True):
        jacobian_value = (
            sum(
                (
                    Fraction(coordinate * unit_coordinate, order)
                    for coordinate, order, unit_coordinate in zip(
                        coordinates, orders, row, strict=True
                    )
                ),
                Fraction(0),
            )
            % 1
        )
        pari_log = tuple(int(value) for value in pari.znlog(unit, pari_group))
        if len(pari_log) != len(pari_orders):
            raise RuntimeError("PARI returned a malformed unit discrete logarithm")
        pari_fraction = sum(
            (
                Fraction(coefficient * log_value, pari_order)
                for coefficient, log_value, pari_order in zip(
                    pari_character, pari_log, pari_orders, strict=True
                )
            ),
            Fraction(0),
        )
        if (pari_fraction - jacobian_value).denominator != 1:
            raise RuntimeError(
                "PARI character conversion disagrees with Jacobian on a unit residue"
            )
    return pari_group, pari_character_vector


def _character_basis_vectors(
    pari: Any, pari_space: Any, precision: int, dimension: int, field_order: int
) -> list[list[list[list[str]]]]:
    """Normalize and encode the admitted character basis prefix."""
    if not 3 <= precision <= _MAX_PRECISION:
        raise ValueError(
            "character basis worker precision is outside its admitted range"
        )
    vectors: list[list[list[list[str]]]] = []
    if not dimension:
        return vectors
    coefficient_matrix = pari.mfcoefs(pari_space, precision - 1)
    row_count, column_count = (int(value) for value in coefficient_matrix.matsize())
    if row_count != precision or column_count != dimension:
        raise RuntimeError("PARI returned an unexpected q-coefficient matrix shape")
    for basis in range(dimension):
        raw_coefficients = [
            coefficient_matrix[term, basis] for term in range(precision)
        ]
        pivot = next((value for value in raw_coefficients if value != 0), None)
        if pivot is None:
            raise RuntimeError("PARI character basis has no Sturm-visible coefficient")
        normalized_coefficients = [value / pivot for value in raw_coefficients]
        vectors.append(
            [
                _as_cyclotomic_coordinates(pari, value, field_order)
                for value in normalized_coefficients
            ]
        )
    return vectors


def main() -> int:
    input_bytes = sys.stdin.buffer.read()
    request = loads_strict_json(input_bytes)
    if not isinstance(request, dict) or set(request) not in (
        {
            "level",
            "weight",
            "kind",
            "precision",
            "expected_dimension",
        },
        {
            "level",
            "weight",
            "kind",
            "precision",
            "expected_dimension",
            "character",
            "coefficient_field_order",
            "bridge_only",
        },
    ):
        raise ValueError("PARI modular-form worker request has an invalid shape")
    level = _int(request["level"], minimum=1, maximum=_MAX_LEVEL)
    weight = _int(request["weight"], minimum=0, maximum=_MAX_WEIGHT)
    precision = _int(request["precision"], minimum=1, maximum=_MAX_PRECISION)
    expected_dimension = _int(
        request["expected_dimension"], minimum=0, maximum=_MAX_DIMENSION
    )
    kind = request["kind"]
    if kind not in ("M", "S"):
        raise ValueError("PARI modular-form worker requires M or S")
    character_bridge_only = request.get("bridge_only", False)
    if (
        type(character_bridge_only) is not bool
        or ("character" in request and "coefficient_field_order" not in request)
        or ("character" not in request and character_bridge_only)
    ):
        raise ValueError("PARI character request has an invalid bridge mode")

    import cypari

    pari = cypari.pari
    if "character" in request:
        character_request = request["character"]
        if (
            not isinstance(character_request, dict)
            or character_request.get("modulus") != level
        ):
            raise ValueError("PARI character modulus differs from modular-form level")
        pari_group, pari_character = _character_vector(
            pari, character_request, request["coefficient_field_order"]
        )
        character_parent = [pari_group, pari_character]
        pari_space = pari.mfinit(
            [level, weight, character_parent], 4 if kind == "M" else 1
        )
        dimension = int(pari.mfdim(pari_space))
        if dimension != expected_dimension:
            raise RuntimeError("PARI and caller dimension formula disagree")
        if character_bridge_only:
            response: dict[str, object] = {
                "kind": "character_bridge",
                "backend_dimension": dimension,
                "character_coordinates": [int(value) for value in pari_character],
                "coefficient_field_order": request["coefficient_field_order"],
                "request_digest": hashlib.sha256(input_bytes).hexdigest(),
            }
        else:
            field_order = request["coefficient_field_order"]
            vectors = _character_basis_vectors(
                pari, pari_space, precision, dimension, field_order
            )
            response = {
                "kind": "character_complete",
                "backend_dimension": dimension,
                "vectors": vectors,
                "character_coordinates": [int(value) for value in pari_character],
                "coefficient_field_order": request["coefficient_field_order"],
                "request_digest": hashlib.sha256(input_bytes).hexdigest(),
            }
        sys.stdout.buffer.write(encode_worker_result_frame(response))
        return 0
    else:
        pari_space = pari.mfinit([level, weight], 4 if kind == "M" else 1)
    dimension = int(pari.mfdim(pari_space))
    if dimension != expected_dimension:
        raise RuntimeError("PARI and caller dimension formula disagree")
    rational_vectors: list[list[list[str]]] = []
    if dimension:
        coefficient_matrix = pari.mfcoefs(pari_space, precision - 1)
        row_count, column_count = (int(value) for value in coefficient_matrix.matsize())
        if row_count != precision or column_count != dimension:
            raise RuntimeError("PARI returned an unexpected q-coefficient matrix shape")
        rational_vectors = [
            [
                _as_fraction_pair(coefficient_matrix[term, basis])
                for term in range(precision)
            ]
            for basis in range(dimension)
        ]
    digest = hashlib.sha256(input_bytes).hexdigest()
    rational_response: dict[str, object] = {
        "kind": "complete",
        "backend_dimension": dimension,
        "vectors": rational_vectors,
        "request_digest": digest,
    }
    sys.stdout.buffer.write(encode_worker_result_frame(rational_response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
