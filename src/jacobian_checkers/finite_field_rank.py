"""Independent SymPy replay for prime-field linear-map rank.

This checker parses the passive wire values itself. It does not import the
finite-field producer, its FLINT conversions, or Jacobian domain contracts.
"""

from __future__ import annotations

from typing import Any

from jacobian_checkers.bound_artifacts import bound_request

_MAX_DIMENSION = 256


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _axis(value: object) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, dict) or set(value) != {"name", "labels"}:
        raise ValueError("axis is malformed")
    name = value["name"]
    labels = value["labels"]
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(labels, list)
        or not 1 <= len(labels) <= _MAX_DIMENSION
        or any(not isinstance(label, str) or not label for label in labels)
        or len(labels) != len(set(labels))
    ):
        raise ValueError("axis is not a bounded ordered label set")
    return name, tuple(labels)


def _presentation(value: object) -> tuple[int, tuple[int, ...], str]:
    if not isinstance(value, dict) or set(value) != {
        "characteristic",
        "modulus_coefficients",
        "generator",
        "element_encoding_version",
    }:
        raise ValueError("finite-field presentation is malformed")
    prime = value["characteristic"]
    coefficients = value["modulus_coefficients"]
    generator = value["generator"]
    if (
        type(prime) is not int
        or not isinstance(coefficients, list)
        or not 3 <= len(coefficients) <= 17
        or coefficients[-1] != 1
        or any(type(item) is not int or not 0 <= item < prime for item in coefficients)
        or not isinstance(generator, str)
        or not generator
        or value["element_encoding_version"] != "power-basis-v1"
    ):
        raise ValueError("finite-field presentation is not canonical and bounded")

    from sympy import Poly, isprime, symbols

    if not isprime(prime):
        raise ValueError("finite-field characteristic is not prime")
    variable = symbols("x")
    modulus = Poly(
        sum(
            coefficient * variable**power
            for power, coefficient in enumerate(coefficients)
        ),
        variable,
        modulus=prime,
    )
    if not modulus.is_irreducible:
        raise ValueError("finite-field modulus is reducible")
    return prime, tuple(coefficients), generator


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


def _restriction_claim(
    claim: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if set(claim) != {"subspace", "direction"}:
        raise ValueError("restriction request is malformed")
    subspace = claim["subspace"]
    direction = claim["direction"]
    if not isinstance(subspace, dict) or set(subspace) != {
        "presentation",
        "basis_axis",
        "basis",
    }:
        raise ValueError("finite-dimensional subspace is malformed")
    if not isinstance(direction, dict) or set(direction) != {
        "presentation",
        "axis",
        "coordinates",
    }:
        raise ValueError("projective direction is malformed")
    presentation = subspace["presentation"]
    if not isinstance(presentation, dict) or direction["presentation"] != presentation:
        raise ValueError("restriction values use different field presentations")
    return subspace, direction, presentation


def _basis_matrix(
    raw_matrix: object,
    *,
    presentation: dict[str, Any],
    direction_axis: tuple[str, tuple[str, ...]],
    column_axis: tuple[str, tuple[str, ...]] | None,
    polynomial: Any,
    prime: int,
    degree: int,
) -> tuple[tuple[str, tuple[str, ...]], tuple[tuple[Any, ...], ...]]:
    if not isinstance(raw_matrix, dict) or set(raw_matrix) != {
        "presentation",
        "row_axis",
        "column_axis",
        "entries",
    }:
        raise ValueError("axis-bound matrix is malformed")
    current_rows = _axis(raw_matrix["row_axis"])
    current_columns = _axis(raw_matrix["column_axis"])
    if (
        raw_matrix["presentation"] != presentation
        or current_rows != direction_axis
        or (column_axis is not None and current_columns != column_axis)
    ):
        raise ValueError("subspace matrix parent or axes do not match")
    entries = raw_matrix["entries"]
    if (
        not isinstance(entries, list)
        or len(entries) != len(current_rows[1])
        or any(
            not isinstance(row, list) or len(row) != len(current_columns[1])
            for row in entries
        )
    ):
        raise ValueError("axis-bound matrix dimensions do not match")
    return current_columns, tuple(
        tuple(
            polynomial(
                _element(
                    value,
                    presentation=presentation,
                    prime=prime,
                    degree=degree,
                )
            )
            for value in row
        )
        for row in entries
    )


def _restricted_column(
    matrix: tuple[tuple[Any, ...], ...],
    direction: tuple[Any, ...],
    *,
    column_count: int,
    degree: int,
    prime: int,
    modulus: Any,
    variable: Any,
) -> tuple[int, ...]:
    from sympy import Poly

    coordinates: list[int] = []
    for column in range(column_count):
        image = Poly(0, variable, modulus=prime)
        for row, value in enumerate(direction):
            image = (image + matrix[row][column] * value).rem(modulus)
        coordinates.extend(int(image.nth(power)) % prime for power in range(degree))
    return tuple(coordinates)


def _replay_restriction(claim: dict[str, Any]) -> dict[str, Any]:
    subspace, direction, presentation = _restriction_claim(claim)
    prime, modulus_coefficients, generator = _presentation(presentation)
    degree = len(modulus_coefficients) - 1
    basis_axis = _axis(subspace["basis_axis"])
    direction_axis = _axis(direction["axis"])
    direction_values = (
        tuple(
            _element(
                value,
                presentation=presentation,
                prime=prime,
                degree=degree,
            )
            for value in direction["coordinates"]
        )
        if isinstance(direction["coordinates"], list)
        else ()
    )
    if len(direction_values) != len(direction_axis[1]):
        raise ValueError("projective direction does not match its axis")

    basis = subspace["basis"]
    if (
        not isinstance(basis, list)
        or not 1 <= len(basis) <= _MAX_DIMENSION
        or len(basis) != len(basis_axis[1])
    ):
        raise ValueError("subspace basis is not bounded by its axis")

    from sympy import Poly, symbols

    variable = symbols("x")
    modulus = Poly(
        sum(
            coefficient * variable**power
            for power, coefficient in enumerate(modulus_coefficients)
        ),
        variable,
        modulus=prime,
    )

    def polynomial(coordinates: tuple[int, ...]) -> Any:
        return Poly(
            sum(
                coefficient * variable**power
                for power, coefficient in enumerate(coordinates)
            ),
            variable,
            modulus=prime,
        )

    direction_polynomials = tuple(polynomial(value) for value in direction_values)
    columns: list[tuple[int, ...]] = []
    column_axis: tuple[str, tuple[str, ...]] | None = None
    for raw_matrix in basis:
        column_axis, matrix = _basis_matrix(
            raw_matrix,
            presentation=presentation,
            direction_axis=direction_axis,
            column_axis=column_axis,
            polynomial=polynomial,
            prime=prime,
            degree=degree,
        )
        columns.append(
            _restricted_column(
                matrix,
                direction_polynomials,
                column_count=len(column_axis[1]),
                degree=degree,
                prime=prime,
                modulus=modulus,
                variable=variable,
            )
        )
    if column_axis is None:
        raise ValueError("subspace basis is empty")
    ordered_basis = ("1", generator, *(f"{generator}^{n}" for n in range(2, degree)))
    target_labels = tuple(
        f"{label}:{basis_label}"
        for label in column_axis[1]
        for basis_label in ordered_basis
    )
    return {
        "source_axis": subspace["basis_axis"],
        "target_axis": {
            "name": f"Res({column_axis[0]})",
            "labels": list(target_labels),
        },
        "matrix": {
            "prime": prime,
            "entries": [list(row) for row in zip(*columns, strict=True)],
            "columns": len(basis),
        },
    }


def check_finite_field_restriction(request: dict[str, Any]) -> dict[str, Any]:
    claim, candidate = bound_request(
        request,
        operation_id="finite_field.restrict_scalars.compute",
        witness_format="finite-field.restriction.sympy-replay",
    )
    if candidate != _replay_restriction(claim):
        return _reject("candidate map does not match independent SymPy replay")
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_ALGEBRAIC",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": "replayed restriction of scalars with SymPy polynomial arithmetic",
    }


def _matrix_rank(linear_map: object) -> int:
    if not isinstance(linear_map, dict) or set(linear_map) != {
        "source_axis",
        "target_axis",
        "matrix",
    }:
        raise ValueError("linear map is malformed")
    matrix = linear_map["matrix"]
    if not isinstance(matrix, dict) or set(matrix) != {
        "prime",
        "entries",
        "columns",
    }:
        raise ValueError("prime-field matrix is malformed")
    prime = matrix["prime"]
    columns = matrix["columns"]
    entries = matrix["entries"]
    if (
        type(prime) is not int
        or type(columns) is not int
        or not isinstance(entries, list)
        or not 0 <= columns <= _MAX_DIMENSION
        or len(entries) > _MAX_DIMENSION
        or any(
            not isinstance(row, list)
            or len(row) != columns
            or any(type(value) is not int or not 0 <= value < prime for value in row)
            for row in entries
        )
    ):
        raise ValueError("prime-field matrix exceeds its exact bounded shape")

    from sympy import GF, isprime
    from sympy.polys.matrices import DomainMatrix

    if not isprime(prime):
        raise ValueError("matrix modulus is not prime")
    if not entries or columns == 0:
        return 0
    return int(DomainMatrix(entries, (len(entries), columns), GF(prime)).rank())


def check_finite_field_linear_map_rank(request: dict[str, Any]) -> dict[str, Any]:
    claim, candidate = bound_request(
        request,
        operation_id="finite_field.linear_map.rank.compute",
        witness_format="finite-field.linear-map-rank.sympy-replay",
    )
    if set(claim) != {"direction", "linear_map"} or set(candidate) != {
        "direction",
        "linear_map",
        "rank",
    }:
        raise ValueError("rank relation is malformed")
    if (
        candidate["direction"] != claim["direction"]
        or candidate["linear_map"] != claim["linear_map"]
    ):
        return _reject("candidate is not bound to the supplied direction and map")
    expected = _matrix_rank(claim["linear_map"])
    if type(candidate["rank"]) is not int or candidate["rank"] != expected:
        return _reject("candidate rank does not match independent SymPy replay")
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": "recomputed the exact rank over the bound prime field with SymPy",
    }


__all__ = [
    "check_finite_field_linear_map_rank",
    "check_finite_field_restriction",
]
