"""Independent standard-library replay for finite simplicial topology.

This checker deliberately imports neither the topology producer nor its public
contracts. Only passive artifact-bound JSON and the generic binding parser
cross the clean-process checker boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request as _bound_request
from jacobian_checkers.certified_snf import (
    ParsedMatrix,
    ParsedSmithCertificate,
)
from jacobian_checkers.certified_snf import (
    parse_matrix as _parse_integer_matrix,
)
from jacobian_checkers.certified_snf import (
    validate_certificate as _validate_smith_certificate,
)

_MAX_VERTICES = 64
_MAX_FACETS = 128
_MAX_DIMENSION = 7
_MAX_FACES = 2048
_MAX_CHAIN_GROUP = 512
_MAX_MATRIX_CELLS = 131_072
_MAX_PRIME = 251
_MAX_INTEGRAL_CHAIN_GROUP = 16
_MAX_INTEGRAL_TOTAL_CHAIN_RANK = 32
_MAX_INTEGRAL_MATRIX_CELLS = 256
_MAX_INTEGER_DIGITS = 256
_INTEGER = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_LABEL_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:-"
)
_META = {
    "exactness": "EXACT_FINITE",
    "determinism": "DETERMINISTIC",
    "backend": "jacobian.topology",
    "backend_version": "1",
    "verification": "UNVERIFIED",
}


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(operation_id: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": f"independent modular topology replay accepted {operation_id}",
    }


def _strict_int(value: object) -> int:
    if type(value) is not int:
        raise ValueError("expected a strict integer")
    return value


def _label(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 32
        or value[0] not in _LABEL_CHARACTERS - frozenset("_.:-")
        or any(character not in _LABEL_CHARACTERS for character in value)
    ):
        raise ValueError("vertex label is not canonical")
    return value


def _simplex(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= _MAX_DIMENSION + 1:
        raise ValueError("simplex is malformed")
    simplex = tuple(_label(vertex) for vertex in value)
    if len(simplex) != len(set(simplex)):
        raise ValueError("simplex repeats a vertex")
    return simplex


def _closure(
    facets: Sequence[tuple[str, ...]],
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    faces: list[set[tuple[str, ...]]] = [set() for _ in range(_MAX_DIMENSION + 1)]
    for facet in facets:
        for size in range(1, len(facet) + 1):
            faces[size - 1].update(combinations(facet, size))
    highest = max(index for index, values in enumerate(faces) if values)
    return tuple(tuple(sorted(values)) for values in faces[: highest + 1])


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _complex_digest(complex_without_digest: dict[str, Any]) -> str:
    return (
        "sha256:" + hashlib.sha256(_canonical_json(complex_without_digest)).hexdigest()
    )


def _complex_from_request(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"vertices", "facets"}:
        raise ValueError("simplicial-complex request is malformed")
    raw_vertices = value["vertices"]
    raw_facets = value["facets"]
    if (
        not isinstance(raw_vertices, list)
        or not 1 <= len(raw_vertices) <= _MAX_VERTICES
        or not isinstance(raw_facets, list)
        or not 1 <= len(raw_facets) <= _MAX_FACETS
    ):
        raise ValueError("simplicial-complex request exceeds its bounds")
    vertices = tuple(_label(vertex) for vertex in raw_vertices)
    if len(vertices) != len(set(vertices)):
        raise ValueError("vertices are not unique")
    vertex_set = set(vertices)
    facets = tuple(tuple(sorted(_simplex(facet))) for facet in raw_facets)
    if len(facets) != len(set(facets)):
        raise ValueError("facets are not distinct")
    if any(not set(facet).issubset(vertex_set) for facet in facets):
        raise ValueError("facet contains an undeclared vertex")
    if set().union(*(set(facet) for facet in facets)) != vertex_set:
        raise ValueError("an isolated vertex lacks its singleton facet")
    if any(
        set(left) < set(right) or set(right) < set(left)
        for left, right in combinations(facets, 2)
    ):
        raise ValueError("facet presentation contains a non-maximal simplex")
    facets = tuple(sorted(facets))
    closure = _closure(facets)
    f_vector = [len(faces) for faces in closure]
    if sum(f_vector) > _MAX_FACES:
        raise ValueError("simplicial complex exceeds checker bounds")
    complex_without_digest = {
        "complex_format": "jacobian.finite-simplicial-complex/v1",
        "vertices": sorted(vertices),
        "maximal_simplices": [list(facet) for facet in facets],
        "faces_by_dimension": [
            {
                "dimension": dimension,
                "faces": [list(face) for face in faces],
            }
            for dimension, faces in enumerate(closure)
        ],
        "dimension": len(closure) - 1,
        "f_vector": f_vector,
        "closure_size": sum(f_vector),
        "orientation_convention": "LEXICOGRAPHIC_VERTEX_ORDER",
        "empty_simplex_stored": False,
    }
    return {
        **complex_without_digest,
        "complex_digest": _complex_digest(complex_without_digest),
    }


def _parse_complex(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("canonical complex is malformed")
    expected_fields = {
        "complex_format",
        "vertices",
        "maximal_simplices",
        "faces_by_dimension",
        "dimension",
        "f_vector",
        "closure_size",
        "orientation_convention",
        "empty_simplex_stored",
        "complex_digest",
    }
    if set(value) != expected_fields:
        raise ValueError("canonical complex fields are malformed")
    rebuilt = _complex_from_request(
        {
            "vertices": value["vertices"],
            "facets": value["maximal_simplices"],
        }
    )
    if value != rebuilt:
        raise ValueError("canonical complex does not match its complete face closure")
    return rebuilt


def _require_linear_bounds(complex_: dict[str, Any]) -> None:
    f_vector = complex_["f_vector"]
    if any(size > _MAX_CHAIN_GROUP for size in f_vector) or any(
        rows * columns > _MAX_MATRIX_CELLS
        for rows, columns in zip(
            (0, *f_vector[:-1]),
            f_vector,
            strict=True,
        )
    ):
        raise ValueError("simplicial complex exceeds modular replay bounds")


def _prime(value: object) -> int:
    prime = _strict_int(value)
    if not 2 <= prime <= _MAX_PRIME:
        raise ValueError("prime is outside the checker bound")
    divisor = 2
    while divisor * divisor <= prime:
        if prime % divisor == 0:
            raise ValueError("coefficient modulus is not prime")
        divisor += 1
    return prime


def _boundary(
    complex_: dict[str, Any],
    dimension: int,
    *,
    ring: str,
    prime: int | None,
) -> dict[str, Any]:
    field_prime: int | None = None
    if ring == "PRIME_FIELD":
        if prime is None:
            raise ValueError("prime-field boundary requires a prime")
        field_prime = prime
    source = [
        tuple(face) for face in complex_["faces_by_dimension"][dimension]["faces"]
    ]
    if dimension == 0:
        return {
            "source_dimension": 0,
            "target_dimension": -1,
            "rows": 0,
            "columns": len(source),
            "entries": [],
        }
    target = [
        tuple(face) for face in complex_["faces_by_dimension"][dimension - 1]["faces"]
    ]
    row_for_face = {face: row for row, face in enumerate(target)}
    entries: list[dict[str, int]] = []
    for column, simplex in enumerate(source):
        for removed in range(len(simplex)):
            face = simplex[:removed] + simplex[removed + 1 :]
            coefficient = 1 if removed % 2 == 0 else -1
            if field_prime is not None:
                coefficient %= field_prime
            entries.append(
                {
                    "row": row_for_face[face],
                    "column": column,
                    "value": coefficient,
                }
            )
    return {
        "source_dimension": dimension,
        "target_dimension": dimension - 1,
        "rows": len(target),
        "columns": len(source),
        "entries": sorted(entries, key=lambda item: (item["row"], item["column"])),
    }


def _augmentation(vertex_count: int) -> dict[str, Any]:
    return {
        "source_dimension": 0,
        "target_dimension": -1,
        "rows": 1,
        "columns": vertex_count,
        "entries": [
            {"row": 0, "column": column, "value": 1} for column in range(vertex_count)
        ],
    }


def _dense(matrix: dict[str, Any], *, prime: int | None) -> list[list[int]]:
    rows = _strict_int(matrix["rows"])
    columns = _strict_int(matrix["columns"])
    dense = [[0] * columns for _ in range(rows)]
    entries = matrix["entries"]
    if not isinstance(entries, list):
        raise ValueError("sparse matrix entries are malformed")
    previous: tuple[int, int] | None = None
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"row", "column", "value"}:
            raise ValueError("sparse matrix entry is malformed")
        row = _strict_int(entry["row"])
        column = _strict_int(entry["column"])
        coefficient = _strict_int(entry["value"])
        coordinate = (row, column)
        if (
            not 0 <= row < rows
            or not 0 <= column < columns
            or coefficient == 0
            or (previous is not None and coordinate <= previous)
        ):
            raise ValueError("sparse matrix entry is not canonical")
        if prime is not None:
            if not 1 <= coefficient < prime:
                raise ValueError("matrix coefficient lies outside the prime field")
            coefficient %= prime
        elif coefficient not in {-1, 1}:
            raise ValueError("integer boundary coefficient is not an orientation sign")
        dense[row][column] = coefficient
        previous = coordinate
    return dense


def _matrix_product_is_zero(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
    *,
    prime: int | None,
) -> bool:
    if not left:
        return True
    middle_count = len(right)
    if len(left[0]) != middle_count:
        return False
    columns = len(right[0]) if right else 0
    for row in range(len(left)):
        for column in range(columns):
            value = sum(
                left[row][middle] * right[middle][column]
                for middle in range(middle_count)
            )
            if prime is not None:
                value %= prime
            if value != 0:
                return False
    return True


def _chain_expected(
    complex_: dict[str, Any],
    *,
    ring: str,
    prime: int | None,
    convention: str,
) -> dict[str, Any]:
    if ring not in {"INTEGER", "PRIME_FIELD"}:
        raise ValueError("coefficient ring is unsupported")
    if ring == "INTEGER":
        if prime is not None:
            raise ValueError("integer chain request declares a prime")
    elif prime is None:
        raise ValueError("prime-field chain request omits a prime")
    bases = [
        {
            "dimension": item["dimension"],
            "simplices": item["faces"],
        }
        for item in complex_["faces_by_dimension"]
    ]
    boundaries = [
        _boundary(complex_, dimension, ring=ring, prime=prime)
        for dimension in range(complex_["dimension"] + 1)
    ]
    augmentation = (
        _augmentation(len(complex_["vertices"])) if convention == "REDUCED" else None
    )
    if convention not in {"REDUCED", "UNREDUCED"}:
        raise ValueError("homology convention is unsupported")
    ledger = []
    for upper_dimension in range(1, complex_["dimension"] + 1):
        lower = (
            augmentation
            if upper_dimension == 1 and augmentation is not None
            else boundaries[upper_dimension - 1]
        )
        if lower is None:
            raise ValueError("reconstructed boundary sequence is incomplete")
        if not _matrix_product_is_zero(
            _dense(lower, prime=prime),
            _dense(boundaries[upper_dimension], prime=prime),
            prime=prime,
        ):
            raise ValueError("reconstructed boundary does not square to zero")
        ledger.append(
            {
                "upper_dimension": upper_dimension,
                "product_rows": lower["rows"],
                "product_columns": boundaries[upper_dimension]["columns"],
                "nonzero_entries": 0,
                "product_is_zero": True,
            }
        )
    return {
        **_META,
        "complex_digest": complex_["complex_digest"],
        "coefficient_ring": ring,
        "prime": prime,
        "convention": convention,
        "simplex_bases": bases,
        "boundary_matrices": boundaries,
        "augmentation": augmentation,
        "boundary_squared_zero": ledger,
    }


def _rref(
    matrix: Sequence[Sequence[int]],
    *,
    columns: int,
    prime: int,
) -> tuple[list[list[int]], tuple[int, ...]]:
    rows = [[value % prime for value in row] for row in matrix]
    pivots: list[int] = []
    active_row = 0
    for column in range(columns):
        selected = next(
            (row for row in range(active_row, len(rows)) if rows[row][column] % prime),
            None,
        )
        if selected is None:
            continue
        rows[active_row], rows[selected] = rows[selected], rows[active_row]
        scale = pow(rows[active_row][column], prime - 2, prime)
        for index in range(columns):
            rows[active_row][index] = rows[active_row][index] * scale % prime
        for row in range(len(rows)):
            if row == active_row:
                continue
            multiple = rows[row][column] % prime
            if multiple:
                for index in range(columns):
                    rows[row][index] = (
                        rows[row][index] - multiple * rows[active_row][index]
                    ) % prime
        pivots.append(column)
        active_row += 1
        if active_row == len(rows):
            break
    return rows, tuple(pivots)


def _rank(
    matrix: Sequence[Sequence[int]],
    *,
    columns: int,
    prime: int,
) -> int:
    return len(_rref(matrix, columns=columns, prime=prime)[1])


def _vector_rank(vectors: Sequence[Sequence[int]], *, prime: int) -> int:
    if not vectors:
        return 0
    length = len(vectors[0])
    if any(len(vector) != length for vector in vectors):
        raise ValueError("vectors do not share a coordinate space")
    rows = [[vector[row] for vector in vectors] for row in range(length)]
    return _rank(rows, columns=len(vectors), prime=prime)


def _columns(matrix: Sequence[Sequence[int]], *, count: int) -> list[list[int]]:
    return [
        [matrix[row][column] for row in range(len(matrix))] for column in range(count)
    ]


def _matvec(
    matrix: Sequence[Sequence[int]],
    vector: Sequence[int],
    *,
    prime: int,
) -> list[int]:
    return [
        sum(value * coefficient for value, coefficient in zip(row, vector, strict=True))
        % prime
        for row in matrix
    ]


def _is_zero(vector: Sequence[int]) -> bool:
    return all(coordinate == 0 for coordinate in vector)


def _vectors(
    value: object,
    *,
    count: int,
    length: int,
    prime: int,
) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("vector basis has the wrong cardinality")
    vectors: list[list[int]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"coefficients"}:
            raise ValueError("modular vector is malformed")
        coefficients = item["coefficients"]
        if not isinstance(coefficients, list) or len(coefficients) != length:
            raise ValueError("modular vector has the wrong coordinate length")
        vector = [_strict_int(coefficient) for coefficient in coefficients]
        if any(not 0 <= coefficient < prime for coefficient in vector):
            raise ValueError("modular vector coefficient is noncanonical")
        vectors.append(vector)
    return vectors


def _integer(value: object) -> int:
    if (
        not isinstance(value, str)
        or _INTEGER.fullmatch(value) is None
        or len(value.lstrip("-")) > _MAX_INTEGER_DIGITS
    ):
        raise ValueError("integral homology coefficient is noncanonical")
    return int(value)


def _integer_vector(value: object, *, length: int) -> list[int]:
    if not isinstance(value, dict) or set(value) != {"coefficients"}:
        raise ValueError("integral homology vector is malformed")
    coefficients = value["coefficients"]
    if not isinstance(coefficients, list) or len(coefficients) != length:
        raise ValueError("integral homology vector has the wrong coordinate length")
    return [_integer(coefficient) for coefficient in coefficients]


def _parsed_matrix(
    entries: list[list[int]], *, rows: int, columns: int
) -> ParsedMatrix:
    if len(entries) != rows or any(len(row) != columns for row in entries):
        raise ValueError("reconstructed integer matrix has the wrong shape")
    return ParsedMatrix(rows=rows, columns=columns, entries=entries)


def _integer_matvec(matrix: ParsedMatrix, vector: Sequence[int]) -> list[int]:
    if matrix.columns != len(vector):
        raise ValueError("integer matrix and vector are not composable")
    return [
        sum(
            matrix.entries[row][column] * vector[column]
            for column in range(matrix.columns)
        )
        for row in range(matrix.rows)
    ]


def _integer_matmul(left: ParsedMatrix, right: ParsedMatrix) -> ParsedMatrix:
    if left.columns != right.rows:
        raise ValueError("integer matrices are not composable")
    return ParsedMatrix(
        rows=left.rows,
        columns=right.columns,
        entries=[
            [
                sum(
                    left.entries[row][middle] * right.entries[middle][column]
                    for middle in range(left.columns)
                )
                for column in range(right.columns)
            ]
            for row in range(left.rows)
        ],
    )


def _matrix_tail_columns(matrix: ParsedMatrix, *, start: int) -> ParsedMatrix:
    return ParsedMatrix(
        rows=matrix.rows,
        columns=matrix.columns - start,
        entries=[row[start:] for row in matrix.entries],
    )


def _certificate_within_integral_digit_budget(
    certificate: ParsedSmithCertificate,
) -> bool:
    matrices = (
        certificate.source,
        certificate.diagonal,
        certificate.left,
        certificate.right,
    )
    return all(
        len(str(abs(value))) <= _MAX_INTEGER_DIGITS
        for matrix in matrices
        for row in matrix.entries
        for value in row
    )


def _unit_vector(length: int, index: int) -> list[int]:
    return [1 if position == index else 0 for position in range(length)]


def _replay_materialization(source: dict[str, Any], result: dict[str, Any]) -> bool:
    expected_complex = _complex_from_request(source)
    return result == {
        **_META,
        "complex": expected_complex,
        "completeness": "COMPLETE_FACE_CLOSURE",
    }


def _replay_chain(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"complex", "coefficient_ring", "prime", "convention"}:
        return False
    complex_ = _parse_complex(source["complex"])
    _require_linear_bounds(complex_)
    raw_prime = source["prime"]
    prime = None if raw_prime is None else _prime(raw_prime)
    expected = _chain_expected(
        complex_,
        ring=source["coefficient_ring"],
        prime=prime,
        convention=source["convention"],
    )
    return result == expected


_HOMOLOGY_GROUP_FIELDS = {
    "dimension",
    "chain_dimension",
    "outgoing_boundary_rank",
    "cycle_dimension",
    "incoming_boundary_rank",
    "betti_number",
    "cycle_basis",
    "boundary_basis",
    "homology_basis",
    "quotient_span_rank",
}


def _check_homology_group(
    group: dict[str, Any],
    dimension: int,
    complex_: dict[str, Any],
    convention: str,
    boundaries: list[list[list[int]]],
    augmentation: list[list[int]],
    prime: int,
) -> bool:
    if not isinstance(group, dict) or set(group) != _HOMOLOGY_GROUP_FIELDS:
        return False
    chain_dimension = complex_["f_vector"][dimension]
    outgoing = (
        augmentation
        if dimension == 0 and convention == "REDUCED"
        else boundaries[dimension]
    )
    outgoing_rank = _rank(
        outgoing,
        columns=chain_dimension,
        prime=prime,
    )
    cycle_dimension = chain_dimension - outgoing_rank
    if dimension < complex_["dimension"]:
        incoming = boundaries[dimension + 1]
        incoming_columns = complex_["f_vector"][dimension + 1]
    else:
        incoming = [[] for _ in range(chain_dimension)]
        incoming_columns = 0
    actual_boundaries = _columns(incoming, count=incoming_columns)
    incoming_rank = _vector_rank(actual_boundaries, prime=prime)
    betti = cycle_dimension - incoming_rank
    integer_fields = {
        "dimension": dimension,
        "chain_dimension": chain_dimension,
        "outgoing_boundary_rank": outgoing_rank,
        "cycle_dimension": cycle_dimension,
        "incoming_boundary_rank": incoming_rank,
        "betti_number": betti,
        "quotient_span_rank": cycle_dimension,
    }
    if any(group[key] != value for key, value in integer_fields.items()):
        return False
    cycles = _vectors(
        group["cycle_basis"],
        count=cycle_dimension,
        length=chain_dimension,
        prime=prime,
    )
    reported_boundaries = _vectors(
        group["boundary_basis"],
        count=incoming_rank,
        length=chain_dimension,
        prime=prime,
    )
    homology = _vectors(
        group["homology_basis"],
        count=betti,
        length=chain_dimension,
        prime=prime,
    )
    return not (
        any(not _is_zero(_matvec(outgoing, vector, prime=prime)) for vector in cycles)
        or _vector_rank(cycles, prime=prime) != cycle_dimension
        or any(
            not _is_zero(_matvec(outgoing, vector, prime=prime))
            for vector in actual_boundaries
        )
        or _vector_rank(reported_boundaries, prime=prime) != incoming_rank
        or _vector_rank(
            (*actual_boundaries, *reported_boundaries),
            prime=prime,
        )
        != incoming_rank
        or any(
            not _is_zero(_matvec(outgoing, vector, prime=prime)) for vector in homology
        )
        or _vector_rank(
            (*reported_boundaries, *homology),
            prime=prime,
        )
        != cycle_dimension
    )


def _replay_homology(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"complex", "prime", "convention"}:
        return False
    complex_ = _parse_complex(source["complex"])
    _require_linear_bounds(complex_)
    prime = _prime(source["prime"])
    convention = source["convention"]
    if convention not in {"REDUCED", "UNREDUCED"}:
        return False
    result_fields = {
        *_META,
        "complex_digest",
        "coefficient_field",
        "prime",
        "convention",
        "orientation_convention",
        "dimension_range",
        "groups",
    }
    if (
        set(result) != result_fields
        or any(result[key] != value for key, value in _META.items())
        or result["complex_digest"] != complex_["complex_digest"]
        or result["coefficient_field"] != "PRIME_FIELD"
        or result["prime"] != prime
        or result["convention"] != convention
        or result["orientation_convention"] != "LEXICOGRAPHIC_VERTEX_ORDER"
        or result["dimension_range"] != [0, complex_["dimension"]]
        or not isinstance(result["groups"], list)
        or len(result["groups"]) != complex_["dimension"] + 1
    ):
        return False
    boundaries_raw = [
        _boundary(complex_, dimension, ring="PRIME_FIELD", prime=prime)
        for dimension in range(complex_["dimension"] + 1)
    ]
    boundaries = [_dense(matrix, prime=prime) for matrix in boundaries_raw]
    augmentation = _dense(
        _augmentation(len(complex_["vertices"])),
        prime=prime,
    )
    for upper_dimension in range(1, complex_["dimension"] + 1):
        lower = (
            augmentation
            if upper_dimension == 1 and convention == "REDUCED"
            else boundaries[upper_dimension - 1]
        )
        if not _matrix_product_is_zero(
            lower,
            boundaries[upper_dimension],
            prime=prime,
        ):
            return False
    for dimension, group in enumerate(result["groups"]):
        if not _check_homology_group(
            group, dimension, complex_, convention, boundaries, augmentation, prime
        ):
            return False
    return True


_INTEGRAL_GROUP_FIELDS = {
    "dimension",
    "chain_dimension",
    "incoming_chain_dimension",
    "outgoing_boundary_rank",
    "cycle_rank",
    "incoming_boundary_rank",
    "betti_number",
    "torsion_coefficients",
    "free_generators",
    "torsion_generators",
    "outgoing_smith_certificate",
    "boundary_in_cycle_coordinates",
    "incoming_smith_certificate",
    "generator_basis",
}


def _check_integral_free_generators(
    group: dict[str, Any],
    cycle_rank: int,
    chain_dimension: int,
    cycle_basis: Any,
    outgoing: Any,
    incoming_certificate: Any,
    incoming_rank: int,
) -> bool:
    for offset, item in enumerate(group["free_generators"]):
        if not isinstance(item, dict) or set(item) != {
            "cycle",
            "cycle_coordinates",
        }:
            return False
        coordinate = _integer_vector(
            item["cycle_coordinates"],
            length=cycle_rank,
        )
        cycle = _integer_vector(item["cycle"], length=chain_dimension)
        smith_index = incoming_rank + offset
        if (
            _integer_matvec(incoming_certificate.left, coordinate)
            != _unit_vector(cycle_rank, smith_index)
            or _integer_matvec(cycle_basis, coordinate) != cycle
            or any(_integer_matvec(outgoing, cycle))
        ):
            return False
    return True


def _check_integral_torsion_generators(
    group: dict[str, Any],
    cycle_rank: int,
    chain_dimension: int,
    cycle_basis: Any,
    outgoing: Any,
    incoming: Any,
    incoming_certificate: Any,
    incoming_chain_dimension: int,
    torsion_positions: list[tuple[int, int]],
) -> bool:
    for item, (smith_index, factor) in zip(
        group["torsion_generators"],
        torsion_positions,
        strict=True,
    ):
        if not isinstance(item, dict) or set(item) != {
            "order",
            "cycle",
            "cycle_coordinates",
            "bounding_chain",
        }:
            return False
        coordinate = _integer_vector(
            item["cycle_coordinates"],
            length=cycle_rank,
        )
        cycle = _integer_vector(item["cycle"], length=chain_dimension)
        bounding = _integer_vector(
            item["bounding_chain"],
            length=incoming_chain_dimension,
        )
        expected_bounding = [
            incoming_certificate.right.entries[row][smith_index]
            for row in range(incoming_chain_dimension)
        ]
        if (
            item["order"] != str(factor)
            or _integer_matvec(incoming_certificate.left, coordinate)
            != _unit_vector(cycle_rank, smith_index)
            or _integer_matvec(cycle_basis, coordinate) != cycle
            or any(_integer_matvec(outgoing, cycle))
            or bounding != expected_bounding
            or _integer_matvec(incoming, bounding)
            != [factor * value for value in cycle]
        ):
            return False
    return True


def _check_integral_homology_group(
    group: dict[str, Any],
    dimension: int,
    complex_: dict[str, Any],
    convention: str,
    boundaries: list[list[list[int]]],
    raw_boundaries: list[dict[str, Any]],
    augmentation: list[list[int]],
) -> bool:
    if not isinstance(group, dict) or set(group) != _INTEGRAL_GROUP_FIELDS:
        return False
    chain_dimension = complex_["f_vector"][dimension]
    if dimension == 0 and convention == "REDUCED":
        outgoing_entries = augmentation
        outgoing_rows = 1
    else:
        outgoing_entries = boundaries[dimension]
        outgoing_rows = raw_boundaries[dimension]["rows"]
    outgoing = _parsed_matrix(
        outgoing_entries,
        rows=outgoing_rows,
        columns=chain_dimension,
    )
    outgoing_certificate = _validate_smith_certificate(
        group["outgoing_smith_certificate"]
    )
    if (
        not _certificate_within_integral_digit_budget(outgoing_certificate)
        or outgoing_certificate.source != outgoing
    ):
        return False
    outgoing_rank = outgoing_certificate.rank
    cycle_rank = chain_dimension - outgoing_rank
    cycle_basis = _matrix_tail_columns(
        outgoing_certificate.right,
        start=outgoing_rank,
    )
    if dimension < complex_["dimension"]:
        incoming_entries = boundaries[dimension + 1]
        incoming_chain_dimension = complex_["f_vector"][dimension + 1]
    else:
        incoming_chain_dimension = 0
        incoming_entries = [[] for _ in range(chain_dimension)]
    incoming = _parsed_matrix(
        incoming_entries,
        rows=chain_dimension,
        columns=incoming_chain_dimension,
    )
    coordinates = _parse_integer_matrix(
        group["boundary_in_cycle_coordinates"],
        maximum_digits=_MAX_INTEGER_DIGITS,
    )
    if (
        coordinates.rows != cycle_rank
        or coordinates.columns != incoming_chain_dimension
        or _integer_matmul(cycle_basis, coordinates) != incoming
    ):
        return False
    incoming_certificate = _validate_smith_certificate(
        group["incoming_smith_certificate"]
    )
    if (
        not _certificate_within_integral_digit_budget(incoming_certificate)
        or incoming_certificate.source != coordinates
    ):
        return False
    incoming_rank = incoming_certificate.rank
    betti_number = cycle_rank - incoming_rank
    torsion_positions = [
        (index, factor)
        for index, factor in enumerate(incoming_certificate.factors)
        if factor > 1
    ]
    if (
        group["dimension"] != dimension
        or group["chain_dimension"] != chain_dimension
        or group["incoming_chain_dimension"] != incoming_chain_dimension
        or group["outgoing_boundary_rank"] != outgoing_rank
        or group["cycle_rank"] != cycle_rank
        or group["incoming_boundary_rank"] != incoming_rank
        or group["betti_number"] != betti_number
        or group["torsion_coefficients"]
        != [str(factor) for _, factor in torsion_positions]
        or group["generator_basis"]
        != "CANONICAL_SIMPLEX_BASIS_VIA_CERTIFIED_SMITH_TRANSFORMATIONS"
        or not isinstance(group["free_generators"], list)
        or len(group["free_generators"]) != betti_number
        or not isinstance(group["torsion_generators"], list)
        or len(group["torsion_generators"]) != len(torsion_positions)
    ):
        return False
    if not _check_integral_free_generators(
        group,
        cycle_rank,
        chain_dimension,
        cycle_basis,
        outgoing,
        incoming_certificate,
        incoming_rank,
    ):
        return False
    return _check_integral_torsion_generators(
        group,
        cycle_rank,
        chain_dimension,
        cycle_basis,
        outgoing,
        incoming,
        incoming_certificate,
        incoming_chain_dimension,
        torsion_positions,
    )


def _replay_integral_homology(
    source: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    if set(source) != {"complex", "convention"}:
        return False
    complex_ = _parse_complex(source["complex"])
    _require_linear_bounds(complex_)
    if (
        any(size > _MAX_INTEGRAL_CHAIN_GROUP for size in complex_["f_vector"])
        or sum(complex_["f_vector"]) > _MAX_INTEGRAL_TOTAL_CHAIN_RANK
        or any(
            rows * columns > _MAX_INTEGRAL_MATRIX_CELLS
            for rows, columns in zip(
                (0, *complex_["f_vector"][:-1]),
                complex_["f_vector"],
                strict=True,
            )
        )
    ):
        return False
    convention = source["convention"]
    if convention not in {"REDUCED", "UNREDUCED"}:
        return False
    result_fields = {
        *_META,
        "complex_digest",
        "coefficient_ring",
        "convention",
        "orientation_convention",
        "dimension_range",
        "groups",
        "completeness",
        "decomposition",
    }
    if (
        set(result) != result_fields
        or any(result[key] != value for key, value in _META.items())
        or result["complex_digest"] != complex_["complex_digest"]
        or result["coefficient_ring"] != "ZZ"
        or result["convention"] != convention
        or result["orientation_convention"] != "LEXICOGRAPHIC_VERTEX_ORDER"
        or result["dimension_range"] != [0, complex_["dimension"]]
        or result["completeness"] != "FREE_TORSION_AND_BOUND_GENERATORS"
        or result["decomposition"] != "DIRECT_SUM_Z_AND_FINITE_CYCLIC_FACTORS"
        or not isinstance(result["groups"], list)
        or len(result["groups"]) != complex_["dimension"] + 1
    ):
        return False
    raw_boundaries = [
        _boundary(complex_, dimension, ring="INTEGER", prime=None)
        for dimension in range(complex_["dimension"] + 1)
    ]
    boundaries = [_dense(matrix, prime=None) for matrix in raw_boundaries]
    augmentation_raw = _augmentation(len(complex_["vertices"]))
    augmentation = _dense(augmentation_raw, prime=None)
    for dimension, group in enumerate(result["groups"]):
        if not _check_integral_homology_group(
            group,
            dimension,
            complex_,
            convention,
            boundaries,
            raw_boundaries,
            augmentation,
        ):
            return False
    return True


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
            return _reject("topology candidate failed independent exact replay")
        return _accept(operation_id)
    except (KeyError, TypeError, ValueError, ArithmeticError, OverflowError):
        return _reject("malformed, unsupported, or mismatched topology request")


def check_simplicial_complex_materialization(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="topology.simplicial_complex.materialize",
        witness_format="topology.simplicial-complex.closure-replay",
        replay=_replay_materialization,
    )


def check_simplicial_chain_complex(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="topology.simplicial_complex.chain_complex.compute",
        witness_format="topology.simplicial-chain.boundary-replay",
        replay=_replay_chain,
    )


def check_simplicial_homology(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="topology.simplicial_homology.compute",
        witness_format="topology.simplicial-homology.modular-replay",
        replay=_replay_homology,
    )


def check_integral_simplicial_homology(request: object) -> dict[str, Any]:
    return _run(
        request,
        operation_id="topology.simplicial_homology.integral.compute",
        witness_format="topology.simplicial-homology.integral-smith-certificate-v1",
        replay=_replay_integral_homology,
    )


__all__ = [
    "check_integral_simplicial_homology",
    "check_simplicial_chain_complex",
    "check_simplicial_complex_materialization",
    "check_simplicial_homology",
]
