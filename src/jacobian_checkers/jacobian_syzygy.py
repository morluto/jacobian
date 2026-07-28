"""Independent exact replay for bounded graded Jacobian syzygies.

This checker intentionally uses only the Python standard library and does not
import the SymPy producer or its helpers.
"""

from __future__ import annotations

import hashlib
import json
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
        raise ValueError("malformed rational components")
    fraction = Fraction(int(numerator), int(denominator))
    if numerator != str(fraction.numerator) or denominator != str(fraction.denominator):
        raise ValueError("noncanonical rational")
    return fraction


def _wire_rational(value: Fraction) -> dict[str, str]:
    return {"num": str(value.numerator), "den": str(value.denominator)}


def _parse_polynomial(
    value: object,
) -> tuple[tuple[str, str, str], dict[tuple[int, int, int], Fraction]]:
    if not isinstance(value, dict) or set(value) != {
        "polynomial_schema_version",
        "domain",
        "variables",
        "polynomial",
    }:
        raise ValueError("malformed polynomial")
    variables = value["variables"]
    body = value["polynomial"]
    if (
        value["polynomial_schema_version"] != "1"
        or value["domain"] != "QQ"
        or not isinstance(variables, list)
        or len(variables) != 3
        or not all(isinstance(variable, str) for variable in variables)
        or len(variables) != len(set(variables))
        or not isinstance(body, dict)
        or set(body) != {"terms"}
        or not isinstance(body["terms"], list)
    ):
        raise ValueError("malformed polynomial ring")
    terms: dict[tuple[int, int, int], Fraction] = {}
    previous: tuple[int, int, int] | None = None
    for item in body["terms"]:
        if not isinstance(item, dict) or set(item) != {"coefficient", "exponents"}:
            raise ValueError("malformed polynomial term")
        exponents = item["exponents"]
        if (
            not isinstance(exponents, list)
            or len(exponents) != 3
            or not all(
                type(exponent) is int and 0 <= exponent <= 127 for exponent in exponents
            )
        ):
            raise ValueError("malformed polynomial exponents")
        exponent_tuple = (exponents[0], exponents[1], exponents[2])
        coefficient = _rational(item["coefficient"])
        if coefficient == 0 or exponent_tuple in terms:
            raise ValueError("noncanonical polynomial support")
        if previous is not None and previous <= exponent_tuple:
            raise ValueError("polynomial support is not descending lexicographic")
        previous = exponent_tuple
        terms[exponent_tuple] = coefficient
    return (variables[0], variables[1], variables[2]), terms


def _wire_polynomial(
    variables: tuple[str, str, str],
    terms: dict[tuple[int, int, int], Fraction],
) -> dict[str, Any]:
    return {
        "polynomial_schema_version": "1",
        "domain": "QQ",
        "variables": list(variables),
        "polynomial": {
            "terms": [
                {
                    "coefficient": _wire_rational(coefficient),
                    "exponents": list(exponents),
                }
                for exponents, coefficient in sorted(
                    terms.items(),
                    reverse=True,
                )
                if coefficient
            ]
        },
    }


def _differentiate(
    terms: dict[tuple[int, int, int], Fraction],
    variable: int,
) -> dict[tuple[int, int, int], Fraction]:
    derivative: dict[tuple[int, int, int], Fraction] = {}
    for exponents, coefficient in terms.items():
        power = exponents[variable]
        if power:
            derived = list(exponents)
            derived[variable] -= 1
            derivative[(derived[0], derived[1], derived[2])] = coefficient * power
    return derivative


def _parse_linear_factor_product(
    factors: object,
    variables: object,
) -> tuple[tuple[str, str, str], dict[tuple[int, int, int], Fraction]]:
    if (
        not isinstance(variables, list)
        or len(variables) != 3
        or not all(isinstance(variable, str) for variable in variables)
        or len(variables) != len(set(variables))
        or not isinstance(factors, list)
        or not 1 <= len(factors) <= 16
    ):
        raise ValueError("malformed labelled linear-factor product")
    labels: list[str] = []
    product: dict[tuple[int, int, int], Fraction] = {(0, 0, 0): Fraction(1)}
    for factor in factors:
        if not isinstance(factor, dict) or set(factor) != {
            "label",
            "coefficients",
        }:
            raise ValueError("malformed labelled linear factor")
        label = factor["label"]
        coefficients = factor["coefficients"]
        if (
            not isinstance(label, str)
            or not isinstance(coefficients, list)
            or len(coefficients) != 3
        ):
            raise ValueError("malformed labelled linear factor")
        labels.append(label)
        parsed_coefficients = tuple(_rational(value) for value in coefficients)
        if not any(parsed_coefficients):
            raise ValueError("zero linear factor")
        expanded: dict[tuple[int, int, int], Fraction] = {}
        for exponents, product_coefficient in product.items():
            for variable, factor_coefficient in enumerate(parsed_coefficients):
                if not factor_coefficient:
                    continue
                target = list(exponents)
                target[variable] += 1
                target_tuple = (target[0], target[1], target[2])
                expanded[target_tuple] = (
                    expanded.get(target_tuple, Fraction(0))
                    + product_coefficient * factor_coefficient
                )
        product = {
            exponents: coefficient
            for exponents, coefficient in expanded.items()
            if coefficient
        }
    if len(labels) != len(set(labels)) or not product:
        raise ValueError("factor labels must be unique and product nonzero")
    return (variables[0], variables[1], variables[2]), product


def _homogeneous_basis(degree: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (first, second, degree - first - second)
        for first in range(degree, -1, -1)
        for second in range(degree - first, -1, -1)
    )


def _coefficient_matrix(
    partials: tuple[
        dict[tuple[int, int, int], Fraction],
        dict[tuple[int, int, int], Fraction],
        dict[tuple[int, int, int], Fraction],
    ],
    homogeneous_degree: int,
    multiplier_degree: int,
) -> tuple[
    tuple[tuple[int, int, int], ...],
    tuple[tuple[int, int, int], ...],
    list[list[Fraction]],
]:
    source_basis = _homogeneous_basis(multiplier_degree)
    target_basis = _homogeneous_basis(homogeneous_degree - 1 + multiplier_degree)
    row_by_exponent = {exponents: index for index, exponents in enumerate(target_basis)}
    matrix = [[Fraction(0) for _ in range(3 * len(source_basis))] for _ in target_basis]
    for component, partial in enumerate(partials):
        for basis_index, multiplier in enumerate(source_basis):
            column = component * len(source_basis) + basis_index
            for derivative_exponents, coefficient in partial.items():
                target = (
                    multiplier[0] + derivative_exponents[0],
                    multiplier[1] + derivative_exponents[1],
                    multiplier[2] + derivative_exponents[2],
                )
                matrix[row_by_exponent[target]][column] += coefficient
    return source_basis, target_basis, matrix


def _rref(
    matrix: list[list[Fraction]],
) -> tuple[list[list[Fraction]], tuple[int, ...]]:
    reduced = [row[:] for row in matrix]
    row_count = len(reduced)
    column_count = len(reduced[0]) if reduced else 0
    pivot_row = 0
    pivots: list[int] = []
    for column in range(column_count):
        selected = next(
            (row for row in range(pivot_row, row_count) if reduced[row][column]),
            None,
        )
        if selected is None:
            continue
        reduced[pivot_row], reduced[selected] = (
            reduced[selected],
            reduced[pivot_row],
        )
        pivot = reduced[pivot_row][column]
        reduced[pivot_row] = [value / pivot for value in reduced[pivot_row]]
        for row in range(row_count):
            if row == pivot_row:
                continue
            factor = reduced[row][column]
            if factor:
                reduced[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        reduced[row],
                        reduced[pivot_row],
                        strict=True,
                    )
                ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break
    return reduced, tuple(pivots)


def _determinant(matrix: list[list[Fraction]]) -> Fraction:
    if not matrix:
        return Fraction(1)
    work = [row[:] for row in matrix]
    determinant = Fraction(1)
    for column in range(len(work)):
        selected = next(
            (row for row in range(column, len(work)) if work[row][column]),
            None,
        )
        if selected is None:
            return Fraction(0)
        if selected != column:
            work[column], work[selected] = work[selected], work[column]
            determinant = -determinant
        pivot = work[column][column]
        determinant *= pivot
        for row in range(column + 1, len(work)):
            factor = work[row][column] / pivot
            for target_column in range(column + 1, len(work)):
                work[row][target_column] -= factor * work[column][target_column]
    return determinant


def _primitive_vector(vector: list[Fraction]) -> tuple[Fraction, ...]:
    denominator_lcm = 1
    for fraction_value in vector:
        denominator_lcm = (
            denominator_lcm
            * fraction_value.denominator
            // gcd(denominator_lcm, fraction_value.denominator)
        )
    integers = tuple(
        value.numerator * (denominator_lcm // value.denominator) for value in vector
    )
    divisor = 0
    for integer in integers:
        divisor = gcd(divisor, abs(integer))
    if divisor == 0:
        raise ValueError("zero nullspace vector")
    primitive = tuple(value // divisor for value in integers)
    if next(value for value in primitive if value) < 0:
        primitive = tuple(-value for value in primitive)
    return tuple(Fraction(value) for value in primitive)


def _first_kernel(
    reduced: list[list[Fraction]],
    pivots: tuple[int, ...],
    column_count: int,
) -> tuple[Fraction, ...] | None:
    free_columns = tuple(
        column for column in range(column_count) if column not in set(pivots)
    )
    if not free_columns:
        return None
    selected_free = free_columns[0]
    vector = [Fraction(0) for _ in range(column_count)]
    vector[selected_free] = Fraction(1)
    for row, pivot in enumerate(pivots):
        vector[pivot] = -reduced[row][selected_free]
    return _primitive_vector(vector)


def _matrix_entries(
    matrix: list[list[Fraction]],
) -> tuple[tuple[int, int, Fraction], ...]:
    return tuple(
        (row, column, value)
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
        if value
    )


def _matrix_digest(
    *,
    multiplier_degree: int,
    source_basis: tuple[tuple[int, int, int], ...],
    target_basis: tuple[tuple[int, int, int], ...],
    entries: tuple[tuple[int, int, Fraction], ...],
) -> str:
    payload = {
        "protocol": "jacobian.graded-jacobian-map.v1",
        "multiplier_degree": multiplier_degree,
        "source_monomial_basis": [list(item) for item in source_basis],
        "target_monomial_basis": [list(item) for item in target_basis],
        "entries": [
            [row, column, f"{value.numerator}/{value.denominator}"]
            for row, column, value in entries
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _expected_result(source: dict[str, Any]) -> dict[str, Any]:
    if set(source) != {
        "polynomial",
        "linear_factors",
        "linear_factor_variables",
        "max_degree",
        "coefficient_map_detail",
    }:
        raise ValueError("malformed graded syzygy request")
    max_degree = source["max_degree"]
    detail = source["coefficient_map_detail"]
    if (
        type(max_degree) is not int
        or not 0 <= max_degree <= 8
        or detail not in {"CERTIFICATES", "SPARSE_ENTRIES"}
    ):
        raise ValueError("graded syzygy request lies outside checker scope")
    if (source["polynomial"] is None) == (source["linear_factors"] is None):
        raise ValueError("graded syzygy request must select exactly one source")
    if source["polynomial"] is not None:
        if source["linear_factor_variables"] is not None:
            raise ValueError("expanded polynomial cannot carry factor variables")
        variables, polynomial = _parse_polynomial(source["polynomial"])
        source_kind = "EXPANDED_POLYNOMIAL"
    else:
        variables, polynomial = _parse_linear_factor_product(
            source["linear_factors"],
            source["linear_factor_variables"],
        )
        source_kind = "LABELLED_LINEAR_FACTOR_PRODUCT"
    if not polynomial:
        raise ValueError("zero source polynomial")
    degrees = {sum(exponents) for exponents in polynomial}
    if len(degrees) != 1:
        raise ValueError("nonhomogeneous source polynomial")
    homogeneous_degree = next(iter(degrees))
    if not 1 <= homogeneous_degree <= 16:
        raise ValueError("source degree lies outside checker scope")
    partials = (
        _differentiate(polynomial, 0),
        _differentiate(polynomial, 1),
        _differentiate(polynomial, 2),
    )
    degree_maps: list[dict[str, Any]] = []
    kernel_witness: dict[str, Any] | None = None
    first_degree: int | None = None
    for multiplier_degree in range(max_degree + 1):
        source_basis, target_basis, matrix = _coefficient_matrix(
            partials,
            homogeneous_degree,
            multiplier_degree,
        )
        reduced, pivots = _rref(matrix)
        rank = len(pivots)
        entries = _matrix_entries(matrix)
        rank_minor: dict[str, Any] | None = None
        if rank:
            selected_columns = [
                [matrix[row][column] for column in pivots] for row in range(len(matrix))
            ]
            transposed = [
                [selected_columns[row][column] for row in range(len(matrix))]
                for column in range(rank)
            ]
            _, independent_rows = _rref(transposed)
            minor = [
                [matrix[row][column] for column in pivots] for row in independent_rows
            ]
            determinant = _determinant(minor)
            if determinant == 0:
                raise ValueError("checker rank minor unexpectedly vanished")
            rank_minor = {
                "row_indices": list(independent_rows),
                "column_indices": list(pivots),
                "determinant": _wire_rational(determinant),
            }
        nullity = len(matrix[0]) - rank
        degree_maps.append(
            {
                "multiplier_degree": multiplier_degree,
                "source_monomial_basis": [list(item) for item in source_basis],
                "target_monomial_basis": [list(item) for item in target_basis],
                "row_count": len(matrix),
                "column_count": len(matrix[0]),
                "matrix_digest": _matrix_digest(
                    multiplier_degree=multiplier_degree,
                    source_basis=source_basis,
                    target_basis=target_basis,
                    entries=entries,
                ),
                "sparse_entries": (
                    [
                        {
                            "row": row,
                            "column": column,
                            "coefficient": _wire_rational(value),
                        }
                        for row, column, value in entries
                    ]
                    if detail == "SPARSE_ENTRIES"
                    else []
                ),
                "rank": rank,
                "nullity": nullity,
                "pivot_columns": list(pivots),
                "rank_minor": rank_minor,
                "injective": nullity == 0,
            }
        )
        if nullity:
            first_degree = multiplier_degree
            vector = _first_kernel(reduced, pivots, len(matrix[0]))
            if vector is None:
                raise ValueError("rank and nullspace computation disagree")
            block_size = len(source_basis)
            multipliers = []
            for component in range(3):
                coefficients = vector[
                    component * block_size : (component + 1) * block_size
                ]
                multiplier_terms = {
                    exponents: coefficient
                    for exponents, coefficient in zip(
                        source_basis,
                        coefficients,
                        strict=True,
                    )
                    if coefficient
                }
                multipliers.append(_wire_polynomial(variables, multiplier_terms))
            kernel_witness = {
                "multiplier_degree": multiplier_degree,
                "coefficient_vector": [_wire_rational(value) for value in vector],
                "multipliers": multipliers,
            }
            break
    searched_through = first_degree if first_degree is not None else max_degree
    return {
        "result_schema_version": "1",
        "variables": list(variables),
        "source_kind": source_kind,
        "expanded_polynomial": _wire_polynomial(variables, polynomial),
        "homogeneous_degree": homogeneous_degree,
        "searched_through_degree": searched_through,
        "coefficient_map_detail": detail,
        "partial_derivatives": [
            _wire_polynomial(variables, partial) for partial in partials
        ],
        "degree_maps": degree_maps,
        "status": "FOUND" if first_degree is not None else "NONE_THROUGH_BOUND",
        "first_syzygy_degree": first_degree,
        "kernel_witness": kernel_witness,
        "completion": "COMPLETE_THROUGH_BOUND",
        "verification_capability_id": (
            "polynomial.jacobian_syzygy.minimum_degree.verify"
        ),
        "verification_input_field": "result_uri",
    }


def check_graded_jacobian_syzygy(request: dict[str, Any]) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id="polynomial.jacobian_syzygy.minimum_degree.compute",
            witness_format="polynomial.jacobian-syzygy.graded-fraction-replay",
        )
        expected = _expected_result(source)
        if result != expected:
            return _reject(
                "stored result does not match independent exact graded-map replay"
            )
        return _accept(
            "independent exact rational replay accepted the complete graded rank "
            "ledger and first-kernel claim"
        )
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return _reject("malformed, unsupported, or mismatched checker request")


__all__ = ["check_graded_jacobian_syzygy"]
