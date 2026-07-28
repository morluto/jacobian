"""Independent Python-FLINT replay for exact domain-operation results.

The checked producers use SymPy.  This module deliberately imports neither
SymPy nor Jacobian code; only passive JSON values cross the checker boundary.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import flint
from flint import fmpq, fmpq_mat, fmpq_poly, fmpz_mat

from jacobian_checkers.bound_artifacts import bound_request as _bound_request

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_PYTHON_FLINT_VERSION = "0.9.0"
_FLINT_VERSION = "3.6.0"


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
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _integer(value: object) -> int:
    if not isinstance(value, str) or _INTEGER.fullmatch(value) is None:
        raise ValueError("integer is not canonical")
    parsed = int(value)
    if str(parsed) != value:
        raise ValueError("integer is not canonical")
    return parsed


def _fraction(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational is malformed")
    numerator = _integer(value["num"])
    denominator = _integer(value["den"])
    if denominator <= 0:
        raise ValueError("rational denominator must be positive")
    result = Fraction(numerator, denominator)
    if (result.numerator, result.denominator) != (numerator, denominator):
        raise ValueError("rational is not reduced")
    return result


def _q(value: object) -> fmpq:
    value = _fraction(value)
    return fmpq(value.numerator, value.denominator)


def _polynomial(value: object) -> fmpq_poly:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "polynomial_schema_version",
            "domain",
            "variables",
            "polynomial",
        }
        or value["polynomial_schema_version"] != "1"
        or value["domain"] != "QQ"
    ):
        raise ValueError("polynomial is malformed")
    variables = value["variables"]
    if not isinstance(variables, list) or len(variables) != 1:
        raise ValueError("checker supports univariate polynomials only")
    body = value["polynomial"]
    if not isinstance(body, dict) or set(body) != {"terms"}:
        raise ValueError("sparse polynomial is malformed")
    terms = body["terms"]
    if not isinstance(terms, list):
        raise ValueError("polynomial terms are malformed")
    coefficients: dict[int, fmpq] = {}
    for term in terms:
        if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
            raise ValueError("polynomial term is malformed")
        exponents = term["exponents"]
        if (
            not isinstance(exponents, list)
            or len(exponents) != 1
            or not isinstance(exponents[0], int)
            or exponents[0] < 0
            or exponents[0] in coefficients
        ):
            raise ValueError("polynomial exponent is malformed")
        coefficient = _q(term["coefficient"])
        if coefficient == 0:
            raise ValueError("zero sparse terms are not canonical")
        coefficients[exponents[0]] = coefficient
    dense = [fmpq(0)] * (max(coefficients, default=-1) + 1)
    for exponent, coefficient in coefficients.items():
        dense[exponent] = coefficient
    return fmpq_poly(dense)


def _polynomial_variable(value: object) -> str:
    if not isinstance(value, dict):
        raise ValueError("polynomial is malformed")
    variables = value.get("variables")
    if (
        not isinstance(variables, list)
        or len(variables) != 1
        or not isinstance(variables[0], str)
        or not variables[0]
    ):
        raise ValueError("checker supports one named polynomial variable")
    return variables[0]


def _same_polynomial_variable(*values: object) -> str:
    variables = tuple(_polynomial_variable(value) for value in values)
    if len(set(variables)) != 1:
        raise ValueError("polynomial artifacts use different variable names")
    return variables[0]


def _rational_matrix(value: object) -> fmpq_mat:
    if not isinstance(value, dict) or set(value) != {"domain", "entries"}:
        raise ValueError("rational matrix is malformed")
    if value["domain"] != "QQ":
        raise ValueError("rational matrix domain is unsupported")
    entries = value["entries"]
    if (
        not isinstance(entries, list)
        or not entries
        or not isinstance(entries[0], list)
        or not entries[0]
        or any(
            not isinstance(row, list) or len(row) != len(entries[0]) for row in entries
        )
    ):
        raise ValueError("rational matrix shape is malformed")
    return fmpq_mat([[_q(item) for item in row] for row in entries])


def _integer_matrix(value: object) -> fmpz_mat:
    if not isinstance(value, dict) or set(value) != {"domain", "entries"}:
        raise ValueError("integer matrix is malformed")
    if value["domain"] != "ZZ":
        raise ValueError("integer matrix domain is unsupported")
    entries = value["entries"]
    if (
        not isinstance(entries, list)
        or not entries
        or not isinstance(entries[0], list)
        or not entries[0]
        or any(
            not isinstance(row, list) or len(row) != len(entries[0]) for row in entries
        )
    ):
        raise ValueError("integer matrix shape is malformed")
    return fmpz_mat([[_integer(item) for item in row] for row in entries])


def _run(
    request: object,
    *,
    operation_id: str,
    witness_format: str,
    replay: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> dict[str, Any]:
    try:
        if (
            flint.__version__ != _PYTHON_FLINT_VERSION
            or flint.__FLINT_VERSION__ != _FLINT_VERSION
        ):
            return _reject("authorized Python-FLINT runtime is unavailable")
        source, result = _bound_request(
            request,
            operation_id=operation_id,
            witness_format=witness_format,
        )
        if not replay(source, result):
            return _reject(
                "declared result does not match independent Python-FLINT replay"
            )
        return _accept(f"independent Python-FLINT replay accepted {operation_id}")
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


def _gcd(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"left", "right"} or set(result) != {
        "gcd",
        "bezout",
        "normalization",
    }:
        return False
    if result["normalization"] != "MONIC":
        return False
    bezout = result["bezout"]
    if not isinstance(bezout, dict) or set(bezout) != {
        "left_multiplier",
        "right_multiplier",
    }:
        return False
    _same_polynomial_variable(
        source["left"],
        source["right"],
        result["gcd"],
        bezout["left_multiplier"],
        bezout["right_multiplier"],
    )
    left, right, declared = (
        _polynomial(source["left"]),
        _polynomial(source["right"]),
        _polynomial(result["gcd"]),
    )
    left_multiplier = _polynomial(bezout["left_multiplier"])
    right_multiplier = _polynomial(bezout["right_multiplier"])
    return (
        declared == left.gcd(right)
        and left_multiplier * left + right_multiplier * right == declared
    )


def check_polynomial_gcd(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="polynomial.compute.gcd",
        witness_format="polynomial.gcd.flint-replay",
        replay=_gcd,
    )


def _resultant(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"left", "right", "elimination_variable"}:
        return False
    if set(result) != {"elimination_variable", "resultant", "convention"}:
        return False
    if (
        result["convention"] != "SYLVESTER_DETERMINANT"
        or source["elimination_variable"] != result["elimination_variable"]
    ):
        return False
    variable = _same_polynomial_variable(source["left"], source["right"])
    if source["elimination_variable"] != variable:
        raise ValueError("elimination variable does not name the polynomial variable")
    declared = result["resultant"]
    if not isinstance(declared, dict) or set(declared) != {"kind", "value"}:
        return False
    if declared["kind"] != "SCALAR":
        raise ValueError("checker supports univariate resultants only")
    return _q(declared["value"]) == _polynomial(source["left"]).resultant(
        _polynomial(source["right"])
    )


def check_polynomial_resultant(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="polynomial.compute.resultant",
        witness_format="polynomial.resultant.flint-replay",
        replay=_resultant,
    )


def _discriminant(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"polynomial", "variable"}:
        return False
    if set(result) != {"variable", "discriminant", "convention"}:
        return False
    declared = result["discriminant"]
    if (
        source["variable"] != result["variable"]
        or result["convention"] != "STANDARD_UNIVARIATE"
        or not isinstance(declared, dict)
        or set(declared) != {"kind", "value"}
        or declared["kind"] != "SCALAR"
    ):
        return False
    if source["variable"] != _polynomial_variable(source["polynomial"]):
        raise ValueError("discriminant variable does not name the polynomial variable")
    return _q(declared["value"]) == _polynomial(source["polynomial"]).discriminant()


def check_polynomial_discriminant(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="polynomial.compute.discriminant",
        witness_format="polynomial.discriminant.flint-replay",
        replay=_discriminant,
    )


def _square_free(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"polynomial"} or set(result) != {
        "coefficient",
        "factors",
        "reconstructed",
        "normalization",
    }:
        return False
    if result["normalization"] != "MONIC_FACTORS":
        return False
    polynomial = _polynomial(source["polynomial"])
    coefficient, expected = polynomial.factor_squarefree()
    factors = result["factors"]
    if not isinstance(factors, list):
        return False
    declared = []
    for item in factors:
        if not isinstance(item, dict) or set(item) != {"factor", "multiplicity"}:
            return False
        _same_polynomial_variable(source["polynomial"], item["factor"])
        declared.append((_polynomial(item["factor"]), item["multiplicity"]))
    _same_polynomial_variable(source["polynomial"], result["reconstructed"])
    normalized_expected = []
    normalized_coefficient = coefficient
    for factor, multiplicity in expected:
        leading = factor.leading_coefficient()
        normalized_expected.append((factor / leading, multiplicity))
        normalized_coefficient *= leading**multiplicity
    return (
        _q(result["coefficient"]) == normalized_coefficient
        and declared == normalized_expected
        and _polynomial(result["reconstructed"]) == polynomial
    )


def check_polynomial_square_free(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="polynomial.compute.square_free_decomposition",
        witness_format="polynomial.square-free.flint-replay",
        replay=_square_free,
    )


def _matrix_source(source: dict[str, Any], *, integer: bool = False) -> Any:
    if set(source) != {"matrix"}:
        raise ValueError("matrix request is malformed")
    return (_integer_matrix if integer else _rational_matrix)(source["matrix"])


def _rref(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if (
        set(result)
        != {
            "reduced_matrix",
            "rank",
            "pivot_columns",
            "free_columns",
            "convention",
        }
        or result["convention"] != "UNIQUE_RREF_OVER_QQ"
    ):
        return False
    expected, rank = _matrix_source(source).rref()
    declared = _rational_matrix(result["reduced_matrix"])
    pivots = tuple(
        next(column for column in range(expected.ncols()) if expected[row, column])
        for row in range(rank)
    )
    return (
        declared == expected
        and result["rank"] == rank
        and tuple(result["pivot_columns"]) == pivots
        and tuple(result["free_columns"])
        == tuple(column for column in range(expected.ncols()) if column not in pivots)
    )


def check_matrix_rref(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="matrix.normal_form.rref.compute",
        witness_format="matrix.rref.flint-replay",
        replay=_rref,
    )


def _nullspace(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if (
        set(result)
        != {
            "ambient_dimension",
            "nullity",
            "basis_vectors",
            "free_columns",
            "convention",
        }
        or result["convention"] != "RREF_FUNDAMENTAL_BASIS"
    ):
        return False
    matrix = _matrix_source(source)
    reduced, rank = matrix.rref()
    pivots = tuple(
        next(column for column in range(reduced.ncols()) if reduced[row, column])
        for row in range(rank)
    )
    free = tuple(column for column in range(reduced.ncols()) if column not in pivots)
    expected = []
    for free_column in free:
        vector = [fmpq(0)] * reduced.ncols()
        vector[free_column] = fmpq(1)
        for row, pivot in enumerate(pivots):
            vector[pivot] = -reduced[row, free_column]
        expected.append(vector)
    declared = [[_q(item) for item in vector] for vector in result["basis_vectors"]]
    return (
        result["ambient_dimension"] == reduced.ncols()
        and result["nullity"] == len(free)
        and tuple(result["free_columns"]) == free
        and declared == expected
    )


def check_matrix_nullspace(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="matrix.nullspace.compute",
        witness_format="matrix.nullspace.flint-replay",
        replay=_nullspace,
    )


def _characteristic_polynomial(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(result) != {
        "variable",
        "degree",
        "coefficients_descending",
        "monic",
        "convention",
    } or (
        result["variable"] != "lambda"
        or result["monic"] is not True
        or result["convention"] != "DET_LAMBDA_I_MINUS_A"
    ):
        return False
    matrix = _matrix_source(source)
    polynomial = matrix.charpoly()
    expected = [polynomial[index] for index in range(polynomial.degree(), -1, -1)]
    return (
        result["degree"] == polynomial.degree()
        and [_q(item) for item in result["coefficients_descending"]] == expected
    )


def check_matrix_characteristic_polynomial(
    request: dict[str, Any],
) -> dict[str, Any]:
    return _run(
        request,
        operation_id="matrix.characteristic_polynomial.compute",
        witness_format="matrix.characteristic-polynomial.flint-replay",
        replay=_characteristic_polynomial,
    )


def _smith(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(result) != {
        "normal_form",
        "rank",
        "invariant_factors",
        "transformation_available",
        "convention",
    } or (
        result["transformation_available"] is not False
        or result["convention"] != "POSITIVE_DIVISIBILITY_DIAGONAL"
    ):
        return False
    expected = _matrix_source(source, integer=True).snf()
    declared = _integer_matrix(result["normal_form"])
    factors = tuple(
        abs(int(expected[index, index]))
        for index in range(min(expected.nrows(), expected.ncols()))
        if expected[index, index]
    )
    canonical = fmpz_mat(expected.nrows(), expected.ncols())
    for index, factor in enumerate(factors):
        canonical[index, index] = factor
    return (
        declared == canonical
        and result["rank"] == len(factors)
        and tuple(_integer(item) for item in result["invariant_factors"]) == factors
    )


def check_matrix_smith_normal_form(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="matrix.normal_form.smith.compute",
        witness_format="matrix.smith-normal-form.flint-replay",
        replay=_smith,
    )


__all__ = [
    "check_matrix_characteristic_polynomial",
    "check_matrix_nullspace",
    "check_matrix_rref",
    "check_matrix_smith_normal_form",
    "check_polynomial_discriminant",
    "check_polynomial_gcd",
    "check_polynomial_resultant",
    "check_polynomial_square_free",
]
