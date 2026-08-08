"""Independent Python-FLINT replay for exact domain-operation results.

The checked producers use SymPy.  This module deliberately imports neither
SymPy nor Jacobian code; only passive JSON values cross the checker boundary.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from itertools import product
from typing import Any

import flint
from flint import fmpq, fmpq_mat, fmpq_poly, fmpz, fmpz_mat

from jacobian_checkers.bound_artifacts import bound_request as _bound_request

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_PYTHON_FLINT_VERSION = "0.9.0"
_FLINT_VERSION = "3.6.0"
_RESIDUE_VARIABLE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_MAX_RESIDUE_VARIABLES = 6
_MAX_RESIDUE_DOMAIN_SIZE = 32
_MAX_RESIDUE_TERMS = 64
_MAX_RESIDUE_EXPONENT = 32
_MAX_RESIDUE_ASSIGNMENTS = 4_096
_MAX_RESIDUE_MODULUS = 1_000_000
_MAX_MATRIX_DIMENSION = 32
_MAX_MATRIX_INPUT_DIGITS = 256
_MAX_MATRIX_OUTPUT_DIGITS = 32_768


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


def _reject_exact_integer(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept_exhaustive_integer(detail: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": detail,
    }


def _integer(value: object) -> Any:
    if not isinstance(value, str) or _INTEGER.fullmatch(value) is None:
        raise ValueError("integer is not canonical")
    # FLINT parses decimal input without crossing Python's 4,300-digit limit.
    return fmpz(value)


def _fraction(value: object) -> tuple[fmpz, fmpz]:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational is malformed")
    numerator = _integer(value["num"])
    denominator = _integer(value["den"])
    if denominator <= 0:
        raise ValueError("rational denominator must be positive")
    result = fmpq(numerator, denominator)
    if (result.numer(), result.denom()) != (numerator, denominator):
        raise ValueError("rational is not reduced")
    return numerator, denominator


def _q(value: object) -> fmpq:
    numerator, denominator = _fraction(value)
    return fmpq(numerator, denominator)


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
    if (
        not isinstance(value, dict)
        or set(value) != {"matrix_schema_version", "domain", "entries"}
        or value["matrix_schema_version"] != "1"
    ):
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
    if (
        not isinstance(value, dict)
        or set(value) != {"matrix_schema_version", "domain", "entries"}
        or value["matrix_schema_version"] != "1"
    ):
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


def _bounded_rational_matrix(value: object, *, maximum_digits: int) -> fmpq_mat:
    if not isinstance(value, dict):
        raise ValueError("rational matrix is malformed")
    entries = value.get("entries")
    if (
        not isinstance(entries, list)
        or not 1 <= len(entries) <= _MAX_MATRIX_DIMENSION
        or not isinstance(entries[0], list)
        or not 1 <= len(entries[0]) <= _MAX_MATRIX_DIMENSION
    ):
        raise ValueError("rational matrix exceeds the checker dimension bound")
    for row in entries:
        if not isinstance(row, list) or len(row) != len(entries[0]):
            raise ValueError("rational matrix shape is malformed")
        for item in row:
            if not isinstance(item, dict) or set(item) != {"num", "den"}:
                raise ValueError("rational matrix scalar is malformed")
            for component in (item["num"], item["den"]):
                if (
                    not isinstance(component, str)
                    or len(component.lstrip("-")) > maximum_digits
                ):
                    raise ValueError("rational matrix scalar exceeds the checker bound")
    return _rational_matrix(value)


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


def _factorization_value(source: dict[str, Any], *, positive: bool) -> Any:
    if set(source) != {"value", "resource_budget"}:
        raise ValueError("factorization source is malformed")
    budget = source["resource_budget"]
    if (
        not isinstance(budget, dict)
        or set(budget) != {"wall_seconds"}
        or type(budget["wall_seconds"]) is not int
        or not 1 <= budget["wall_seconds"] <= 30
    ):
        raise ValueError("factorization budget is malformed")
    value = _integer(source["value"])
    if (positive and value < 1) or (not positive and value == 0):
        raise ValueError("factorization source is outside the operation domain")
    return value


def _factorization_witness(
    target: int,
    factors: object,
) -> list[tuple[int, int]]:
    if not isinstance(factors, list) or len(factors) > 256:
        raise ValueError("factorization witness is malformed")

    product = 1
    parsed: list[tuple[int, int]] = []
    previous_prime = 1
    for factor in factors:
        if not isinstance(factor, dict) or set(factor) != {"prime", "power"}:
            raise ValueError("prime-power entry is malformed")
        prime = _integer(factor["prime"])
        power = factor["power"]
        if prime <= previous_prime or type(power) is not int or not 1 <= power <= 1_000:
            raise ValueError("prime-power entry is noncanonical")
        for _ in range(power):
            if product > target // prime:
                raise ValueError("prime-power product exceeds the source")
            product *= prime
        parsed.append((prime, power))
        previous_prime = prime
    if product != target:
        raise ValueError("prime-power product does not equal the source")

    replayed = [
        (int(prime), int(power)) for prime, power in flint.fmpz(target).factor()
    ]
    if parsed != replayed:
        raise ValueError("factorization differs from Python-FLINT replay")
    return parsed


def _prime_factorization(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(result) != {"factors"}:
        return False
    value = _factorization_value(source, positive=False)
    _factorization_witness(abs(value), result["factors"])
    return True


def check_integer_prime_factorization(
    request: dict[str, Any],
) -> dict[str, Any]:
    return _run(
        request,
        operation_id="integer.compute.prime_factorization",
        witness_format="integer.prime-factorization.flint-replay",
        replay=_prime_factorization,
    )


def _powerful_number(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(result) != {
        "semantics_version",
        "is_powerful",
        "factors",
        "violating_primes",
    }:
        return False
    if (
        result["semantics_version"] != "powerful-number.prime-exponents-at-least-two.v1"
        or type(result["is_powerful"]) is not bool
    ):
        return False

    value = _factorization_value(source, positive=True)
    factors = _factorization_witness(value, result["factors"])
    expected_violations = [str(prime) for prime, power in factors if power < 2]
    expected_is_powerful = not expected_violations
    return (
        result["violating_primes"] == expected_violations
        and result["is_powerful"] is expected_is_powerful
    )


def check_integer_powerful_number(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="integer.decide.powerful",
        witness_format="integer.powerful.flint-replay",
        replay=_powerful_number,
    )


def _strict_integer(value: object, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("bounded integer is malformed")
    return value


def _modular_residue_source(
    source: dict[str, Any],
) -> tuple[
    int,
    list[str],
    list[list[int]],
    list[tuple[int, list[int]]],
]:
    if set(source) != {"modulus", "variables", "terms"}:
        raise ValueError("modular-polynomial source is malformed")
    modulus = _strict_integer(
        source["modulus"],
        minimum=2,
        maximum=_MAX_RESIDUE_MODULUS,
    )
    names, domains = _modular_residue_variables(source["variables"], modulus)
    terms = _modular_residue_terms(source["terms"], len(names), modulus)
    return modulus, names, domains, terms


def _modular_residue_variables(
    value: object,
    modulus: int,
) -> tuple[list[str], list[list[int]]]:
    if not isinstance(value, list) or not 1 <= len(value) <= _MAX_RESIDUE_VARIABLES:
        raise ValueError("modular-polynomial variables are malformed")
    names: list[str] = []
    domains: list[list[int]] = []
    assignment_count = 1
    for variable in value:
        if not isinstance(variable, dict) or set(variable) != {"name", "residues"}:
            raise ValueError("modular-polynomial variable is malformed")
        name = variable["name"]
        residues = variable["residues"]
        if (
            not isinstance(name, str)
            or _RESIDUE_VARIABLE.fullmatch(name) is None
            or not isinstance(residues, list)
            or not 1 <= len(residues) <= _MAX_RESIDUE_DOMAIN_SIZE
            or any(
                type(residue) is not int or not 0 <= residue < modulus
                for residue in residues
            )
            or residues != sorted(set(residues))
        ):
            raise ValueError("modular-polynomial variable domain is noncanonical")
        names.append(name)
        domains.append(residues)
        assignment_count *= len(residues)
    if len(names) != len(set(names)):
        raise ValueError("modular-polynomial variable names are not unique")
    if assignment_count > _MAX_RESIDUE_ASSIGNMENTS:
        raise ValueError("modular-polynomial assignment scope is too large")
    return names, domains


def _modular_residue_terms(
    value: object,
    variable_count: int,
    modulus: int,
) -> list[tuple[int, list[int]]]:
    if not isinstance(value, list) or len(value) > _MAX_RESIDUE_TERMS:
        raise ValueError("modular-polynomial terms are malformed")
    terms: list[tuple[int, list[int]]] = []
    for term in value:
        if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
            raise ValueError("modular-polynomial term is malformed")
        coefficient_text = term["coefficient"]
        exponents = term["exponents"]
        if (
            not isinstance(coefficient_text, str)
            or len(coefficient_text) > 256
            or not isinstance(exponents, list)
            or len(exponents) != variable_count
            or any(
                type(exponent) is not int or not 0 <= exponent <= _MAX_RESIDUE_EXPONENT
                for exponent in exponents
            )
        ):
            raise ValueError("modular-polynomial sparse term is malformed")
        coefficient = _integer(coefficient_text) % modulus
        if coefficient == 0:
            raise ValueError("modular-polynomial sparse term is zero modulo m")
        terms.append((coefficient, exponents))
    exponent_vectors = [tuple(exponents) for _, exponents in terms]
    if exponent_vectors != sorted(set(exponent_vectors)):
        raise ValueError("modular-polynomial terms are not in canonical order")
    return terms


def _cartesian_assignments(domains: list[list[int]]) -> list[list[int]]:
    return [list(assignment) for assignment in product(*domains)]


def _evaluate_modular_polynomial_with_flint(
    terms: list[tuple[int, list[int]]],
    assignment: list[int],
    modulus: int,
) -> int:
    modulus_value = flint.fmpz(modulus)
    value = flint.fmpz(0)
    for coefficient, exponents in terms:
        monomial = flint.fmpz(coefficient)
        for coordinate, exponent in zip(assignment, exponents, strict=True):
            monomial *= pow(flint.fmpz(coordinate), exponent, modulus_value)
            monomial %= modulus_value
        value += monomial
        value %= modulus_value
    return int(value)


def _strict_result_integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("modular-polynomial result integer is malformed")
    return value


def _modular_residue_result_shape(result: dict[str, Any]) -> None:
    allowed_keys = (
        {
            "semantics_version",
            "modulus",
            "variable_order",
            "domains",
            "normalized_terms",
            "enumeration_scope",
            "total_assignments",
            "image",
            "residue_counts",
            "witnesses",
            "table",
        },
        {
            "semantics_version",
            "modulus",
            "variable_order",
            "domains",
            "normalized_terms",
            "enumeration_scope",
            "total_assignments",
            "image",
            "residue_counts",
            "witnesses",
        },
    )
    if set(result) not in allowed_keys:
        raise ValueError("modular-polynomial result is malformed")
    _strict_result_integer(result["modulus"])
    _strict_result_integer(result["total_assignments"])
    if not isinstance(result["variable_order"], list) or not all(
        isinstance(name, str) for name in result["variable_order"]
    ):
        raise ValueError("modular-polynomial result variable order is malformed")
    if not isinstance(result["domains"], list) or any(
        not isinstance(domain, list)
        or any(type(residue) is not int for residue in domain)
        for domain in result["domains"]
    ):
        raise ValueError("modular-polynomial result domains are malformed")
    if not isinstance(result["normalized_terms"], list) or any(
        not isinstance(term, dict)
        or set(term) != {"coefficient", "exponents"}
        or type(term["coefficient"]) is not int
        or not isinstance(term["exponents"], list)
        or any(type(exponent) is not int for exponent in term["exponents"])
        for term in result["normalized_terms"]
    ):
        raise ValueError("normalized modular-polynomial terms are malformed")
    if not isinstance(result["image"], list) or any(
        type(residue) is not int for residue in result["image"]
    ):
        raise ValueError("modular-polynomial image is malformed")
    if not isinstance(result["residue_counts"], list) or any(
        not isinstance(item, dict)
        or set(item) != {"residue", "count"}
        or type(item["residue"]) is not int
        or type(item["count"]) is not int
        for item in result["residue_counts"]
    ):
        raise ValueError("modular-polynomial residue counts are malformed")
    if not isinstance(result["witnesses"], list) or any(
        not isinstance(item, dict)
        or set(item) != {"residue", "assignment"}
        or type(item["residue"]) is not int
        or not isinstance(item["assignment"], list)
        or any(type(value) is not int for value in item["assignment"])
        for item in result["witnesses"]
    ):
        raise ValueError("modular-polynomial witnesses are malformed")
    if (
        "table" in result
        and result["table"] is not None
        and (
            not isinstance(result["table"], list)
            or any(
                not isinstance(row, dict)
                or set(row) != {"assignment", "residue"}
                or type(row["residue"]) is not int
                or not isinstance(row["assignment"], list)
                or any(type(value) is not int for value in row["assignment"])
                for row in result["table"]
            )
        )
    ):
        raise ValueError("modular-polynomial assignment table is malformed")


def _modular_polynomial_residue_image(
    source: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    modulus, names, domains, terms = _modular_residue_source(source)
    _modular_residue_result_shape(result)
    normalized_terms = [
        {"coefficient": coefficient, "exponents": exponents}
        for coefficient, exponents in terms
    ]
    assignments = _cartesian_assignments(domains)
    residues = [
        _evaluate_modular_polynomial_with_flint(terms, assignment, modulus)
        for assignment in assignments
    ]
    image = sorted(set(residues))
    counts = [
        {"residue": residue, "count": residues.count(residue)} for residue in image
    ]
    first_assignments: dict[int, list[int]] = {}
    for assignment, residue in zip(assignments, residues, strict=True):
        first_assignments.setdefault(residue, assignment)
    witnesses = [
        {"residue": residue, "assignment": first_assignments[residue]}
        for residue in image
    ]
    table = [
        {"assignment": assignment, "residue": residue}
        for assignment, residue in zip(assignments, residues, strict=True)
    ]
    table_matches = (
        "table" not in result or result["table"] is None or result["table"] == table
    )
    return bool(
        result["semantics_version"] == "modular-polynomial-residue-image.v1"
        and result["modulus"] == modulus
        and result["variable_order"] == names
        and result["domains"] == domains
        and result["normalized_terms"] == normalized_terms
        and result["enumeration_scope"] == "COMPLETE_DECLARED_CARTESIAN_PRODUCT"
        and result["total_assignments"] == len(assignments)
        and result["image"] == image
        and result["residue_counts"] == counts
        and result["witnesses"] == witnesses
        and table_matches
    )


def check_modular_polynomial_residue_image(
    request: dict[str, Any],
) -> dict[str, Any]:
    operation_id = "modular.polynomial_residue_image.compute"
    try:
        if (
            flint.__version__ != _PYTHON_FLINT_VERSION
            or flint.__FLINT_VERSION__ != _FLINT_VERSION
        ):
            return _reject_exact_integer(
                "authorized Python-FLINT runtime is unavailable"
            )
        source, result = _bound_request(
            request,
            operation_id=operation_id,
            witness_format="modular.polynomial-residue-image.flint-replay",
        )
        if not _modular_polynomial_residue_image(source, result):
            return _reject_exact_integer(
                "declared result does not match independent Python-FLINT "
                "exhaustive modular-polynomial replay"
            )
        return _accept_exhaustive_integer(
            f"independent Python-FLINT exhaustive replay accepted {operation_id}"
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject_exact_integer(
            "malformed, unsupported, or mismatched checker request"
        )


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
            "rank",
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
        and result["rank"] == rank
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


def _matrix_product(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(source) != {"left", "right"} or set(result) != {
        "product",
        "left_rows",
        "inner_dimension",
        "right_columns",
        "convention",
    }:
        return False
    if result["convention"] != "STANDARD_ROW_BY_COLUMN_PRODUCT_OVER_QQ":
        return False
    left = _bounded_rational_matrix(
        source["left"],
        maximum_digits=_MAX_MATRIX_INPUT_DIGITS,
    )
    right = _bounded_rational_matrix(
        source["right"],
        maximum_digits=_MAX_MATRIX_INPUT_DIGITS,
    )
    if left.ncols() != right.nrows():
        return False
    expected = left * right
    declared = _bounded_rational_matrix(
        result["product"],
        maximum_digits=_MAX_MATRIX_OUTPUT_DIGITS,
    )
    return (
        type(result["left_rows"]) is int
        and type(result["inner_dimension"]) is int
        and type(result["right_columns"]) is int
        and result["left_rows"] == left.nrows()
        and result["inner_dimension"] == left.ncols()
        and result["right_columns"] == right.ncols()
        and declared == expected
    )


def check_matrix_product(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="matrix.multiply.compute",
        witness_format="matrix.product.flint-replay",
        replay=_matrix_product,
    )


def _matrix_determinant(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if set(result) != {"determinant", "method"} or result["method"] != (
        "FRACTION_FREE_BAREISS"
    ):
        return False
    matrix = _matrix_source(source)
    if matrix.nrows() != matrix.ncols():
        return False
    return bool(_q(result["determinant"]) == matrix.det())


def check_matrix_determinant(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="matrix.determinant.compute",
        witness_format="matrix.determinant.flint-replay",
        replay=_matrix_determinant,
    )


def _matrix_rank(source: dict[str, Any], result: dict[str, Any]) -> bool:
    if (
        set(result) != {"rank", "pivot_columns", "method"}
        or result["method"] != "EXACT_RATIONAL_ROW_REDUCTION"
    ):
        return False
    matrix = _matrix_source(source)
    reduced, rank = matrix.rref()
    return (
        type(result["rank"]) is int
        and result["rank"] == rank
        and isinstance(result["pivot_columns"], list)
        and all(type(column) is int for column in result["pivot_columns"])
        and tuple(result["pivot_columns"])
        == tuple(
            next(column for column in range(reduced.ncols()) if reduced[row, column])
            for row in range(rank)
        )
    )


def check_matrix_rank(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="matrix.rank.compute",
        witness_format="matrix.rank.flint-replay",
        replay=_matrix_rank,
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
    "check_integer_powerful_number",
    "check_integer_prime_factorization",
    "check_matrix_characteristic_polynomial",
    "check_matrix_determinant",
    "check_matrix_nullspace",
    "check_matrix_product",
    "check_matrix_rank",
    "check_matrix_rref",
    "check_matrix_smith_normal_form",
    "check_modular_polynomial_residue_image",
    "check_polynomial_discriminant",
    "check_polynomial_gcd",
    "check_polynomial_resultant",
    "check_polynomial_square_free",
]
