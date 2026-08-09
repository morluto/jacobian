"""Independent replay for typed rational-polynomial normalization.

This checker intentionally uses only the Python standard library. It does not
import Jacobian contracts, SymPy, or producer code.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Any

from jacobian_checkers.bound_artifacts import valid_unscoped_unencoded_bindings

_ARTIFACT_URI = re.compile(r"^artifact://sha256/[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_VARIABLE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_ARTIFACT_KEYS = {
    "artifact_uri",
    "object_digest",
    "payload_digest",
    "schema_uri",
    "semantics_uri",
    "parents",
    "payload",
}
_EXPRESSION_BINDING_KEYS = {
    "binding_version",
    "expression_artifact_uri",
    "expression_object_digest",
    "expression_payload_digest",
    "variables",
    "node_count",
    "depth",
    "expanded_term_upper_bound",
    "coefficient_digit_budget",
}
_PROVIDER_KEYS = {
    "runtime_version",
    "provider",
    "availability",
    "version",
    "digest",
    "digest_kind",
    "platform",
    "install_tier",
    "license_id",
    "license_files",
    "features",
    "checker_ids",
    "configuration",
    "distribution_import_name",
    "distribution_required_attributes",
    "diagnostic",
}
_NORMALIZATION_CONFIGURATION = {
    "distribution": "sympy",
    "domain": "QQ",
    "operation": "Poly(expression, *variables, domain=QQ).terms()",
    "expression_schema_version": "1",
    "maximum_variables": 4,
    "maximum_nodes": 128,
    "maximum_depth": 16,
    "maximum_expanded_terms": 1024,
    "maximum_exponent_per_variable": 127,
    "maximum_coefficient_digit_budget": 4096,
}
_MAX_VARIABLES = 4
_MAX_NODES = 128
_MAX_DEPTH = 16
_MAX_OPERANDS = 16
_MAX_POWER = 32
_MAX_TERMS = 1024
_MAX_EXPONENT = 127
_MAX_INTEGER_DIGITS = 256
_MAX_COEFFICIENT_DIGIT_BUDGET = 4096
_Polynomial = dict[tuple[int, ...], Fraction]


@dataclass(frozen=True, slots=True)
class _Analysis:
    nodes: int
    depth: int
    term_upper_bound: int
    maximum_exponents: tuple[int, ...]
    coefficient_digit_budget: int


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _valid_artifact(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != _ARTIFACT_KEYS:
        return False
    if (
        not isinstance(value["artifact_uri"], str)
        or _ARTIFACT_URI.fullmatch(value["artifact_uri"]) is None
        or not isinstance(value["object_digest"], str)
        or _DIGEST.fullmatch(value["object_digest"]) is None
        or not isinstance(value["payload_digest"], str)
        or _DIGEST.fullmatch(value["payload_digest"]) is None
        or not isinstance(value["schema_uri"], str)
        or _ARTIFACT_URI.fullmatch(value["schema_uri"]) is None
        or not isinstance(value["semantics_uri"], str)
        or _ARTIFACT_URI.fullmatch(value["semantics_uri"]) is None
    ):
        return False
    parents = value["parents"]
    return (
        isinstance(parents, list)
        and len(parents) == len(set(parents))
        and all(
            isinstance(parent, str) and _ARTIFACT_URI.fullmatch(parent) is not None
            for parent in parents
        )
    )


def _canonical_integer(value: object) -> int:
    if (
        not isinstance(value, str)
        or _INTEGER.fullmatch(value) is None
        or len(value.lstrip("-")) > _MAX_INTEGER_DIGITS
    ):
        raise ValueError("value is not a bounded canonical integer")
    result = int(value)
    if str(result) != value:
        raise ValueError("integer is not canonical")
    return result


def _rational(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational literal is malformed")
    numerator = _canonical_integer(value["num"])
    denominator = _canonical_integer(value["den"])
    if denominator <= 0:
        raise ValueError("rational denominator must be positive")
    result = Fraction(numerator, denominator)
    if str(result.numerator) != value["num"] or str(result.denominator) != value["den"]:
        raise ValueError("rational literal is not reduced and canonical")
    return result


def _expression_artifact(
    payload: object,
) -> tuple[list[str], _Polynomial, _Analysis]:
    if not isinstance(payload, dict) or set(payload) != {
        "expression_schema_version",
        "domain",
        "variables",
        "expression",
    }:
        raise ValueError("typed polynomial expression has an invalid shape")
    variables = payload["variables"]
    if (
        payload["expression_schema_version"] != "1"
        or payload["domain"] != "QQ"
        or not isinstance(variables, list)
        or not 1 <= len(variables) <= _MAX_VARIABLES
        or any(
            not isinstance(variable, str) or _VARIABLE.fullmatch(variable) is None
            for variable in variables
        )
        or len(set(variables)) != len(variables)
    ):
        raise ValueError("typed polynomial expression has an invalid ring")
    variable_indices = {variable: index for index, variable in enumerate(variables)}
    polynomial, analysis = _expression(payload["expression"], variable_indices)
    _require_analysis_bounds(analysis)
    return variables, polynomial, analysis


def _expression_rational(
    value: dict[str, Any],
    zero_exponents: tuple[int, ...],
) -> tuple[_Polynomial, _Analysis]:
    if set(value) != {"kind", "value"}:
        raise ValueError("rational node is malformed")
    coefficient = _rational(value["value"])
    literal_polynomial = {} if coefficient == 0 else {zero_exponents: coefficient}
    analysis = _Analysis(
        nodes=1,
        depth=1,
        term_upper_bound=int(coefficient != 0),
        maximum_exponents=zero_exponents,
        coefficient_digit_budget=(
            len(value["value"]["num"].lstrip("-")) + len(value["value"]["den"])
        ),
    )
    return literal_polynomial, analysis


def _expression_variable(
    value: dict[str, Any],
    variable_indices: dict[str, int],
    dimension: int,
) -> tuple[_Polynomial, _Analysis]:
    if set(value) != {"kind", "name"}:
        raise ValueError("variable node is malformed")
    name = value["name"]
    if not isinstance(name, str) or name not in variable_indices:
        raise ValueError("expression uses an undeclared variable")
    exponents = [0] * dimension
    exponents[variable_indices[name]] = 1
    return {tuple(exponents): Fraction(1)}, _Analysis(
        1,
        1,
        1,
        tuple(exponents),
        1,
    )


def _expression_negate(
    value: dict[str, Any],
    variable_indices: dict[str, int],
) -> tuple[_Polynomial, _Analysis]:
    if set(value) != {"kind", "operand"}:
        raise ValueError("negate node is malformed")
    operand, child = _expression(value["operand"], variable_indices)
    return {monomial: -coefficient for monomial, coefficient in operand.items()}, (
        _Analysis(
            1 + child.nodes,
            1 + child.depth,
            child.term_upper_bound,
            child.maximum_exponents,
            child.coefficient_digit_budget,
        )
    )


def _expression_power(
    value: dict[str, Any],
    variable_indices: dict[str, int],
    dimension: int,
    zero_exponents: tuple[int, ...],
) -> tuple[_Polynomial, _Analysis]:
    if set(value) != {"kind", "base", "exponent"}:
        raise ValueError("power node is malformed")
    exponent = value["exponent"]
    if type(exponent) is not int or not 0 <= exponent <= _MAX_POWER:
        raise ValueError("power exponent is outside the declared bound")
    base, child = _expression(value["base"], variable_indices)
    if exponent == 0:
        terms = 1
        degrees = zero_exponents
        digit_budget = 1
    elif child.term_upper_bound == 0:
        terms = 0
        degrees = zero_exponents
        digit_budget = child.coefficient_digit_budget * exponent
    else:
        terms = min(
            comb(child.term_upper_bound + exponent - 1, exponent),
            _MAX_TERMS + 1,
        )
        degrees = tuple(
            child_exponent * exponent for child_exponent in child.maximum_exponents
        )
        digit_budget = child.coefficient_digit_budget * exponent
    analysis = _Analysis(
        1 + child.nodes,
        1 + child.depth,
        terms,
        degrees,
        digit_budget,
    )
    _require_analysis_bounds(analysis)
    return _power(base, exponent, dimension), analysis


def _expression_variadic(
    value: dict[str, Any],
    variable_indices: dict[str, int],
    dimension: int,
    zero_exponents: tuple[int, ...],
    kind: str,
) -> tuple[_Polynomial, _Analysis]:
    if set(value) != {"kind", "operands"}:
        raise ValueError("variadic expression node is malformed")
    operands = value["operands"]
    if not isinstance(operands, list) or not 2 <= len(operands) <= _MAX_OPERANDS:
        raise ValueError("variadic expression node has invalid arity")
    children = tuple(_expression(operand, variable_indices) for operand in operands)
    analyses = tuple(child[1] for child in children)
    nodes = 1 + sum(child.nodes for child in analyses)
    depth = 1 + max(child.depth for child in analyses)
    if kind == "add":
        term_bound = _bounded_sum(child.term_upper_bound for child in analyses)
        degrees = tuple(
            max(child.maximum_exponents[index] for child in analyses)
            for index in range(dimension)
        )
        digit_budget = sum(child.coefficient_digit_budget for child in analyses) + len(
            analyses
        )
        combined_polynomial: _Polynomial = {}
        for child_polynomial, _ in children:
            combined_polynomial = _add(combined_polynomial, child_polynomial)
    else:
        if any(child.term_upper_bound == 0 for child in analyses):
            term_bound = 0
            degrees = zero_exponents
        else:
            term_bound = _bounded_product(child.term_upper_bound for child in analyses)
            degrees = tuple(
                sum(child.maximum_exponents[index] for child in analyses)
                for index in range(dimension)
            )
        digit_budget = sum(child.coefficient_digit_budget for child in analyses)
        combined_polynomial = {zero_exponents: Fraction(1)}
        for child_polynomial, _ in children:
            combined_polynomial = _multiply(
                combined_polynomial,
                child_polynomial,
            )
    analysis = _Analysis(nodes, depth, term_bound, degrees, digit_budget)
    _require_analysis_bounds(analysis)
    return combined_polynomial, analysis


def _expression(
    value: object,
    variable_indices: dict[str, int],
) -> tuple[_Polynomial, _Analysis]:
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise ValueError("expression node is malformed")
    dimension = len(variable_indices)
    zero_exponents = (0,) * dimension
    kind = value["kind"]
    if kind == "rational":
        return _expression_rational(value, zero_exponents)
    if kind == "variable":
        return _expression_variable(value, variable_indices, dimension)
    if kind == "negate":
        return _expression_negate(value, variable_indices)
    if kind == "power":
        return _expression_power(value, variable_indices, dimension, zero_exponents)
    if kind in {"add", "multiply"}:
        return _expression_variadic(
            value, variable_indices, dimension, zero_exponents, kind
        )
    raise ValueError("expression node kind is unsupported")


def _require_analysis_bounds(analysis: _Analysis) -> None:
    if analysis.nodes > _MAX_NODES or analysis.depth > _MAX_DEPTH:
        raise ValueError("expression AST exceeds its structural bound")
    if analysis.term_upper_bound > _MAX_TERMS:
        raise ValueError("expression exceeds its expansion budget")
    if any(exponent > _MAX_EXPONENT for exponent in analysis.maximum_exponents):
        raise ValueError("expression exponent exceeds its ring bound")
    if analysis.coefficient_digit_budget > _MAX_COEFFICIENT_DIGIT_BUDGET:
        raise ValueError("expression exceeds its coefficient digit budget")


def _bounded_sum(values: Any) -> int:
    total = 0
    for value in values:
        total += value
        if total > _MAX_TERMS:
            return _MAX_TERMS + 1
    return total


def _bounded_product(values: Any) -> int:
    total = 1
    for value in values:
        total *= value
        if total > _MAX_TERMS:
            return _MAX_TERMS + 1
    return total


def _add(left: _Polynomial, right: _Polynomial) -> _Polynomial:
    result = dict(left)
    for monomial, coefficient in right.items():
        combined = result.get(monomial, Fraction(0)) + coefficient
        if combined:
            result[monomial] = combined
        else:
            result.pop(monomial, None)
    if len(result) > _MAX_TERMS:
        raise ValueError("expanded polynomial exceeds the term budget")
    return result


def _multiply(left: _Polynomial, right: _Polynomial) -> _Polynomial:
    if not left or not right:
        return {}
    result: _Polynomial = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            monomial = tuple(
                left_exponent + right_exponent
                for left_exponent, right_exponent in zip(
                    left_monomial,
                    right_monomial,
                    strict=True,
                )
            )
            if any(exponent > _MAX_EXPONENT for exponent in monomial):
                raise ValueError("expanded exponent exceeds the ring bound")
            coefficient = (
                result.get(monomial, Fraction(0)) + left_coefficient * right_coefficient
            )
            if coefficient:
                result[monomial] = coefficient
            else:
                result.pop(monomial, None)
            if len(result) > _MAX_TERMS:
                raise ValueError("expanded polynomial exceeds the term budget")
    return result


def _power(base: _Polynomial, exponent: int, dimension: int) -> _Polynomial:
    result: _Polynomial = {(0,) * dimension: Fraction(1)}
    for _ in range(exponent):
        result = _multiply(result, base)
    return result


def _normalized_polynomial(payload: object, dimension: int) -> _Polynomial:
    if not isinstance(payload, dict) or set(payload) != {"terms"}:
        raise ValueError("normalized sparse polynomial is malformed")
    terms = payload["terms"]
    if not isinstance(terms, list) or len(terms) > _MAX_TERMS:
        raise ValueError("normalized sparse polynomial has too many terms")
    result: _Polynomial = {}
    ordered_exponents: list[tuple[int, ...]] = []
    for term in terms:
        if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
            raise ValueError("normalized polynomial term is malformed")
        coefficient = _rational(term["coefficient"])
        exponents = term["exponents"]
        if (
            coefficient == 0
            or not isinstance(exponents, list)
            or len(exponents) != dimension
            or any(
                type(exponent) is not int or not 0 <= exponent <= _MAX_EXPONENT
                for exponent in exponents
            )
        ):
            raise ValueError("normalized polynomial term is invalid")
        monomial = tuple(exponents)
        if monomial in result:
            raise ValueError("normalized polynomial contains duplicate monomials")
        result[monomial] = coefficient
        ordered_exponents.append(monomial)
    if ordered_exponents != sorted(ordered_exponents, reverse=True):
        raise ValueError("normalized polynomial term order is not canonical")
    return result


def _validate_candidate(
    payload: object,
    *,
    claim: dict[str, Any],
    candidate: dict[str, Any],
    variables: list[str],
    analysis: _Analysis,
) -> _Polynomial:
    if not isinstance(payload, dict) or set(payload) != {
        "normalization_schema_version",
        "source",
        "declared_scope",
        "normalized",
        "producer",
        "resource_budget",
        "method",
    }:
        raise ValueError("normalization candidate has an invalid shape")
    if (
        payload["normalization_schema_version"] != "1"
        or payload["declared_scope"] != "FULL_EXPRESSION"
        or payload["method"] != "SYMPY_POLY_QQ_CANONICAL_TERMS"
    ):
        raise ValueError("normalization candidate uses unsupported semantics")
    binding = payload["source"]
    if not isinstance(binding, dict) or set(binding) != _EXPRESSION_BINDING_KEYS:
        raise ValueError("normalization source binding is malformed")
    if binding != {
        "binding_version": "1",
        "expression_artifact_uri": claim["artifact_uri"],
        "expression_object_digest": claim["object_digest"],
        "expression_payload_digest": claim["payload_digest"],
        "variables": variables,
        "node_count": analysis.nodes,
        "depth": analysis.depth,
        "expanded_term_upper_bound": analysis.term_upper_bound,
        "coefficient_digit_budget": analysis.coefficient_digit_budget,
    }:
        raise ValueError("normalization is not exactly bound to its source expression")
    if claim["artifact_uri"] not in candidate["parents"]:
        raise ValueError("normalization is missing source-expression lineage")
    provider = payload["producer"]
    if (
        not isinstance(provider, dict)
        or set(provider) != _PROVIDER_KEYS
        or provider["runtime_version"] != "1"
        or provider["provider"] != "jacobian.sympy"
        or provider["availability"] != "AVAILABLE"
        or provider["version"] != "1.14.0"
        or not isinstance(provider["digest"], str)
        or _DIGEST.fullmatch(provider["digest"]) is None
        or provider["digest_kind"] != "PYTHON_DISTRIBUTION_RECORD"
        or provider["install_tier"] != "T0"
        or provider["license_id"] != "BSD-3-Clause"
        or provider["configuration"] != _NORMALIZATION_CONFIGURATION
    ):
        raise ValueError("normalization producer identity is malformed")
    budget = payload["resource_budget"]
    if (
        not isinstance(budget, dict)
        or set(budget) != {"budget_version", "wall_seconds"}
        or budget["budget_version"] != "1"
        or type(budget["wall_seconds"]) is not int
        or not 1 <= budget["wall_seconds"] <= 60
    ):
        raise ValueError("normalization resource budget is malformed")
    return _normalized_polynomial(payload["normalized"], len(variables))


_NORMALIZATION_REQUEST_KEYS = {
    "request_version",
    "claim",
    "candidate",
    "scope",
    "witness",
    "expected_bindings",
}
_NORMALIZATION_WITNESS_ENVELOPE_KEYS = {
    "evidence_schema_version",
    "witness_format",
    "format_version",
    "role",
    "bindings",
    "payload",
}


def _check_normalization_request_envelope(request: object) -> str | None:
    if not isinstance(request, dict) or set(request) != _NORMALIZATION_REQUEST_KEYS:
        return "malformed checker request"
    if request["request_version"] != "1" or request["scope"] is not None:
        return "unsupported checker request"
    return None


def _check_normalization_artifacts(
    claim: dict[str, Any],
    candidate: dict[str, Any],
    witness: dict[str, Any],
    expected_bindings: object,
) -> str | None:
    if not all(_valid_artifact(item) for item in (claim, candidate, witness)):
        return "checker artifact metadata is malformed"
    if not valid_unscoped_unencoded_bindings(expected_bindings):
        return "expected evidence bindings are malformed"
    if (
        claim["semantics_uri"] != candidate["semantics_uri"]
        or claim["semantics_uri"] != witness["semantics_uri"]
    ):
        return "checker artifacts use different semantics"
    return None


def _check_normalization_digests(
    claim: dict[str, Any],
    candidate: dict[str, Any],
    witness: dict[str, Any],
) -> str | None:
    for artifact, label in (
        (claim, "source expression"),
        (candidate, "normalization candidate"),
        (witness, "normalization witness"),
    ):
        if artifact["payload_digest"] != _sha256(_canonical_json(artifact["payload"])):
            return f"{label} payload digest does not match"
    return None


def _check_normalization_binding_match(
    expected_bindings: dict[str, Any],
    claim: dict[str, Any],
    candidate: dict[str, Any],
) -> str | None:
    if (
        expected_bindings["claim_digest"] != claim["object_digest"]
        or expected_bindings["candidate_digest"] != candidate["object_digest"]
    ):
        return "expected evidence bindings do not match artifacts"
    return None


def _check_normalization_witness_envelope(
    envelope: object,
    expected_bindings: object,
) -> str | None:
    if (
        not isinstance(envelope, dict)
        or set(envelope) != _NORMALIZATION_WITNESS_ENVELOPE_KEYS
    ):
        return "normalization witness envelope is malformed"
    if (
        envelope["evidence_schema_version"] != "1"
        or envelope["witness_format"] != "polynomial.expression_normalization"
        or envelope["format_version"] != "1"
        or envelope["role"] != "SUPPORTS_CLAIM"
        or envelope["bindings"] != expected_bindings
    ):
        return "normalization witness is not exactly bound"
    return None


def _check_normalization_witness_payload(
    witness: dict[str, Any],
    claim: dict[str, Any],
    candidate: dict[str, Any],
) -> str | None:
    if witness["payload"]["payload"] != {
        "expression_uri": claim["artifact_uri"],
        "normalization_uri": candidate["artifact_uri"],
    }:
        return "normalization witness points at different artifacts"
    if not {
        claim["artifact_uri"],
        candidate["artifact_uri"],
    }.issubset(set(witness["parents"])):
        return "normalization witness is missing required lineage"
    return None


def _check_normalization_witness(
    witness: dict[str, Any],
    expected_bindings: object,
    claim: dict[str, Any],
    candidate: dict[str, Any],
) -> str | None:
    error = _check_normalization_witness_envelope(witness["payload"], expected_bindings)
    if error is not None:
        return error
    return _check_normalization_witness_payload(witness, claim, candidate)


def check_polynomial_expression_normalization(
    request: dict[str, Any],
) -> dict[str, Any]:
    """Accept only exact coefficient equality with the full typed AST."""

    try:
        error = _check_normalization_request_envelope(request)
        if error is not None:
            return _reject(error)
        claim = request["claim"]
        candidate = request["candidate"]
        witness = request["witness"]
        expected_bindings = request["expected_bindings"]
        error = _check_normalization_artifacts(
            claim, candidate, witness, expected_bindings
        )
        if error is not None:
            return _reject(error)
        error = _check_normalization_digests(claim, candidate, witness)
        if error is not None:
            return _reject(error)

        variables, expanded, analysis = _expression_artifact(claim["payload"])
        normalized = _validate_candidate(
            candidate["payload"],
            claim=claim,
            candidate=candidate,
            variables=variables,
            analysis=analysis,
        )
        error = _check_normalization_binding_match(expected_bindings, claim, candidate)
        if error is not None:
            return _reject(error)
        error = _check_normalization_witness(
            witness, expected_bindings, claim, candidate
        )
        if error is not None:
            return _reject(error)
        if expanded != normalized:
            return _reject(
                "canonical coefficients do not equal the full typed expression"
            )
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "EXHAUSTIVE_FINITE",
            "coverage": "EXHAUSTIVE",
            "relation_id": "polynomial.relation.expression-normalization-of",
            "relationship_source_artifact_uris": [candidate["artifact_uri"]],
            "relationship_target_artifact_uris": [claim["artifact_uri"]],
            "detail": (
                f"independently expanded all {analysis.nodes} AST nodes and matched "
                f"{len(normalized)} canonical coefficients over QQ"
            ),
        }
    except (
        ArithmeticError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return _reject("malformed polynomial normalization request")
