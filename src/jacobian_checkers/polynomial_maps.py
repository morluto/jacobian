"""Independent exact replay for sparse rational polynomial-map claims."""

from __future__ import annotations

import re
from fractions import Fraction
from itertools import permutations
from typing import Any

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_VARIABLE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_MAX_DIMENSION = 4
_MAX_TERMS = 1024
_MAX_SOURCE_EXPONENT = 32
_MAX_DERIVED_EXPONENT = 4 * _MAX_SOURCE_EXPONENT - 1
_MAX_INTERMEDIATE_TERMS = 250_000

Exponent = tuple[int, ...]
Polynomial = dict[Exponent, Fraction]
ParsedPolynomial = tuple[tuple[Fraction, Exponent], ...]


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
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
        raise ValueError("rational integers are not canonical")
    parsed = Fraction(int(numerator), int(denominator))
    if str(parsed.numerator) != numerator or str(parsed.denominator) != denominator:
        raise ValueError("rational is not reduced and canonical")
    return parsed


def _parse_point(value: object, dimension: int) -> tuple[Fraction, ...]:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError("point dimension does not match the map")
    return tuple(_parse_rational(coordinate) for coordinate in value)


def _parse_map(
    value: object,
) -> tuple[int, tuple[str, ...], tuple[ParsedPolynomial, ...]]:
    if not isinstance(value, dict):
        raise ValueError("candidate map must be an object")
    if value.get("map_schema_version") != "1" or value.get("domain") != "QQ":
        raise ValueError("unsupported polynomial-map semantics")
    variables = value.get("variables")
    coordinates = value.get("coordinates")
    if (
        not isinstance(variables, list)
        or not 1 <= len(variables) <= _MAX_DIMENSION
        or any(
            not isinstance(variable, str) or _VARIABLE.fullmatch(variable) is None
            for variable in variables
        )
        or len(set(variables)) != len(variables)
        or not isinstance(coordinates, list)
        or len(coordinates) != len(variables)
    ):
        raise ValueError("malformed square rational polynomial map")
    dimension = len(variables)
    parsed_coordinates: list[ParsedPolynomial] = []
    for coordinate in coordinates:
        parsed_coordinates.append(
            _parse_polynomial(
                coordinate,
                dimension,
                maximum_exponent=_MAX_SOURCE_EXPONENT,
            )
        )
    return dimension, tuple(variables), tuple(parsed_coordinates)


def _parse_polynomial(
    value: object,
    dimension: int,
    *,
    maximum_exponent: int,
) -> ParsedPolynomial:
    if not isinstance(value, dict) or set(value) != {"terms"}:
        raise ValueError("polynomial coordinate must contain terms")
    terms = value["terms"]
    if not isinstance(terms, list) or len(terms) > _MAX_TERMS:
        raise ValueError("polynomial term list exceeds checker limits")
    parsed_terms: list[tuple[Fraction, Exponent]] = []
    seen: set[Exponent] = set()
    last: Exponent | None = None
    for term in terms:
        if not isinstance(term, dict) or set(term) != {
            "coefficient",
            "exponents",
        }:
            raise ValueError("malformed polynomial term")
        coefficient = _parse_rational(term["coefficient"])
        exponents = term["exponents"]
        if (
            coefficient == 0
            or not isinstance(exponents, list)
            or len(exponents) != dimension
            or any(
                not isinstance(exponent, int)
                or isinstance(exponent, bool)
                or not 0 <= exponent <= maximum_exponent
                for exponent in exponents
            )
        ):
            raise ValueError("invalid polynomial term")
        exponent_tuple = tuple(exponents)
        if exponent_tuple in seen or (last is not None and exponent_tuple >= last):
            raise ValueError("polynomial terms are not in canonical order")
        seen.add(exponent_tuple)
        last = exponent_tuple
        parsed_terms.append((coefficient, exponent_tuple))
    return tuple(parsed_terms)


def _evaluate(
    coordinates: tuple[ParsedPolynomial, ...],
    point: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    return tuple(
        sum(
            (
                coefficient * _monomial_value(point, exponents)
                for coefficient, exponents in polynomial
            ),
            start=Fraction(0),
        )
        for polynomial in coordinates
    )


def _monomial_value(
    point: tuple[Fraction, ...],
    exponents: tuple[int, ...],
) -> Fraction:
    result = Fraction(1)
    for value, exponent in zip(point, exponents, strict=True):
        result *= value**exponent
    return result


def check_collision(request: dict[str, Any]) -> dict[str, Any]:
    """Check that two distinct rational points have the same exact map image."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim_artifact = request["claim"]
        claim = claim_artifact["payload"]
        candidate_artifact = request["candidate"]
        if (
            not isinstance(claim, dict)
            or set(claim) != {"claim_schema_version", "predicate", "domain", "map_uri"}
            or claim.get("claim_schema_version") != "1"
            or claim.get("predicate") != "POLYNOMIAL_MAP_INJECTIVE"
            or claim.get("domain") != "QQ"
            or not isinstance(candidate_artifact, dict)
            or claim.get("map_uri") != candidate_artifact.get("artifact_uri")
        ):
            return _reject("unexpected polynomial-map claim")
        witness_artifact = request["witness"]
        witness = witness_artifact["payload"]
        if (
            not isinstance(witness, dict)
            or witness.get("evidence_schema_version") != "1"
            or witness.get("witness_format") != "polynomial.map_collision"
            or witness.get("format_version") != "1"
            or witness.get("role") != "REFUTES_CLAIM"
        ):
            return _reject("unexpected collision witness format or role")
        if witness.get("bindings") != request.get("expected_bindings"):
            return _reject("collision witness bindings do not match")
        dimension, _, coordinates = _parse_map(candidate_artifact["payload"])
        payload = witness.get("payload")
        if not isinstance(payload, dict) or set(payload) != {
            "first_point",
            "second_point",
            "image",
        }:
            return _reject("collision witness payload is malformed")
        first = _parse_point(payload["first_point"], dimension)
        second = _parse_point(payload["second_point"], dimension)
        declared_image = _parse_point(payload["image"], dimension)
        if first == second:
            return _reject("collision points are not distinct")
        first_image = _evaluate(coordinates, first)
        second_image = _evaluate(coordinates, second)
        if first_image != second_image or first_image != declared_image:
            return _reject("declared collision does not replay exactly")
        witness_uri = witness_artifact.get("artifact_uri")
        claim_uri = claim_artifact.get("artifact_uri")
        if not isinstance(witness_uri, str) or not isinstance(claim_uri, str):
            return _reject("collision relationship endpoints are unavailable")
        return {
            "accepted": True,
            "conclusion": "FALSE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "DIRECT_WITNESS",
            "coverage": "NOT_APPLICABLE",
            "detail": (
                "distinct rational points have the same exact polynomial-map image"
            ),
            "relation_id": "polynomial.relation.collision-refutes-injectivity",
            "relationship_source_artifact_uris": [witness_uri],
            "relationship_target_artifact_uris": [claim_uri],
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _reject("malformed polynomial-map collision request")


def check_collision_refutes_inverse(request: dict[str, Any]) -> dict[str, Any]:
    """Replay a collision whose exact consequence is no two-sided inverse."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim_artifact = request["claim"]
        claim = claim_artifact["payload"]
        candidate_artifact = request["candidate"]
        if (
            not isinstance(claim, dict)
            or set(claim) != {"claim_schema_version", "predicate", "domain", "map_uri"}
            or claim.get("claim_schema_version") != "1"
            or claim.get("predicate") != "POLYNOMIAL_MAP_NO_TWO_SIDED_INVERSE"
            or claim.get("domain") != "QQ"
            or not isinstance(candidate_artifact, dict)
            or claim.get("map_uri") != candidate_artifact.get("artifact_uri")
        ):
            return _reject("unexpected non-invertibility claim")
        witness_artifact = request["witness"]
        witness = witness_artifact["payload"]
        if (
            not isinstance(witness, dict)
            or witness.get("evidence_schema_version") != "1"
            or witness.get("witness_format")
            != "polynomial.map_collision_refutes_inverse"
            or witness.get("format_version") != "1"
            or witness.get("role") != "SUPPORTS_CLAIM"
        ):
            return _reject("unexpected inverse-obstruction witness format or role")
        if witness.get("bindings") != request.get("expected_bindings"):
            return _reject("inverse-obstruction witness bindings do not match")
        dimension, _, coordinates = _parse_map(candidate_artifact["payload"])
        payload = witness.get("payload")
        if not isinstance(payload, dict) or set(payload) != {
            "first_point",
            "second_point",
            "image",
        }:
            return _reject("inverse-obstruction witness payload is malformed")
        first = _parse_point(payload["first_point"], dimension)
        second = _parse_point(payload["second_point"], dimension)
        declared_image = _parse_point(payload["image"], dimension)
        if first == second:
            return _reject("collision points are not distinct")
        first_image = _evaluate(coordinates, first)
        second_image = _evaluate(coordinates, second)
        if first_image != second_image or first_image != declared_image:
            return _reject("declared collision does not replay exactly")
        witness_uri = witness_artifact.get("artifact_uri")
        claim_uri = claim_artifact.get("artifact_uri")
        if not isinstance(witness_uri, str) or not isinstance(claim_uri, str):
            return _reject("inverse-obstruction relationship endpoints are unavailable")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "DIRECT_WITNESS",
            "coverage": "NOT_APPLICABLE",
            "detail": ("a collision over QQ rules out a two-sided polynomial inverse"),
            "relation_id": "polynomial.relation.collision-refutes-two-sided-inverse",
            "relationship_source_artifact_uris": [witness_uri],
            "relationship_target_artifact_uris": [claim_uri],
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _reject("malformed inverse-obstruction witness request")


def check_identity(request: dict[str, Any]) -> dict[str, Any]:
    """Independently compare two canonical sparse polynomials over QQ."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim = request["claim"]["payload"]
        candidate = request["candidate"]
        scope = request["scope"]
        certificate = request["certificate"]["payload"]
        supporting_artifacts = request.get("supporting_artifacts", [])
        if (
            not isinstance(claim, dict)
            or claim.get("claim_schema_version") != "1"
            or claim.get("predicate") != "POLYNOMIAL_IDENTITY"
            or claim.get("domain") != "QQ"
            or not isinstance(candidate, dict)
            or not isinstance(scope, dict)
            or supporting_artifacts != []
            or claim.get("left_uri") != scope.get("artifact_uri")
            or claim.get("right_uri") != candidate.get("artifact_uri")
        ):
            return _reject("unexpected polynomial identity claim or artifact binding")
        variables = claim.get("variables")
        if (
            not isinstance(variables, list)
            or not 1 <= len(variables) <= _MAX_DIMENSION
            or any(
                not isinstance(variable, str) or _VARIABLE.fullmatch(variable) is None
                for variable in variables
            )
            or len(set(variables)) != len(variables)
        ):
            return _reject("invalid polynomial ring variables")
        if (
            not isinstance(certificate, dict)
            or certificate.get("certificate_type") != "polynomial.identity_replay"
            or certificate.get("format_version") != "1"
            or certificate.get("bindings") != request.get("expected_bindings")
            or certificate.get("payload")
            != {
                "method": "DIRECT_SPARSE_REPLAY",
                "variables": variables,
                "left_uri": scope["artifact_uri"],
                "right_uri": candidate["artifact_uri"],
            }
        ):
            return _reject("unexpected identity certificate or bindings")
        left_artifact = scope["payload"]
        right_artifact = candidate["payload"]
        if (
            not isinstance(left_artifact, dict)
            or not isinstance(right_artifact, dict)
            or left_artifact.get("polynomial_schema_version") != "1"
            or right_artifact.get("polynomial_schema_version") != "1"
            or left_artifact.get("domain") != "QQ"
            or right_artifact.get("domain") != "QQ"
            or left_artifact.get("variables") != variables
            or right_artifact.get("variables") != variables
        ):
            return _reject("polynomial artifacts do not match the declared ring")
        left = _as_polynomial(
            _parse_polynomial(
                left_artifact.get("polynomial"),
                len(variables),
                maximum_exponent=_MAX_DERIVED_EXPONENT,
            )
        )
        right = _as_polynomial(
            _parse_polynomial(
                right_artifact.get("polynomial"),
                len(variables),
                maximum_exponent=_MAX_DERIVED_EXPONENT,
            )
        )
        equal = left == right
        return {
            "accepted": True,
            "conclusion": "TRUE" if equal else "FALSE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "EXHAUSTIVE",
            "detail": (
                "polynomials have identical exact coefficients in the declared ring"
                if equal
                else "polynomials differ in at least one exact coefficient"
            ),
            **({"relation_id": "polynomial.relation.identity"} if equal else {}),
            **(
                {
                    "relationship_source_artifact_uris": [
                        scope["artifact_uri"],
                    ],
                    "relationship_target_artifact_uris": [
                        candidate["artifact_uri"],
                    ],
                }
                if equal
                else {}
            ),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _reject("malformed polynomial identity request")


def check_map_inverse(request: dict[str, Any]) -> dict[str, Any]:
    """Independently replay both polynomial-map compositions over QQ."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim_artifact = request["claim"]
        claim = claim_artifact["payload"]
        forward_artifact = request["scope"]
        residual_artifact = request["candidate"]
        certificate = request["certificate"]["payload"]
        supporting = request.get("supporting_artifacts", [])
        if (
            not isinstance(claim, dict)
            or claim.get("claim_schema_version") != "1"
            or claim.get("predicate") != "POLYNOMIAL_MAP_TWO_SIDED_INVERSE"
            or claim.get("domain") != "QQ"
            or not isinstance(forward_artifact, dict)
            or not isinstance(residual_artifact, dict)
            or not isinstance(supporting, list)
        ):
            return _reject("unexpected polynomial-map inverse claim")
        inverse_matches = [
            artifact
            for artifact in supporting
            if artifact.get("artifact_uri") == claim.get("inverse_map_uri")
        ]
        if len(inverse_matches) != 1:
            return _reject("inverse source artifact is missing or duplicated")
        inverse_artifact = inverse_matches[0]
        forward_uri = forward_artifact.get("artifact_uri")
        inverse_uri = inverse_artifact.get("artifact_uri")
        residual_uri = residual_artifact.get("artifact_uri")
        if (
            claim.get("forward_map_uri") != forward_uri
            or claim.get("inverse_map_uri") != inverse_uri
        ):
            return _reject("source map artifact bindings do not match")
        dimension, source_variables, forward = _parse_map(
            forward_artifact.get("payload")
        )
        inverse_dimension, target_variables, inverse = _parse_map(
            inverse_artifact.get("payload")
        )
        if (
            dimension != inverse_dimension
            or claim.get("source_variables") != list(source_variables)
            or claim.get("target_variables") != list(target_variables)
        ):
            return _reject("coefficient domain, dimension, or variable order mismatch")
        residual = residual_artifact.get("payload")
        if (
            not isinstance(residual, dict)
            or residual.get("residual_schema_version") != "1"
            or residual.get("domain") != "QQ"
            or residual.get("forward_map_uri") != forward_uri
            or residual.get("inverse_map_uri") != inverse_uri
            or residual.get("source_variables") != list(source_variables)
            or residual.get("target_variables") != list(target_variables)
        ):
            return _reject("composition residual artifact binding mismatch")
        replay_payload = certificate.get("payload")
        expected_records = residual.get(
            "inverse_after_forward_checker_records", []
        ) + residual.get("forward_after_inverse_checker_records", [])
        supporting_uris = [artifact.get("artifact_uri") for artifact in supporting]
        if (
            certificate.get("certificate_type")
            != "polynomial.map.inverse.two_sided_replay"
            or certificate.get("format_version") != "1"
            or certificate.get("bindings") != request.get("expected_bindings")
            or not isinstance(replay_payload, dict)
            or replay_payload.get("method") != "DIRECT_TWO_SIDED_SPARSE_REPLAY"
            or replay_payload.get("forward_map_uri") != forward_uri
            or replay_payload.get("inverse_map_uri") != inverse_uri
            or replay_payload.get("residuals_uri") != residual_uri
            or replay_payload.get("source_variables") != list(source_variables)
            or replay_payload.get("target_variables") != list(target_variables)
            or replay_payload.get("inverse_after_forward_checker_records")
            != residual.get("inverse_after_forward_checker_records")
            or replay_payload.get("forward_after_inverse_checker_records")
            != residual.get("forward_after_inverse_checker_records")
            or not isinstance(
                residual.get("inverse_after_forward_checker_records"), list
            )
            or len(residual["inverse_after_forward_checker_records"]) != dimension
            or not isinstance(
                residual.get("forward_after_inverse_checker_records"), list
            )
            or len(residual["forward_after_inverse_checker_records"]) != dimension
            or any(uri not in supporting_uris for uri in expected_records)
        ):
            return _reject("two-sided replay certificate or checker records mismatch")
        expected_left = _composition_residuals(
            outer=inverse, inner=forward, dimension=dimension
        )
        expected_right = _composition_residuals(
            outer=forward, inner=inverse, dimension=dimension
        )
        declared_left = _parse_residual_family(
            residual.get("inverse_after_forward"), dimension
        )
        declared_right = _parse_residual_family(
            residual.get("forward_after_inverse"), dimension
        )
        if declared_left != expected_left or declared_right != expected_right:
            return _reject("declared composition residuals do not replay exactly")
        inverse_holds = all(not polynomial for polynomial in expected_left) and all(
            not polynomial for polynomial in expected_right
        )
        return {
            "accepted": True,
            "conclusion": "TRUE" if inverse_holds else "FALSE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "EXHAUSTIVE",
            "detail": (
                "both exact polynomial compositions are identity"
                if inverse_holds
                else "at least one exact polynomial composition is not identity"
            ),
            **(
                {
                    "relation_id": "polynomial.relation.two-sided-inverse",
                    "relationship_source_artifact_uris": [forward_uri, inverse_uri],
                    "relationship_target_artifact_uris": [
                        claim_artifact["artifact_uri"]
                    ],
                }
                if inverse_holds
                else {}
            ),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _reject("malformed polynomial-map inverse replay request")


def _parse_residual_family(value: object, dimension: int) -> tuple[Polynomial, ...]:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError("residual family dimension mismatch")
    return tuple(
        _as_polynomial(
            _parse_polynomial(item, dimension, maximum_exponent=_MAX_DERIVED_EXPONENT)
        )
        for item in value
    )


def _composition_residuals(
    *,
    outer: tuple[ParsedPolynomial, ...],
    inner: tuple[ParsedPolynomial, ...],
    dimension: int,
) -> tuple[Polynomial, ...]:
    inner_polynomials = tuple(_as_polynomial(item) for item in inner)
    result: list[Polynomial] = []
    one = {(0,) * dimension: Fraction(1)}
    for coordinate_index, coordinate in enumerate(outer):
        composed: Polynomial = {}
        for coefficient, exponents in coordinate:
            term = one
            for polynomial, exponent in zip(inner_polynomials, exponents, strict=True):
                power = one
                for _ in range(exponent):
                    power = _multiply(power, polynomial)
                term = _multiply(term, power)
            for monomial, value in term.items():
                composed[monomial] = (
                    composed.get(monomial, Fraction(0)) + coefficient * value
                )
                if composed[monomial] == 0:
                    del composed[monomial]
        identity_exponent = tuple(
            1 if index == coordinate_index else 0 for index in range(dimension)
        )
        composed[identity_exponent] = composed.get(identity_exponent, Fraction(0)) - 1
        if composed[identity_exponent] == 0:
            del composed[identity_exponent]
        result.append(composed)
    return tuple(result)


def check_jacobian(request: dict[str, Any]) -> dict[str, Any]:
    """Replay a sparse polynomial Jacobian without importing SymPy."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim_artifact = request["claim"]
        candidate_artifact = request["candidate"]
        scope_artifact = request["scope"]
        certificate = request["certificate"]["payload"]
        if (
            not isinstance(claim_artifact, dict)
            or not isinstance(candidate_artifact, dict)
            or not isinstance(scope_artifact, dict)
            or not isinstance(certificate, dict)
        ):
            return _reject("Jacobian replay artifacts are malformed")
        claim = claim_artifact.get("payload")
        candidate = candidate_artifact.get("payload")
        source_map = scope_artifact.get("payload")
        if (
            not isinstance(claim, dict)
            or claim.get("claim_schema_version") != "1"
            or claim.get("predicate") != "EXACT_POLYNOMIAL_JACOBIAN"
        ):
            return _reject("unexpected polynomial Jacobian claim")
        if (
            certificate.get("evidence_schema_version") != "1"
            or certificate.get("certificate_type") != "polynomial.jacobian_replay"
            or certificate.get("format_version") != "1"
            or certificate.get("bindings") != request.get("expected_bindings")
        ):
            return _reject("unexpected Jacobian certificate format or bindings")
        payload = certificate.get("payload")
        if (
            not isinstance(payload, dict)
            or set(payload) != {"method", "source_map_uri", "jacobian_uri"}
            or payload.get("method") != "DIRECT_SPARSE_REPLAY"
        ):
            return _reject("Jacobian replay payload is malformed")
        source_map_uri = scope_artifact.get("artifact_uri")
        jacobian_uri = candidate_artifact.get("artifact_uri")
        if (
            claim.get("source_map_uri") != source_map_uri
            or payload.get("source_map_uri") != source_map_uri
            or payload.get("jacobian_uri") != jacobian_uri
        ):
            return _reject("Jacobian replay artifact identities do not match")
        dimension, variables, coordinates = _parse_map(source_map)
        matrix, determinant = _parse_jacobian_candidate(
            candidate,
            dimension=dimension,
            variables=variables,
            source_map_uri=source_map_uri,
        )
        expected_matrix = tuple(
            tuple(
                _differentiate(_as_polynomial(poly), column)
                for column in range(dimension)
            )
            for poly in coordinates
        )
        if matrix != expected_matrix:
            return _reject("declared Jacobian matrix does not replay exactly")
        if determinant != _determinant(expected_matrix, dimension):
            return _reject("declared Jacobian determinant does not replay exactly")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "NOT_APPLICABLE",
            "detail": (
                "Jacobian matrix and determinant replayed by independent sparse "
                "rational arithmetic"
            ),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _reject("malformed polynomial Jacobian replay request")


def check_keller_condition(request: dict[str, Any]) -> dict[str, Any]:
    """Replay a Jacobian and decide whether its determinant is nonzero constant."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim_artifact = request["claim"]
        candidate_artifact = request["candidate"]
        scope_artifact = request["scope"]
        certificate = request["certificate"]["payload"]
        if (
            not isinstance(claim_artifact, dict)
            or not isinstance(candidate_artifact, dict)
            or not isinstance(scope_artifact, dict)
            or not isinstance(certificate, dict)
        ):
            return _reject("Keller-condition replay artifacts are malformed")
        claim = claim_artifact.get("payload")
        if (
            not isinstance(claim, dict)
            or set(claim)
            != {
                "claim_schema_version",
                "predicate",
                "domain",
                "map_uri",
                "jacobian_uri",
            }
            or claim.get("claim_schema_version") != "1"
            or claim.get("predicate") != "POLYNOMIAL_MAP_KELLER_CONDITION"
            or claim.get("domain") != "QQ"
        ):
            return _reject("unexpected Keller-condition claim")
        payload = certificate.get("payload")
        if (
            certificate.get("evidence_schema_version") != "1"
            or certificate.get("certificate_type")
            != "polynomial.map.keller_condition.replay"
            or certificate.get("format_version") != "1"
            or certificate.get("bindings") != request.get("expected_bindings")
            or not isinstance(payload, dict)
            or set(payload) != {"method", "map_uri", "jacobian_uri"}
            or payload.get("method") != "DIRECT_SPARSE_KELLER_REPLAY"
        ):
            return _reject("unexpected Keller-condition certificate")
        map_uri = scope_artifact.get("artifact_uri")
        jacobian_uri = candidate_artifact.get("artifact_uri")
        if (
            claim.get("map_uri") != map_uri
            or claim.get("jacobian_uri") != jacobian_uri
            or payload.get("map_uri") != map_uri
            or payload.get("jacobian_uri") != jacobian_uri
        ):
            return _reject("Keller-condition artifact identities do not match")
        dimension, variables, coordinates = _parse_map(scope_artifact.get("payload"))
        matrix, determinant = _parse_jacobian_candidate(
            candidate_artifact.get("payload"),
            dimension=dimension,
            variables=variables,
            source_map_uri=map_uri,
        )
        expected_matrix = tuple(
            tuple(
                _differentiate(_as_polynomial(poly), column)
                for column in range(dimension)
            )
            for poly in coordinates
        )
        if matrix != expected_matrix:
            return _reject("declared Keller Jacobian matrix does not replay exactly")
        expected_determinant = _determinant(expected_matrix, dimension)
        if determinant != expected_determinant:
            return _reject("declared Keller determinant does not replay exactly")
        nonzero_constant = (
            len(determinant) == 1
            and (0,) * dimension in determinant
            and determinant[(0,) * dimension] != 0
        )
        return {
            "accepted": True,
            "conclusion": "TRUE" if nonzero_constant else "FALSE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "NOT_APPLICABLE",
            "detail": (
                "the exact Jacobian determinant is a nonzero constant"
                if nonzero_constant
                else "the exact Jacobian determinant is not a nonzero constant"
            ),
            **(
                {
                    "relation_id": "polynomial.relation.keller-condition",
                    "relationship_source_artifact_uris": [map_uri],
                    "relationship_target_artifact_uris": [jacobian_uri],
                }
                if nonzero_constant
                else {}
            ),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _reject("malformed Keller-condition replay request")


def _parse_jacobian_candidate(
    value: object,
    *,
    dimension: int,
    variables: tuple[str, ...],
    source_map_uri: object,
) -> tuple[tuple[tuple[Polynomial, ...], ...], Polynomial]:
    if not isinstance(value, dict) or set(value) != {
        "jacobian_schema_version",
        "map_uri",
        "variable_order",
        "matrix",
        "determinant",
        "backend",
        "backend_version",
    }:
        raise ValueError("malformed polynomial Jacobian candidate")
    matrix = value["matrix"]
    if (
        value["jacobian_schema_version"] != "1"
        or value["map_uri"] != source_map_uri
        or value["variable_order"] != list(variables)
        or value["backend"] != "sympy"
        or not isinstance(value["backend_version"], str)
        or not value["backend_version"]
        or not isinstance(matrix, list)
        or len(matrix) != dimension
        or any(not isinstance(row, list) or len(row) != dimension for row in matrix)
    ):
        raise ValueError("polynomial Jacobian metadata does not match the source")
    parsed_matrix = tuple(
        tuple(
            _as_polynomial(
                _parse_polynomial(
                    entry,
                    dimension,
                    maximum_exponent=_MAX_DERIVED_EXPONENT,
                )
            )
            for entry in row
        )
        for row in matrix
    )
    determinant = _as_polynomial(
        _parse_polynomial(
            value["determinant"],
            dimension,
            maximum_exponent=_MAX_DERIVED_EXPONENT,
        )
    )
    return parsed_matrix, determinant


def _as_polynomial(terms: ParsedPolynomial) -> Polynomial:
    return {exponents: coefficient for coefficient, exponents in terms}


def _differentiate(polynomial: Polynomial, variable: int) -> Polynomial:
    result: Polynomial = {}
    for exponents, coefficient in polynomial.items():
        exponent = exponents[variable]
        if exponent == 0:
            continue
        derived = list(exponents)
        derived[variable] -= 1
        result[tuple(derived)] = coefficient * exponent
    return result


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    if not left or not right:
        return {}
    if len(left) * len(right) > _MAX_INTERMEDIATE_TERMS:
        raise ValueError("polynomial replay exceeds the checker term budget")
    result: Polynomial = {}
    for left_exponents, left_coefficient in left.items():
        for right_exponents, right_coefficient in right.items():
            exponents = tuple(
                left_degree + right_degree
                for left_degree, right_degree in zip(
                    left_exponents, right_exponents, strict=True
                )
            )
            result[exponents] = (
                result.get(exponents, Fraction(0))
                + left_coefficient * right_coefficient
            )
            if result[exponents] == 0:
                del result[exponents]
            if len(result) > _MAX_INTERMEDIATE_TERMS:
                raise ValueError("polynomial replay exceeds the checker term budget")
    return result


def _add_scaled(
    target: Polynomial,
    source: Polynomial,
    scale: int,
) -> Polynomial:
    result = dict(target)
    for exponents, coefficient in source.items():
        result[exponents] = result.get(exponents, Fraction(0)) + scale * coefficient
        if result[exponents] == 0:
            del result[exponents]
    if len(result) > _MAX_INTERMEDIATE_TERMS:
        raise ValueError("polynomial replay exceeds the checker term budget")
    return result


def _determinant(
    matrix: tuple[tuple[Polynomial, ...], ...],
    dimension: int,
) -> Polynomial:
    result: Polynomial = {}
    one = {(0,) * dimension: Fraction(1)}
    for permutation in permutations(range(dimension)):
        term = one
        for row, column in enumerate(permutation):
            term = _multiply(term, matrix[row][column])
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(dimension)
            for right in range(left + 1, dimension)
        )
        result = _add_scaled(result, term, -1 if inversions % 2 else 1)
    return result
