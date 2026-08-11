"""Clean-room exact verifier copied into every symbolic-coordination-v1 task.

This module does not import Jacobian, a mathematical backend, the dataset
generator, or Oracle solution code. Its only external dependency is the
maintained JSON Schema validator used by the task-local protocol helper.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    SUBMISSION_FIELDS,
    is_regular_bounded_file,
    load_submission_raw,
    normalize_reward_file,
    read_evidence_json,
    submission_matches_public_schema,
    workspace_input_is_bound,
)

W = Path("/app")
T = Path("/tests")
MAX_TERMS = 128
MAX_INTERMEDIATE_TERMS = 4096
CHECKER_ID = "symbolic-coordination-v1.clean-room-polynomial-map-checker@1"
SEMANTICS_ID = "exact-sparse-polynomial-maps-over-QQ@1"
ASSURANCE_LEVELS = {"UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED"}
SCOREABLE_ASSURANCE_LEVELS = {"UNVERIFIED", "COMPUTED", "CHECKED"}


def _load_json(path: Path, *, maximum_bytes: int) -> object | None:
    if not is_regular_bounded_file(path, max_bytes=maximum_bytes):
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return None


def _sha256_object(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _fraction(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational must contain exactly num and den")
    numerator = value["num"]
    denominator = value["den"]
    if type(numerator) is not str or type(denominator) is not str:
        raise ValueError("rational components must be strings")
    parsed = Fraction(int(numerator), int(denominator))
    return parsed


def _rational(value: Fraction) -> dict[str, str]:
    return {"num": str(value.numerator), "den": str(value.denominator)}


def _poly(
    value: object,
    dimension: int,
    *,
    canonical: bool,
) -> dict[tuple[int, ...], Fraction]:
    if not isinstance(value, dict) or set(value) != {"terms"}:
        raise ValueError("polynomial must contain exactly terms")
    terms = value["terms"]
    if not isinstance(terms, list) or len(terms) > MAX_TERMS:
        raise ValueError("polynomial term list is malformed or too large")
    result: dict[tuple[int, ...], Fraction] = {}
    previous: tuple[int, ...] | None = None
    for term in terms:
        if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
            raise ValueError("term is malformed")
        coefficient = _fraction(term["coefficient"])
        exponents = term["exponents"]
        if (
            not isinstance(exponents, list)
            or len(exponents) != dimension
            or any(
                type(exponent) is not int or not 0 <= exponent <= 32
                for exponent in exponents
            )
        ):
            raise ValueError("term exponent vector is malformed")
        key = tuple(exponents)
        if canonical and (
            coefficient == 0
            or key in result
            or (previous is not None and key >= previous)
        ):
            raise ValueError("canonical polynomial term contract is violated")
        result[key] = result.get(key, Fraction(0)) + coefficient
        if result[key] == 0:
            del result[key]
        previous = key
    return result


def _encoded_poly(poly: dict[tuple[int, ...], Fraction]) -> dict[str, object]:
    return {
        "terms": [
            {"coefficient": _rational(poly[exponents]), "exponents": list(exponents)}
            for exponents in sorted(poly, reverse=True)
        ]
    }


def _map(
    value: object,
    *,
    canonical: bool,
) -> tuple[tuple[str, ...], tuple[dict[tuple[int, ...], Fraction], ...]]:
    if not isinstance(value, dict) or set(value) != {
        "map_schema_version",
        "domain",
        "variables",
        "coordinates",
    }:
        raise ValueError("map object is malformed")
    variables = value["variables"]
    coordinates = value["coordinates"]
    if (
        value["map_schema_version"] != "1"
        or value["domain"] != "QQ"
        or not isinstance(variables, list)
        or not 1 <= len(variables) <= 3
        or any(type(variable) is not str or not variable for variable in variables)
        or len(set(variables)) != len(variables)
        or not isinstance(coordinates, list)
        or len(coordinates) != len(variables)
    ):
        raise ValueError("unsupported map semantics")
    return tuple(variables), tuple(
        _poly(coordinate, len(variables), canonical=canonical)
        for coordinate in coordinates
    )


def _encoded_map(
    variables: tuple[str, ...],
    coordinates: tuple[dict[tuple[int, ...], Fraction], ...],
) -> dict[str, object]:
    return {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": list(variables),
        "coordinates": [_encoded_poly(coordinate) for coordinate in coordinates],
    }


def _add(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    result = dict(left)
    for exponents, coefficient in right.items():
        result[exponents] = result.get(exponents, Fraction(0)) + coefficient
        if result[exponents] == 0:
            del result[exponents]
    return result


def _scale(
    poly: dict[tuple[int, ...], Fraction], coefficient: Fraction
) -> dict[tuple[int, ...], Fraction]:
    return {
        exponents: coefficient * value
        for exponents, value in poly.items()
        if coefficient * value
    }


def _multiply(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for left_exponents, left_coefficient in left.items():
        for right_exponents, right_coefficient in right.items():
            key = tuple(
                a + b for a, b in zip(left_exponents, right_exponents, strict=True)
            )
            result[key] = (
                result.get(key, Fraction(0)) + left_coefficient * right_coefficient
            )
            if result[key] == 0:
                del result[key]
            if len(result) > MAX_INTERMEDIATE_TERMS:
                raise ValueError("polynomial intermediate exceeds verifier budget")
    return result


def _power(poly: dict[tuple[int, ...], Fraction], exponent: int, dimension: int):
    result = {(0,) * dimension: Fraction(1)}
    base = poly
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _multiply(result, base)
        remaining //= 2
        if remaining:
            base = _multiply(base, base)
    return result


def _compose(
    outer: tuple[dict[tuple[int, ...], Fraction], ...],
    inner: tuple[dict[tuple[int, ...], Fraction], ...],
) -> tuple[dict[tuple[int, ...], Fraction], ...]:
    dimension = len(inner)
    composed = []
    for coordinate in outer:
        total: dict[tuple[int, ...], Fraction] = {}
        for exponents, coefficient in coordinate.items():
            term = {(0,) * dimension: coefficient}
            for inner_coordinate, exponent in zip(inner, exponents, strict=True):
                term = _multiply(term, _power(inner_coordinate, exponent, dimension))
            total = _add(total, term)
        composed.append(total)
    return tuple(composed)


def _identity_residuals(
    composed: tuple[dict[tuple[int, ...], Fraction], ...],
) -> tuple[dict[tuple[int, ...], Fraction], ...]:
    dimension = len(composed)
    residuals = []
    for index, coordinate in enumerate(composed):
        identity_exponents = tuple(int(i == index) for i in range(dimension))
        residuals.append(_add(coordinate, {identity_exponents: Fraction(-1)}))
    return tuple(residuals)


def _derivative(
    poly: dict[tuple[int, ...], Fraction], variable: int
) -> dict[tuple[int, ...], Fraction]:
    result = {}
    for exponents, coefficient in poly.items():
        if exponents[variable] == 0:
            continue
        derived = list(exponents)
        derived[variable] -= 1
        result[tuple(derived)] = coefficient * exponents[variable]
    return result


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _jacobian_determinant(
    coordinates: tuple[dict[tuple[int, ...], Fraction], ...],
) -> dict[tuple[int, ...], Fraction]:
    dimension = len(coordinates)
    matrix = tuple(
        tuple(_derivative(coordinate, column) for column in range(dimension))
        for coordinate in coordinates
    )
    determinant: dict[tuple[int, ...], Fraction] = {}
    for permutation in itertools.permutations(range(dimension)):
        product = {(0,) * dimension: Fraction(_permutation_sign(permutation))}
        for row, column in enumerate(permutation):
            product = _multiply(product, matrix[row][column])
        determinant = _add(determinant, product)
    return determinant


def _point(value: object, dimension: int) -> tuple[Fraction, ...]:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError("point dimension mismatch")
    return tuple(_fraction(coordinate) for coordinate in value)


def _encoded_point(point: tuple[Fraction, ...]) -> list[dict[str, str]]:
    return [_rational(coordinate) for coordinate in point]


def _evaluate(
    coordinates: tuple[dict[tuple[int, ...], Fraction], ...],
    point: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    image = []
    for coordinate in coordinates:
        value = Fraction(0)
        for exponents, coefficient in coordinate.items():
            monomial = coefficient
            for coordinate_value, exponent in zip(point, exponents, strict=True):
                monomial *= coordinate_value**exponent
            value += monomial
        image.append(value)
    return tuple(image)


def _grid(
    record: dict[str, object], dimension: int
) -> tuple[tuple[Fraction, ...], ...]:
    lower = record.get("min_numerator")
    upper = record.get("max_numerator")
    denominator = record.get("max_denominator")
    if type(lower) is not int or type(upper) is not int or type(denominator) is not int:
        raise ValueError("grid bounds must be integers")
    if denominator != 1 or lower > upper or upper - lower > 8:
        raise ValueError("unsupported pilot grid")
    values = tuple(Fraction(value) for value in range(lower, upper + 1))
    return tuple(itertools.product(values, repeat=dimension))


def _computed_bindings(data: dict[str, object]) -> dict[str, str]:
    case_type = data["case_type"]
    subject = (
        {
            "candidate_inverse": data["candidate_inverse"],
            "supplied_evidence": data.get("supplied_evidence"),
        }
        if case_type == "inverse"
        else data["supplied_certificate"]
        if case_type == "keller"
        else data["search_record"]
    )
    return {
        "binding_schema_version": "1",
        "claim_id": str(data["claim_id"]),
        "semantics_id": SEMANTICS_ID,
        "scope_id": str(data["scope_id"]),
        "forward_map_sha256": _sha256_object(data["forward_map"]),
        "subject_sha256": _sha256_object(subject),
        "checker_id": CHECKER_ID,
    }


def _inverse_assessment(data: dict[str, object], certificate: object):
    if not isinstance(certificate, dict) or set(certificate) != {
        "kind",
        "source_variables",
        "target_variables",
        "inverse_map",
        "inverse_after_forward_residuals",
        "forward_after_inverse_residuals",
        "checked_directions",
    }:
        return False, False, None
    if certificate.get("kind") != "TWO_SIDED_COMPOSITION_REPLAY":
        return False, False, None
    source_variables, forward = _map(data["forward_map"], canonical=False)
    target_variables, inverse = _map(data["candidate_inverse"], canonical=False)
    submitted_variables, submitted_inverse = _map(
        certificate.get("inverse_map"), canonical=False
    )
    left = _identity_residuals(_compose(inverse, forward))
    right = _identity_residuals(_compose(forward, inverse))
    submitted_left = tuple(
        _poly(value, len(source_variables), canonical=False)
        for value in certificate.get("inverse_after_forward_residuals", [])
    )
    submitted_right = tuple(
        _poly(value, len(target_variables), canonical=False)
        for value in certificate.get("forward_after_inverse_residuals", [])
    )
    shape = bool(
        certificate.get("source_variables") == list(source_variables)
        and certificate.get("target_variables") == list(target_variables)
        and submitted_variables == target_variables
        and len(submitted_left) == len(source_variables)
        and len(submitted_right) == len(target_variables)
        and certificate.get("checked_directions")
        == ["INVERSE_AFTER_FORWARD", "FORWARD_AFTER_INVERSE"]
    )
    holds = all(not residual for residual in left) and all(
        not residual for residual in right
    )
    expected_verdict = (
        "VALID_TWO_SIDED_INVERSE" if holds else "INVALID_INVERSE_CANDIDATE"
    )
    correct = bool(
        shape
        and submitted_inverse == inverse
        and submitted_left == left
        and submitted_right == right
    )
    return shape, correct, expected_verdict


def _keller_assessment(data: dict[str, object], certificate: object):
    if not isinstance(certificate, dict) or set(certificate) != {
        "kind",
        "variable_order",
        "determinant",
        "keller_condition",
        "global_invertibility",
    }:
        return False, False, None
    if certificate.get("kind") != "KELLER_DETERMINANT_REPLAY":
        return False, False, None
    variables, forward = _map(data["forward_map"], canonical=False)
    determinant = _jacobian_determinant(forward)
    submitted = _poly(certificate.get("determinant"), len(variables), canonical=False)
    constant_key = (0,) * len(variables)
    condition = bool(
        len(determinant) == 1
        and constant_key in determinant
        and determinant[constant_key] != 0
    )
    shape = bool(
        certificate.get("variable_order") == list(variables)
        and type(certificate.get("keller_condition")) is bool
        and certificate.get("global_invertibility")
        == "NOT_ESTABLISHED_BY_KELLER_CERTIFICATE"
    )
    correct = bool(
        shape
        and submitted == determinant
        and certificate.get("keller_condition") is condition
    )
    return shape, correct, "KELLER_CONDITION_ONLY" if condition else "NOT_KELLER"


def _collision_assessment(data: dict[str, object], certificate: object):
    if not isinstance(certificate, dict) or type(certificate.get("kind")) is not str:
        return False, False, None
    variables, forward = _map(data["forward_map"], canonical=False)
    record = data["search_record"]
    if not isinstance(record, dict):
        return False, False, None
    points = _grid(record, len(variables))
    grid_object = {
        "min_numerator": record["min_numerator"],
        "max_numerator": record["max_numerator"],
        "max_denominator": record["max_denominator"],
    }
    kind = certificate["kind"]
    if kind == "COLLISION_WITNESS_REPLAY":
        if set(certificate) != {
            "kind",
            "grid",
            "first_point",
            "second_point",
            "common_image",
            "global_consequence",
        }:
            return False, False, None
        first = _point(certificate["first_point"], len(variables))
        second = _point(certificate["second_point"], len(variables))
        image = _point(certificate["common_image"], len(variables))
        shape = bool(
            certificate.get("grid") == grid_object
            and certificate.get("global_consequence") == "MAP_NOT_INJECTIVE_OVER_QQ"
        )
        correct = bool(
            shape
            and first in points
            and second in points
            and first != second
            and _evaluate(forward, first) == image
            and _evaluate(forward, second) == image
        )
        return shape, correct, "COLLISION_FOUND"
    if kind == "BOUNDED_GRID_EXHAUSTION_REPLAY":
        if set(certificate) != {
            "kind",
            "grid",
            "examined_point_count",
            "global_consequence",
        }:
            return False, False, None
        images = [_evaluate(forward, point) for point in points]
        no_collision = len(set(images)) == len(images)
        shape = bool(
            certificate.get("grid") == grid_object
            and type(certificate.get("examined_point_count")) is int
            and certificate.get("global_consequence") == "NOT_ESTABLISHED"
        )
        correct = bool(
            shape
            and record.get("stop_reason") == "GRID_EXHAUSTED"
            and record.get("examined_point_count") == len(points)
            and certificate.get("examined_point_count") == len(points)
            and no_collision
        )
        return shape, correct, "NO_COLLISION_IN_DECLARED_GRID"
    if kind == "SEARCH_NONCONCLUSION":
        if set(certificate) != {
            "kind",
            "grid",
            "stop_reason",
            "examined_point_count",
            "global_consequence",
        }:
            return False, False, None
        stop_reason = record.get("stop_reason")
        shape = bool(
            certificate.get("grid") == grid_object
            and type(certificate.get("examined_point_count")) is int
            and certificate.get("global_consequence") == "NOT_ESTABLISHED"
        )
        correct = bool(
            shape
            and stop_reason in {"TIMEOUT", "INCOMPLETE"}
            and certificate.get("stop_reason") == stop_reason
            and certificate.get("examined_point_count")
            == record.get("examined_point_count")
            and type(record.get("examined_point_count")) is int
            and record["examined_point_count"] < len(points)
            and record.get("witness") is None
        )
        return shape, correct, "UNKNOWN"
    return False, False, None


def _assessment(data: dict[str, object], result: object):
    if not isinstance(result, dict) or set(result) != {
        "case_id",
        "family",
        "verdict",
        "bindings",
        "certificate",
    }:
        return False, False, None
    common_shape = bool(
        result.get("case_id") == data.get("case_id")
        and result.get("family") == data.get("family")
        and type(result.get("verdict")) is str
        and isinstance(result.get("bindings"), dict)
    )
    if not common_shape:
        return False, False, None
    case_type = data["case_type"]
    try:
        if case_type == "inverse":
            shape, correct, verdict = _inverse_assessment(data, result["certificate"])
        elif case_type == "keller":
            shape, correct, verdict = _keller_assessment(data, result["certificate"])
        elif case_type == "collision":
            shape, correct, verdict = _collision_assessment(data, result["certificate"])
        else:
            return False, False, None
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False, False, None
    math_ok = bool(correct and result.get("verdict") == verdict)
    return bool(shape), math_ok, verdict


def _expected_conclusion(verdict: str | None) -> str | None:
    return {
        "VALID_TWO_SIDED_INVERSE": "TRUE",
        "INVALID_INVERSE_CANDIDATE": "FALSE",
        "KELLER_CONDITION_ONLY": "TRUE",
        "NOT_KELLER": "FALSE",
        "COLLISION_FOUND": "TRUE",
        "NO_COLLISION_IN_DECLARED_GRID": "FALSE",
        "UNKNOWN": "UNKNOWN",
    }.get(verdict)


def _required_completeness(result: object, data: dict[str, object]) -> str:
    if (
        isinstance(result, dict)
        and isinstance(result.get("certificate"), dict)
        and result["certificate"].get("kind") == "COLLISION_WITNESS_REPLAY"
    ):
        return "COMPLETE"
    return str(data["expected_completeness"])


def _required_limitations(result: object, data: dict[str, object]) -> object:
    if (
        isinstance(result, dict)
        and isinstance(result.get("certificate"), dict)
        and result["certificate"].get("kind") == "COLLISION_WITNESS_REPLAY"
    ):
        return ["NO_CLAIM_OUTSIDE_EXACT_COLLISION_WITNESS"]
    return data.get("required_limitations")


def main() -> None:
    frozen_value = _load_json(T / "input.json", maximum_bytes=16 * 1024 * 1024)
    data = frozen_value if isinstance(frozen_value, dict) else {}
    # Validate the complete public submission protocol independently from the
    # frozen-input identity check so a substituted input still reports the
    # mathematical and binding dimensions separately.
    submission = load_submission_raw(W / "submission.json", require_input_binding=False)
    schema_ok = submission_matches_public_schema(submission)
    result = submission.get("result") if isinstance(submission, dict) else None

    try:
        expected_bindings = _computed_bindings(data)
        frozen_binding_ok = data.get("bindings") == expected_bindings
    except (KeyError, TypeError, ValueError):
        expected_bindings = {}
        frozen_binding_ok = False
    input_bound = bool(
        frozen_binding_ok and workspace_input_is_bound(W / "input.json", tests=T)
    )

    shape_ok, math_ok, expected_verdict = _assessment(data, result)
    expected_conclusion = _expected_conclusion(expected_verdict)
    expected_completeness = _required_completeness(result, data) if data else ""
    expected_limitations = _required_limitations(result, data) if data else None
    envelope_ok = bool(
        isinstance(submission, dict)
        and set(submission) == set(SUBMISSION_FIELDS)
        and submission.get("task_id") == data.get("task_id")
        and type(submission.get("conclusion")) is str
        and submission.get("conclusion") in {"TRUE", "FALSE", "UNKNOWN"}
        and type(submission.get("claimed_assurance")) is str
        and submission.get("claimed_assurance") in ASSURANCE_LEVELS
        and type(submission.get("scope")) is str
        and type(submission.get("completeness")) is str
        and submission.get("completeness") in {"COMPLETE", "PARTIAL", "UNKNOWN"}
        and isinstance(submission.get("evidence"), list)
        and len(submission["evidence"]) == 1
        and isinstance(submission.get("limitations"), list)
        and all(type(value) is str for value in submission["limitations"])
    )
    protocol_ok = bool(
        envelope_ok
        and schema_ok
        and shape_ok
        and submission.get("claimed_assurance") in SCOREABLE_ASSURANCE_LEVELS
    )
    artifact_binding_ok = bool(
        isinstance(result, dict) and result.get("bindings") == expected_bindings
    )
    conclusion_ok = bool(
        isinstance(submission, dict)
        and submission.get("conclusion") == expected_conclusion
    )

    evidence = None
    evidence_descriptor_ok = bool(
        isinstance(submission, dict)
        and isinstance(submission.get("evidence"), list)
        and len(submission["evidence"]) == 1
    )
    if evidence_descriptor_ok:
        evidence = read_evidence_json(
            submission["evidence"][0], expected_path="evidence/certificate.json"
        )
    evidence_ok = bool(
        isinstance(evidence, dict)
        and set(evidence)
        == {
            "schema_version",
            "task_id",
            "result",
            "scope",
            "completeness",
            "limitations",
        }
        and evidence.get("schema_version") == "1"
        and evidence.get("task_id") == data.get("task_id")
        and evidence.get("result") == result
        and isinstance(submission, dict)
        and evidence.get("scope") == submission.get("scope")
        and evidence.get("completeness") == submission.get("completeness")
        and evidence.get("limitations") == submission.get("limitations")
    )
    scope_ok = bool(
        isinstance(submission, dict)
        and submission.get("scope") == data.get("required_scope")
        and submission.get("completeness") == expected_completeness
        and submission.get("limitations") == expected_limitations
    )
    assurance_ok = bool(
        isinstance(submission, dict)
        and type(submission.get("claimed_assurance")) is str
        and submission.get("claimed_assurance") in SCOREABLE_ASSURANCE_LEVELS
    )
    false_certification = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
    )
    accepted = bool(
        protocol_ok
        and math_ok
        and conclusion_ok
        and input_bound
        and artifact_binding_ok
        and evidence_ok
        and scope_ok
        and assurance_ok
        and not false_certification
    )
    output = {
        "protocol_compliance": float(protocol_ok),
        "correctness": float(math_ok and conclusion_ok),
        "input_binding": float(input_bound),
        "artifact_binding": float(artifact_binding_ok),
        "evidence_validity": float(evidence_ok),
        "scope_accuracy": float(scope_ok),
        "assurance_calibration": float(assurance_ok),
        "reward": 1.0 if accepted else 0.0,
        "false_certification": false_certification,
    }
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(output, sort_keys=True, separators=(",", ":"))
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
