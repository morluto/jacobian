import json
from itertools import combinations
from math import gcd
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission_raw,
    read_evidence_json,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
RANK_PRIME = 1_000_003
COEFFICIENT_LIMIT = 1_000_000_000


def _load_frozen_input():
    try:
        workspace = WORKSPACE / "input.json"
        frozen = TESTS / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        frozen_bytes = frozen.read_bytes()
        if workspace.read_bytes() != frozen_bytes:
            return {}
        value = json.loads(frozen_bytes)
    except (OSError, ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _integer(value):
    return type(value) is int


def _monomials(degree):
    return [
        (x_degree, y_degree, degree - x_degree - y_degree)
        for x_degree in range(degree, -1, -1)
        for y_degree in range(degree - x_degree, -1, -1)
    ]


def _multiply(left, right):
    result = {}
    for left_exp, left_coefficient in left.items():
        for right_exp, right_coefficient in right.items():
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp, strict=True))
            result[exponent] = (
                result.get(exponent, 0) + left_coefficient * right_coefficient
            )
    return {
        exponent: coefficient for exponent, coefficient in result.items() if coefficient
    }


def _expand(factors):
    polynomial = {(0, 0, 0): 1}
    for x_coefficient, y_coefficient, z_coefficient in factors:
        polynomial = _multiply(
            polynomial,
            {
                (1, 0, 0): x_coefficient,
                (0, 1, 0): y_coefficient,
                (0, 0, 1): z_coefficient,
            },
        )
    return polynomial


def _derivative(polynomial, variable):
    result = {}
    for exponent, coefficient in polynomial.items():
        if exponent[variable]:
            target = list(exponent)
            target[variable] -= 1
            result[tuple(target)] = coefficient * exponent[variable]
    return result


def _canonical_projective_point(point):
    divisor = 0
    for coordinate in point:
        divisor = gcd(divisor, abs(coordinate))
    if divisor == 0:
        return None
    normalized = tuple(coordinate // divisor for coordinate in point)
    first = next(coordinate for coordinate in normalized if coordinate)
    return tuple(-coordinate for coordinate in normalized) if first < 0 else normalized


def _non_double_flats(factors):
    flats = set()
    for first, second in combinations(range(len(factors)), 2):
        left = factors[first]
        right = factors[second]
        point = _canonical_projective_point(
            (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )
        )
        if point is None:
            return None
        incident = tuple(
            index + 1
            for index, line in enumerate(factors)
            if sum(
                coefficient * coordinate
                for coefficient, coordinate in zip(line, point, strict=True)
            )
            == 0
        )
        if len(incident) >= 3:
            flats.add(incident)
    return sorted(flats)


def _parse_flats(value):
    if not isinstance(value, list):
        return None
    normalized = []
    for flat in value:
        if (
            not isinstance(flat, list)
            or not 3 <= len(flat) <= 9
            or not all(_integer(index) and 1 <= index <= 9 for index in flat)
            or flat != sorted(flat)
            or len(flat) != len(set(flat))
        ):
            return None
        normalized.append(tuple(flat))
    if normalized != sorted(normalized) or len(normalized) != len(set(normalized)):
        return None
    return normalized


def _parse_polynomial(value, degree):
    if not isinstance(value, list) or len(value) > len(_monomials(degree)):
        return None
    order = {exponent: index for index, exponent in enumerate(_monomials(degree))}
    polynomial = {}
    positions = []
    for term in value:
        if not isinstance(term, dict) or set(term) != {"exponents", "coefficient"}:
            return None
        exponents = term.get("exponents")
        coefficient = term.get("coefficient")
        if (
            not isinstance(exponents, list)
            or len(exponents) != 3
            or not all(_integer(exponent) and exponent >= 0 for exponent in exponents)
            or sum(exponents) != degree
            or not _integer(coefficient)
            or coefficient == 0
            or abs(coefficient) > COEFFICIENT_LIMIT
        ):
            return None
        exponent = tuple(exponents)
        if exponent in polynomial:
            return None
        polynomial[exponent] = coefficient
        positions.append(order[exponent])
    if positions != sorted(positions):
        return None
    return polynomial


def _parse_relation(value, expected_degree):
    if (
        not isinstance(value, dict)
        or set(value) != {"degree", "A", "B", "C"}
        or value.get("degree") != expected_degree
    ):
        return None
    polynomials = tuple(
        _parse_polynomial(value.get(name), expected_degree) for name in ("A", "B", "C")
    )
    if any(polynomial is None for polynomial in polynomials) or not any(polynomials):
        return None
    return polynomials


def _parse_certificate(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "non_double_flats", "relations"}
        or value.get("schema_version") != "1"
        or not isinstance(value.get("relations"), dict)
        or set(value["relations"]) != {"f", "g"}
    ):
        return None
    flats = _parse_flats(value.get("non_double_flats"))
    relation_f = _parse_relation(value["relations"].get("f"), 4)
    relation_g = _parse_relation(value["relations"].get("g"), 5)
    if flats is None or relation_f is None or relation_g is None:
        return None
    return flats, {"f": relation_f, "g": relation_g}


def _parse_instance(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"variables", "linear_factors", "listed_non_double_flats"}
        or value.get("variables") != ["x", "y", "z"]
        or not isinstance(value.get("linear_factors"), dict)
        or set(value["linear_factors"]) != {"f", "g"}
    ):
        return None
    factors_by_name = {}
    for name in ("f", "g"):
        factors = value["linear_factors"].get(name)
        if (
            not isinstance(factors, list)
            or len(factors) != 9
            or not all(
                isinstance(factor, list)
                and len(factor) == 3
                and all(_integer(coefficient) for coefficient in factor)
                and any(factor)
                for factor in factors
            )
        ):
            return None
        factors_by_name[name] = [tuple(factor) for factor in factors]
    listed = _parse_flats(value.get("listed_non_double_flats"))
    return (factors_by_name, listed) if listed is not None else None


def _relation_holds(factors, relation):
    source = _expand(factors)
    derivatives = tuple(_derivative(source, variable) for variable in range(3))
    residual = {}
    for multiplier, derivative in zip(relation, derivatives, strict=True):
        for exponent, coefficient in _multiply(multiplier, derivative).items():
            residual[exponent] = residual.get(exponent, 0) + coefficient
    return bool(
        any(relation) and all(coefficient == 0 for coefficient in residual.values())
    )


def _graded_matrix(factors, degree):
    source = _expand(factors)
    derivatives = tuple(_derivative(source, variable) for variable in range(3))
    domain = _monomials(degree)
    codomain = _monomials(degree + 8)
    row_index = {exponent: index for index, exponent in enumerate(codomain)}
    matrix = [[0] * (3 * len(domain)) for _ in codomain]
    for variable, derivative in enumerate(derivatives):
        for monomial_index, monomial in enumerate(domain):
            column = variable * len(domain) + monomial_index
            for exponent, coefficient in derivative.items():
                target = tuple(
                    left + right for left, right in zip(exponent, monomial, strict=True)
                )
                matrix[row_index[target]][column] = coefficient
    return matrix


def _rank_mod_prime(matrix):
    rows = [[value % RANK_PRIME for value in row] for row in matrix]
    rank = 0
    columns = len(rows[0]) if rows else 0
    for column in range(columns):
        pivot = next(
            (index for index in range(rank, len(rows)) if rows[index][column]),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = pow(rows[rank][column], -1, RANK_PRIME)
        for index in range(rank + 1, len(rows)):
            if rows[index][column]:
                factor = rows[index][column] * inverse % RANK_PRIME
                rows[index][column:] = [
                    (left - factor * right) % RANK_PRIME
                    for left, right in zip(
                        rows[index][column:],
                        rows[rank][column:],
                        strict=True,
                    )
                ]
        rank += 1
        if rank == columns:
            break
    return rank


def _injective_below(factors, degree):
    for lower_degree in range(degree):
        matrix = _graded_matrix(factors, lower_degree)
        if _rank_mod_prime(matrix) != len(matrix[0]):
            return False
    return True


def _result_matches(result):
    return bool(
        isinstance(result, dict)
        and set(result) == {"certificate_path", "same_non_double_flats", "mdr"}
        and result.get("certificate_path") == "evidence/syzygy-certificate.json"
        and result.get("same_non_double_flats") is True
        and result.get("mdr") == {"f": 4, "g": 5}
    )


def _evaluate(input_data, certificate, result):
    instance = _parse_instance(input_data.get("instance"))
    parsed = _parse_certificate(certificate)
    if instance is None or parsed is None:
        return False, False
    factors_by_name, listed = instance
    submitted_flats, relations = parsed
    computed_f = _non_double_flats(factors_by_name["f"])
    computed_g = _non_double_flats(factors_by_name["g"])
    evidence_valid = True
    correct = bool(
        computed_f == listed
        and computed_g == listed
        and submitted_flats == listed
        and _injective_below(factors_by_name["f"], 4)
        and _injective_below(factors_by_name["g"], 5)
        and _relation_holds(factors_by_name["f"], relations["f"])
        and _relation_holds(factors_by_name["g"], relations["g"])
        and _result_matches(result)
    )
    return evidence_valid, correct


def main():
    submission = load_submission_raw()
    data = submission if isinstance(submission, dict) else {}
    expected = json.loads((TESTS / "expected.json").read_text())
    input_data = _load_frozen_input()
    contract = bool(
        strict_submission_contract(
            submission,
            task_id=expected["task_id"],
            conclusion=expected["conclusion"],
            allowed_assurances=frozenset({expected["maximum_assurance"]}),
            verification_record="forbidden",
        )
        and data.get("limitations") == expected["required_limitations"]
    )

    evidence = data.get("evidence")
    certificate = None
    if contract and isinstance(evidence, list) and len(evidence) == 1:
        certificate = read_evidence_json(
            evidence[0],
            expected_path=expected["evidence_path"],
        )
    evidence_valid, math_correct = (
        _evaluate(
            input_data,
            certificate,
            data.get("result"),
        )
        if contract
        else (False, False)
    )
    evidence_valid = bool(contract and evidence_valid)
    math_correct = bool(contract and math_correct)
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    reward = float(
        all(
            (
                math_correct,
                evidence_valid,
                scope_correct,
                assurance_correct,
            )
        )
        and not false_certification
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
