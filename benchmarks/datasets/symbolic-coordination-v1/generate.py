#!/usr/bin/env python3
"""Deterministically render the symbolic-coordination-v1 pilot task bundles."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tooling.harbor_suite import verifier_bundle_checksum_bytes  # noqa: E402
from benchmarks.tooling.public_contract import (  # noqa: E402
    PublicContract,
    render_instruction,
    render_submission_schema,
)

DATASET = Path(__file__).resolve().parent
# Keep this pilot stable when the reusable task template evolves.  The
# committed copy in the generated task is the deliberate migration boundary.
TEMPLATE_SUPPORT = (
    DATASET / "symbolic-coordination-valid-inverse-01/tests/verifier_support.py"
)
VERIFIER_TEMPLATE = DATASET / "verifier_template.py"
IMAGE = "python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de"
GENERATOR_VERSION = "symbolic-coordination-pilot-generator@1"
CASE_VERSION = "symbolic-coordination-v1/pilot-1"
CHECKER_ID = "symbolic-coordination-v1.clean-room-polynomial-map-checker@1"
SEMANTICS_ID = "exact-sparse-polynomial-maps-over-QQ@1"

Poly = dict[tuple[int, ...], Fraction]
MapCoordinates = tuple[Poly, ...]


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_object(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def coefficient(value: int | str | Fraction) -> dict[str, str]:
    parsed = Fraction(value)
    return {"num": str(parsed.numerator), "den": str(parsed.denominator)}


def polynomial(
    *terms: tuple[int | str | Fraction, tuple[int, ...]],
) -> dict[str, object]:
    return {
        "terms": [
            {"coefficient": coefficient(value), "exponents": list(exponents)}
            for value, exponents in terms
        ]
    }


def polynomial_map(variables: tuple[str, ...], *coordinates: dict[str, object]):
    return {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": list(variables),
        "coordinates": list(coordinates),
    }


def parse_poly(value: dict[str, object], dimension: int) -> Poly:
    result: Poly = {}
    for term in value["terms"]:  # type: ignore[index]
        rational = term["coefficient"]  # type: ignore[index]
        parsed = Fraction(int(rational["num"]), int(rational["den"]))
        exponents = tuple(term["exponents"])  # type: ignore[index]
        if len(exponents) != dimension:
            raise ValueError("fixture exponent dimension mismatch")
        result[exponents] = result.get(exponents, Fraction(0)) + parsed
        if result[exponents] == 0:
            del result[exponents]
    return result


def parse_map(value: dict[str, object]) -> tuple[tuple[str, ...], MapCoordinates]:
    variables = tuple(value["variables"])  # type: ignore[arg-type]
    coordinates = tuple(
        parse_poly(coordinate, len(variables))
        for coordinate in value["coordinates"]  # type: ignore[union-attr]
    )
    return variables, coordinates


def rational(value: Fraction) -> dict[str, str]:
    return {"num": str(value.numerator), "den": str(value.denominator)}


def encode_poly(value: Poly) -> dict[str, object]:
    return {
        "terms": [
            {"coefficient": rational(value[exponents]), "exponents": list(exponents)}
            for exponents in sorted(value, reverse=True)
        ]
    }


def encode_map(variables: tuple[str, ...], coordinates: MapCoordinates):
    return polynomial_map(
        variables, *(encode_poly(coordinate) for coordinate in coordinates)
    )


def add(left: Poly, right: Poly) -> Poly:
    result = dict(left)
    for exponents, value in right.items():
        result[exponents] = result.get(exponents, Fraction(0)) + value
        if result[exponents] == 0:
            del result[exponents]
    return result


def multiply(left: Poly, right: Poly) -> Poly:
    result: Poly = {}
    for left_exponents, left_value in left.items():
        for right_exponents, right_value in right.items():
            exponents = tuple(
                a + b for a, b in zip(left_exponents, right_exponents, strict=True)
            )
            result[exponents] = (
                result.get(exponents, Fraction(0)) + left_value * right_value
            )
            if result[exponents] == 0:
                del result[exponents]
    return result


def power(value: Poly, exponent: int, dimension: int) -> Poly:
    result = {(0,) * dimension: Fraction(1)}
    for _ in range(exponent):
        result = multiply(result, value)
    return result


def compose(outer: MapCoordinates, inner: MapCoordinates) -> MapCoordinates:
    dimension = len(inner)
    result = []
    for coordinate in outer:
        total: Poly = {}
        for exponents, value in coordinate.items():
            term = {(0,) * dimension: value}
            for inner_coordinate, exponent in zip(inner, exponents, strict=True):
                term = multiply(term, power(inner_coordinate, exponent, dimension))
            total = add(total, term)
        result.append(total)
    return tuple(result)


def residuals(value: MapCoordinates) -> MapCoordinates:
    dimension = len(value)
    return tuple(
        add(
            coordinate,
            {
                tuple(
                    int(position == index) for position in range(dimension)
                ): Fraction(-1)
            },
        )
        for index, coordinate in enumerate(value)
    )


def derivative(value: Poly, variable: int) -> Poly:
    result = {}
    for exponents, current in value.items():
        if exponents[variable]:
            derived = list(exponents)
            derived[variable] -= 1
            result[tuple(derived)] = current * exponents[variable]
    return result


def sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def determinant(coordinates: MapCoordinates) -> Poly:
    dimension = len(coordinates)
    matrix = tuple(
        tuple(derivative(coordinate, column) for column in range(dimension))
        for coordinate in coordinates
    )
    result: Poly = {}
    for permutation in itertools.permutations(range(dimension)):
        term = {(0,) * dimension: Fraction(sign(permutation))}
        for row, column in enumerate(permutation):
            term = multiply(term, matrix[row][column])
        result = add(result, term)
    return result


def evaluate(coordinates: MapCoordinates, point: tuple[Fraction, ...]):
    return tuple(
        sum(
            (
                value
                * _product(
                    point[position] ** exponent
                    for position, exponent in enumerate(exponents)
                )
                for exponents, value in coordinate.items()
            ),
            start=Fraction(0),
        )
        for coordinate in coordinates
    )


def _product(values):
    result = Fraction(1)
    for value in values:
        result *= value
    return result


def encoded_point(point: tuple[Fraction, ...]):
    return [rational(value) for value in point]


def triangular_x_y2(source=("x", "y"), target=("u", "v")):
    forward = polynomial_map(
        source,
        polynomial((1, (1, 0)), (1, (0, 2))),
        polynomial((1, (0, 1))),
    )
    inverse = polynomial_map(
        target,
        polynomial((1, (1, 0)), (-1, (0, 2))),
        polynomial((1, (0, 1))),
    )
    return forward, inverse


def triangular_y_x2(source=("x", "y"), target=("u", "v")):
    forward = polynomial_map(
        source,
        polynomial((1, (1, 0))),
        polynomial((1, (2, 0)), (1, (0, 1))),
    )
    inverse = polynomial_map(
        target,
        polynomial((1, (1, 0))),
        polynomial((-1, (2, 0)), (1, (0, 1))),
    )
    return forward, inverse


def affine_shift(source=("x", "y"), target=("u", "v")):
    return (
        polynomial_map(
            source,
            polynomial((1, (1, 0)), (2, (0, 1)), (1, (0, 0))),
            polynomial((1, (0, 1)), (-3, (0, 0))),
        ),
        polynomial_map(
            target,
            polynomial((1, (1, 0)), (-2, (0, 1)), (-7, (0, 0))),
            polynomial((1, (0, 1)), (3, (0, 0))),
        ),
    )


def linear_half(source=("x", "y"), target=("u", "v")):
    return (
        polynomial_map(
            source,
            polynomial((1, (1, 0)), (-1, (0, 1))),
            polynomial((1, (1, 0)), (1, (0, 1))),
        ),
        polynomial_map(
            target,
            polynomial((Fraction(1, 2), (1, 0)), (Fraction(1, 2), (0, 1))),
            polynomial((Fraction(-1, 2), (1, 0)), (Fraction(1, 2), (0, 1))),
        ),
    )


def triangular_three():
    return (
        polynomial_map(
            ("x", "y", "z"),
            polynomial((1, (1, 0, 0)), (1, (0, 2, 0))),
            polynomial((1, (0, 1, 0)), (1, (0, 0, 1))),
            polynomial((1, (0, 0, 1))),
        ),
        polynomial_map(
            ("u", "v", "w"),
            polynomial(
                (1, (1, 0, 0)),
                (-1, (0, 2, 0)),
                (2, (0, 1, 1)),
                (-1, (0, 0, 2)),
            ),
            polynomial((1, (0, 1, 0)), (-1, (0, 0, 1))),
            polynomial((1, (0, 0, 1))),
        ),
    )


def inverse_case(
    slug: str,
    family: str,
    forward: dict[str, object],
    candidate: dict[str, object],
    *,
    checked_directions: tuple[str, ...] = (),
    note: str,
) -> dict[str, object]:
    supplied = {
        "evidence_schema_version": "1",
        "status": "CHECKED" if checked_directions else "COMPUTED",
        "checked_directions": list(checked_directions),
        "claimed_assurance": "VERIFIED" if checked_directions else "COMPUTED",
        "note": "Non-authoritative supplied material; independently replay both directions.",
    }
    return {
        "slug": slug,
        "family": family,
        "case_type": "inverse",
        "forward_map": forward,
        "candidate_inverse": candidate,
        "supplied_evidence": supplied,
        "expected_completeness": "COMPLETE",
        "required_limitations": [
            "NO_CLAIM_BEYOND_SUPPLIED_CANDIDATE_AND_EXACT_QQ_COMPOSITIONS"
        ],
        "note": note,
    }


def keller_case(slug: str, forward: dict[str, object], *, note: str):
    _, coordinates = parse_map(forward)
    supplied = {
        "certificate_schema_version": "1",
        "determinant": encode_poly(determinant(coordinates)),
        "provider_status": "COMPUTED",
        "global_invertibility": "UNASSESSED",
    }
    return {
        "slug": slug,
        "family": "constant-nonzero-jacobian",
        "case_type": "keller",
        "forward_map": forward,
        "supplied_certificate": supplied,
        "expected_completeness": "COMPLETE",
        "required_limitations": [
            "KELLER_CONDITION_DOES_NOT_LICENSE_GLOBAL_INVERTIBILITY"
        ],
        "note": note,
    }


def collision_case(
    slug: str,
    forward: dict[str, object],
    *,
    stop_reason: str,
    examined: int,
    note: str,
):
    variables, coordinates = parse_map(forward)
    points = tuple(
        itertools.product(
            (Fraction(-2), Fraction(-1), Fraction(0), Fraction(1), Fraction(2)),
            repeat=len(variables),
        )
    )
    seen: dict[tuple[Fraction, ...], tuple[Fraction, ...]] = {}
    found = None
    for point in points:
        image = evaluate(coordinates, point)
        if image in seen and seen[image] != point:
            found = {
                "first_point": encoded_point(seen[image]),
                "second_point": encoded_point(point),
                "common_image": encoded_point(image),
            }
            break
        seen[image] = point
    if stop_reason == "FOUND" and found is None:
        raise ValueError(f"{slug} has no collision")
    record = {
        "search_schema_version": "1",
        "min_numerator": -2,
        "max_numerator": 2,
        "max_denominator": 1,
        "grid_point_count": len(points),
        "examined_point_count": examined,
        "execution_status": (
            "TIMEOUT"
            if stop_reason == "TIMEOUT"
            else "CANCELLED"
            if stop_reason == "INCOMPLETE"
            else "COMPLETED"
        ),
        "stop_reason": stop_reason,
        "witness": found if stop_reason == "FOUND" else None,
    }
    if stop_reason == "GRID_EXHAUSTED" and found is not None:
        raise ValueError(f"{slug} is not collision-free on the grid")
    limitations = (
        ["NO_CLAIM_OUTSIDE_EXACT_COLLISION_WITNESS"]
        if stop_reason == "FOUND"
        else ["NO_GLOBAL_INJECTIVITY_OR_INVERTIBILITY_CONCLUSION"]
        if stop_reason == "GRID_EXHAUSTED"
        else [
            "SEARCH_NONCONCLUSION",
            "NO_GLOBAL_INJECTIVITY_OR_INVERTIBILITY_CONCLUSION",
        ]
    )
    return {
        "slug": slug,
        "family": "bounded-collision-scope",
        "case_type": "collision",
        "forward_map": forward,
        "search_record": record,
        "expected_completeness": (
            "UNKNOWN"
            if stop_reason == "TIMEOUT"
            else "PARTIAL"
            if stop_reason == "INCOMPLETE"
            else "COMPLETE"
        ),
        "required_limitations": limitations,
        "note": note,
    }


def cases() -> list[dict[str, object]]:
    a_forward, a_inverse = triangular_x_y2()
    b_forward, b_inverse = triangular_y_x2()
    c_forward, c_inverse = affine_shift()
    d_forward, d_inverse = linear_half()
    e_forward, e_inverse = triangular_three()
    result = [
        inverse_case(
            "symbolic-coordination-valid-inverse-01",
            "valid-two-sided-inverse",
            a_forward,
            a_inverse,
            note="Two-variable triangular shear.",
        ),
        inverse_case(
            "symbolic-coordination-valid-inverse-02",
            "valid-two-sided-inverse",
            b_forward,
            b_inverse,
            note="Triangular shear in the second coordinate.",
        ),
        inverse_case(
            "symbolic-coordination-valid-inverse-03",
            "valid-two-sided-inverse",
            c_forward,
            c_inverse,
            note="Affine translation and shear.",
        ),
        inverse_case(
            "symbolic-coordination-valid-inverse-04",
            "valid-two-sided-inverse",
            d_forward,
            d_inverse,
            note="Linear inverse with rational coefficients.",
        ),
        inverse_case(
            "symbolic-coordination-valid-inverse-05",
            "valid-two-sided-inverse",
            e_forward,
            e_inverse,
            note="Three-variable triangular composition.",
        ),
    ]
    near_1 = deepcopy(a_inverse)
    near_1["coordinates"][0] = polynomial((1, (1, 0)), (-2, (0, 2)))
    near_2 = deepcopy(b_inverse)
    near_2["coordinates"][1] = polynomial((1, (2, 0)), (1, (0, 1)))
    near_3 = deepcopy(c_inverse)
    near_3["coordinates"][0] = polynomial((1, (1, 0)), (-2, (0, 1)), (-6, (0, 0)))
    near_4 = deepcopy(d_inverse)
    near_4["coordinates"][1] = polynomial(
        (Fraction(-1, 3), (1, 0)), (Fraction(1, 3), (0, 1))
    )
    result.extend(
        [
            inverse_case(
                "symbolic-coordination-near-miss-01",
                "perturbed-near-miss",
                a_forward,
                near_1,
                note="Quadratic inverse coefficient perturbed.",
            ),
            inverse_case(
                "symbolic-coordination-near-miss-02",
                "perturbed-near-miss",
                b_forward,
                near_2,
                note="Inverse shear sign perturbed.",
            ),
            inverse_case(
                "symbolic-coordination-near-miss-03",
                "perturbed-near-miss",
                c_forward,
                near_3,
                note="Affine inverse constant perturbed.",
            ),
            inverse_case(
                "symbolic-coordination-near-miss-04",
                "perturbed-near-miss",
                d_forward,
                near_4,
                note="Linear inverse denominator perturbed.",
            ),
            inverse_case(
                "symbolic-coordination-one-direction-01",
                "one-direction-only-evidence",
                a_forward,
                a_inverse,
                checked_directions=("INVERSE_AFTER_FORWARD",),
                note="Supplied material checks only inverse-after-forward.",
            ),
            inverse_case(
                "symbolic-coordination-one-direction-02",
                "one-direction-only-evidence",
                b_forward,
                b_inverse,
                checked_directions=("FORWARD_AFTER_INVERSE",),
                note="Supplied material checks only forward-after-inverse.",
            ),
            inverse_case(
                "symbolic-coordination-one-direction-03",
                "one-direction-only-evidence",
                e_forward,
                e_inverse,
                checked_directions=("INVERSE_AFTER_FORWARD",),
                note="Three-dimensional supplied material omits one direction.",
            ),
        ]
    )
    keller_2 = polynomial_map(
        ("x", "y"),
        polynomial((1, (1, 0)), (1, (0, 2))),
        polynomial((2, (0, 1)), (1, (0, 0))),
    )
    keller_3 = polynomial_map(
        ("x", "y"),
        polynomial((1, (1, 0)), (2, (0, 1))),
        polynomial((3, (1, 0)), (5, (0, 1))),
    )
    keller_4 = polynomial_map(
        ("x", "y", "z"),
        polynomial((1, (1, 0, 0)), (1, (0, 2, 0))),
        polynomial((1, (0, 1, 0)), (1, (0, 0, 2))),
        polynomial((1, (0, 0, 1)), (1, (0, 0, 0))),
    )
    result.extend(
        [
            keller_case(
                "symbolic-coordination-keller-only-01",
                a_forward,
                note="Unit Jacobian triangular map; certificate scope remains Keller-only.",
            ),
            keller_case(
                "symbolic-coordination-keller-only-02",
                keller_2,
                note="Constant Jacobian two over QQ.",
            ),
            keller_case(
                "symbolic-coordination-keller-only-03",
                keller_3,
                note="Constant Jacobian minus one for a linear map.",
            ),
            keller_case(
                "symbolic-coordination-keller-only-04",
                keller_4,
                note="Three-variable unit Jacobian triangular map.",
            ),
        ]
    )
    symmetric = polynomial_map(
        ("x", "y"),
        polynomial((1, (1, 0)), (1, (0, 1))),
        polynomial((1, (1, 1))),
    )
    square = polynomial_map(
        ("x", "y"), polynomial((1, (2, 0))), polynomial((1, (0, 1)))
    )
    identity = polynomial_map(
        ("x", "y"), polynomial((1, (1, 0))), polynomial((1, (0, 1)))
    )
    odd = polynomial_map(("x", "y"), polynomial((1, (3, 0))), polynomial((1, (0, 1))))
    result.extend(
        [
            collision_case(
                "symbolic-coordination-collision-found-01",
                symmetric,
                stop_reason="FOUND",
                examined=3,
                note="Symmetry yields alternate swap witnesses.",
            ),
            collision_case(
                "symbolic-coordination-collision-found-02",
                square,
                stop_reason="FOUND",
                examined=6,
                note="Even first coordinate yields several alternate witnesses.",
            ),
            collision_case(
                "symbolic-coordination-grid-exhausted-01",
                identity,
                stop_reason="GRID_EXHAUSTED",
                examined=25,
                note="Grid exhaustion licenses only a bounded no-collision claim.",
            ),
            collision_case(
                "symbolic-coordination-grid-exhausted-02",
                odd,
                stop_reason="GRID_EXHAUSTED",
                examined=25,
                note="Odd cubic is collision-free on the declared grid.",
            ),
            collision_case(
                "symbolic-coordination-search-timeout-01",
                square,
                stop_reason="TIMEOUT",
                examined=3,
                note="Timeout without a witness is a non-conclusion.",
            ),
            collision_case(
                "symbolic-coordination-search-incomplete-01",
                symmetric,
                stop_reason="INCOMPLETE",
                examined=5,
                note="Cancelled incomplete search is a non-conclusion.",
            ),
        ]
    )
    semantic_1_forward = polynomial_map(
        ("a", "b"),
        polynomial((1, (0, 2)), (3, (1, 0)), (-2, (1, 0)), (0, (0, 0))),
        polynomial((2, (0, 1)), (-1, (0, 1))),
    )
    semantic_1_inverse = polynomial_map(
        ("p", "q"),
        polynomial((-1, (0, 2)), (2, (1, 0)), (-1, (1, 0)), (0, (0, 0))),
        polynomial((1, (0, 1))),
    )
    semantic_2_forward, semantic_2_inverse = affine_shift(("r", "s"), ("alpha", "beta"))
    semantic_2_forward["coordinates"][0] = polynomial(
        (2, (0, 1)), (4, (1, 0)), (-3, (1, 0)), (1, (0, 0))
    )
    semantic_2_inverse["coordinates"][0] = polynomial(
        (-7, (0, 0)), (-2, (0, 1)), (1, (1, 0))
    )
    semantic_3_forward, semantic_3_inverse = triangular_x_y2(
        ("m", "n"), ("rho", "sigma")
    )
    semantic_3_inverse["coordinates"][0] = polynomial((-1, (0, 2)), (1, (1, 0)))
    semantic_4_forward = polynomial_map(
        ("c", "d"),
        polynomial((Fraction(1, 2), (0, 2)), (1, (1, 0)), (2, (0, 1)), (-2, (0, 1))),
        polynomial((1, (0, 1))),
    )
    semantic_4_inverse = polynomial_map(
        ("gamma", "delta"),
        polynomial((1, (1, 0)), (Fraction(-1, 2), (0, 2))),
        polynomial((1, (0, 1))),
    )
    result.extend(
        [
            inverse_case(
                "symbolic-coordination-semantic-equivalence-01",
                "semantic-equivalence",
                semantic_1_forward,
                semantic_1_inverse,
                note="Term reordering, duplicates, cancellations, and variable renaming.",
            ),
            inverse_case(
                "symbolic-coordination-semantic-equivalence-02",
                "semantic-equivalence",
                semantic_2_forward,
                semantic_2_inverse,
                note="Affine duplicate terms and renamed source/target variables.",
            ),
            inverse_case(
                "symbolic-coordination-semantic-equivalence-03",
                "semantic-equivalence",
                semantic_3_forward,
                semantic_3_inverse,
                note="Equivalent reordered sparse inverse encoding.",
            ),
            inverse_case(
                "symbolic-coordination-semantic-equivalence-04",
                "semantic-equivalence",
                semantic_4_forward,
                semantic_4_inverse,
                note="Rational coefficients plus cancelling sparse terms and renaming.",
            ),
        ]
    )
    if len(result) != 26 or len({case["slug"] for case in result}) != 26:
        raise AssertionError("pilot must contain 26 unique cases")
    return result


def bind_case(case: dict[str, object]) -> dict[str, object]:
    value = deepcopy(case)
    slug = str(value.pop("slug"))
    note = str(value.pop("note"))
    value.update(
        {
            "task_id": f"jacobian/{slug}",
            "case_id": slug,
            "case_version": CASE_VERSION,
            "claim_id": f"claim://symbolic-coordination-v1/{slug}",
            "scope_id": f"scope://symbolic-coordination-v1/{slug}",
            "required_scope": (
                f"EXACT_TWO_SIDED_COMPOSITION_OVER_QQ:{slug}"
                if value["case_type"] == "inverse"
                else f"KELLER_CONDITION_ONLY_OVER_QQ:{slug}"
                if value["case_type"] == "keller"
                else f"DECLARED_RATIONAL_GRID_AND_SUPPLIED_SEARCH_RECORD:{slug}"
            ),
            "case_note": note,
        }
    )
    subject = (
        {
            "candidate_inverse": value["candidate_inverse"],
            "supplied_evidence": value.get("supplied_evidence"),
        }
        if value["case_type"] == "inverse"
        else value["supplied_certificate"]
        if value["case_type"] == "keller"
        else value["search_record"]
    )
    value["bindings"] = {
        "binding_schema_version": "1",
        "claim_id": value["claim_id"],
        "semantics_id": SEMANTICS_ID,
        "scope_id": value["scope_id"],
        "forward_map_sha256": sha256_object(value["forward_map"]),
        "subject_sha256": sha256_object(subject),
        "checker_id": CHECKER_ID,
    }
    return value


def inverse_certificate(data: dict[str, object]):
    source_variables, forward = parse_map(data["forward_map"])  # type: ignore[arg-type]
    target_variables, inverse = parse_map(data["candidate_inverse"])  # type: ignore[arg-type]
    left = residuals(compose(inverse, forward))
    right = residuals(compose(forward, inverse))
    valid = all(not item for item in (*left, *right))
    return (
        "VALID_TWO_SIDED_INVERSE" if valid else "INVALID_INVERSE_CANDIDATE",
        {
            "kind": "TWO_SIDED_COMPOSITION_REPLAY",
            "source_variables": list(source_variables),
            "target_variables": list(target_variables),
            "inverse_map": encode_map(target_variables, inverse),
            "inverse_after_forward_residuals": [encode_poly(item) for item in left],
            "forward_after_inverse_residuals": [encode_poly(item) for item in right],
            "checked_directions": ["INVERSE_AFTER_FORWARD", "FORWARD_AFTER_INVERSE"],
        },
    )


def keller_certificate(data: dict[str, object]):
    variables, forward = parse_map(data["forward_map"])  # type: ignore[arg-type]
    value = determinant(forward)
    constant = (
        len(value) == 1
        and (0,) * len(variables) in value
        and value[(0,) * len(variables)] != 0
    )
    return (
        "KELLER_CONDITION_ONLY" if constant else "NOT_KELLER",
        {
            "kind": "KELLER_DETERMINANT_REPLAY",
            "variable_order": list(variables),
            "determinant": encode_poly(value),
            "keller_condition": constant,
            "global_invertibility": "NOT_ESTABLISHED_BY_KELLER_CERTIFICATE",
        },
    )


def collision_certificate(data: dict[str, object]):
    record = data["search_record"]
    grid = {
        "min_numerator": record["min_numerator"],
        "max_numerator": record["max_numerator"],
        "max_denominator": record["max_denominator"],
    }
    if record["stop_reason"] == "FOUND":
        witness = record["witness"]
        return "COLLISION_FOUND", {
            "kind": "COLLISION_WITNESS_REPLAY",
            "grid": grid,
            **witness,
            "global_consequence": "MAP_NOT_INJECTIVE_OVER_QQ",
        }
    if record["stop_reason"] == "GRID_EXHAUSTED":
        return "NO_COLLISION_IN_DECLARED_GRID", {
            "kind": "BOUNDED_GRID_EXHAUSTION_REPLAY",
            "grid": grid,
            "examined_point_count": record["examined_point_count"],
            "global_consequence": "NOT_ESTABLISHED",
        }
    return "UNKNOWN", {
        "kind": "SEARCH_NONCONCLUSION",
        "grid": grid,
        "stop_reason": record["stop_reason"],
        "examined_point_count": record["examined_point_count"],
        "global_consequence": "NOT_ESTABLISHED",
    }


def solution(data: dict[str, object]) -> tuple[dict[str, object], bytes]:
    if data["case_type"] == "inverse":
        verdict, certificate_value = inverse_certificate(data)
    elif data["case_type"] == "keller":
        verdict, certificate_value = keller_certificate(data)
    else:
        verdict, certificate_value = collision_certificate(data)
    conclusion = {
        "VALID_TWO_SIDED_INVERSE": "TRUE",
        "INVALID_INVERSE_CANDIDATE": "FALSE",
        "KELLER_CONDITION_ONLY": "TRUE",
        "NOT_KELLER": "FALSE",
        "COLLISION_FOUND": "TRUE",
        "NO_COLLISION_IN_DECLARED_GRID": "FALSE",
        "UNKNOWN": "UNKNOWN",
    }[verdict]
    result = {
        "case_id": data["case_id"],
        "family": data["family"],
        "verdict": verdict,
        "bindings": data["bindings"],
        "certificate": certificate_value,
    }
    evidence = {
        "schema_version": "1",
        "task_id": data["task_id"],
        "result": result,
        "scope": data["required_scope"],
        "completeness": data["expected_completeness"],
        "limitations": data["required_limitations"],
    }
    evidence_bytes = json_bytes(evidence)
    submission = {
        "task_id": data["task_id"],
        "conclusion": conclusion,
        "result": result,
        "claimed_assurance": "CHECKED",
        "scope": data["required_scope"],
        "completeness": data["expected_completeness"],
        "evidence": [
            {
                "path": "evidence/certificate.json",
                "sha256": sha256_bytes(evidence_bytes),
            }
        ],
        "limitations": data["required_limitations"],
    }
    return submission, evidence_bytes


_COLLISION_WITNESS_LIMITATIONS = ["NO_CLAIM_OUTSIDE_EXACT_COLLISION_WITNESS"]


def _limitations_schema(data: dict[str, object]) -> dict[str, object]:
    """Allow collision-witness limitations when a timeout/incomplete search may
    still yield an exact collision witness in the declared grid."""

    required = data["required_limitations"]
    if (
        data["case_type"] == "collision"
        and required != _COLLISION_WITNESS_LIMITATIONS
        and data["search_record"].get("stop_reason") in {"TIMEOUT", "INCOMPLETE"}
    ):
        return {"enum": [required, _COLLISION_WITNESS_LIMITATIONS]}
    return {"const": required}


def submission_schema_parts(data: dict[str, object]) -> dict[str, object]:
    rational_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["num", "den"],
        "properties": {
            "num": {"type": "string", "pattern": "^-?(0|[1-9][0-9]*)$"},
            "den": {"type": "string", "pattern": "^[1-9][0-9]*$"},
        },
    }
    polynomial_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["terms"],
        "properties": {
            "terms": {
                "type": "array",
                "maxItems": 128,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["coefficient", "exponents"],
                    "properties": {
                        "coefficient": {"$ref": "#/$defs/rational"},
                        "exponents": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "items": {"type": "integer", "minimum": 0, "maximum": 32},
                        },
                    },
                },
            }
        },
    }
    map_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["map_schema_version", "domain", "variables", "coordinates"],
        "properties": {
            "map_schema_version": {"const": "1"},
            "domain": {"const": "QQ"},
            "variables": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {"type": "string"},
            },
            "coordinates": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {"$ref": "#/$defs/polynomial"},
            },
        },
    }
    bindings = data["bindings"]
    binding_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(bindings),
        "properties": {key: {"const": value} for key, value in bindings.items()},
    }
    grid_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["min_numerator", "max_numerator", "max_denominator"],
        "properties": {
            "min_numerator": {"type": "integer"},
            "max_numerator": {"type": "integer"},
            "max_denominator": {"type": "integer", "minimum": 1},
        },
    }
    point_schema = {
        "type": "array",
        "minItems": 1,
        "maxItems": 3,
        "items": {"$ref": "#/$defs/rational"},
    }
    certificate_variants = [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "kind",
                "source_variables",
                "target_variables",
                "inverse_map",
                "inverse_after_forward_residuals",
                "forward_after_inverse_residuals",
                "checked_directions",
            ],
            "properties": {
                "kind": {"const": "TWO_SIDED_COMPOSITION_REPLAY"},
                "source_variables": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {"type": "string"},
                },
                "target_variables": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {"type": "string"},
                },
                "inverse_map": {"$ref": "#/$defs/map"},
                "inverse_after_forward_residuals": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {"$ref": "#/$defs/polynomial"},
                },
                "forward_after_inverse_residuals": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {"$ref": "#/$defs/polynomial"},
                },
                "checked_directions": {
                    "const": ["INVERSE_AFTER_FORWARD", "FORWARD_AFTER_INVERSE"]
                },
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "kind",
                "variable_order",
                "determinant",
                "keller_condition",
                "global_invertibility",
            ],
            "properties": {
                "kind": {"const": "KELLER_DETERMINANT_REPLAY"},
                "variable_order": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {"type": "string"},
                },
                "determinant": {"$ref": "#/$defs/polynomial"},
                "keller_condition": {"type": "boolean"},
                "global_invertibility": {
                    "const": "NOT_ESTABLISHED_BY_KELLER_CERTIFICATE"
                },
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "kind",
                "grid",
                "first_point",
                "second_point",
                "common_image",
                "global_consequence",
            ],
            "properties": {
                "kind": {"const": "COLLISION_WITNESS_REPLAY"},
                "grid": {"$ref": "#/$defs/grid"},
                "first_point": {"$ref": "#/$defs/point"},
                "second_point": {"$ref": "#/$defs/point"},
                "common_image": {"$ref": "#/$defs/point"},
                "global_consequence": {"const": "MAP_NOT_INJECTIVE_OVER_QQ"},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "grid", "examined_point_count", "global_consequence"],
            "properties": {
                "kind": {"const": "BOUNDED_GRID_EXHAUSTION_REPLAY"},
                "grid": {"$ref": "#/$defs/grid"},
                "examined_point_count": {"type": "integer", "minimum": 1},
                "global_consequence": {"const": "NOT_ESTABLISHED"},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "kind",
                "grid",
                "stop_reason",
                "examined_point_count",
                "global_consequence",
            ],
            "properties": {
                "kind": {"const": "SEARCH_NONCONCLUSION"},
                "grid": {"$ref": "#/$defs/grid"},
                "stop_reason": {"enum": ["TIMEOUT", "INCOMPLETE"]},
                "examined_point_count": {"type": "integer", "minimum": 0},
                "global_consequence": {"const": "NOT_ESTABLISHED"},
            },
        },
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "task_id",
            "conclusion",
            "result",
            "claimed_assurance",
            "scope",
            "completeness",
            "evidence",
            "limitations",
        ],
        "properties": {
            "task_id": {"const": data["task_id"]},
            "conclusion": {"enum": ["TRUE", "FALSE", "UNKNOWN"]},
            "result": {
                "type": "object",
                "additionalProperties": False,
                "required": ["case_id", "family", "verdict", "bindings", "certificate"],
                "properties": {
                    "case_id": {"const": data["case_id"]},
                    "family": {"const": data["family"]},
                    "verdict": {
                        "enum": [
                            "VALID_TWO_SIDED_INVERSE",
                            "INVALID_INVERSE_CANDIDATE",
                            "KELLER_CONDITION_ONLY",
                            "NOT_KELLER",
                            "COLLISION_FOUND",
                            "NO_COLLISION_IN_DECLARED_GRID",
                            "UNKNOWN",
                        ]
                    },
                    "bindings": binding_schema,
                    "certificate": {"oneOf": certificate_variants},
                },
            },
            "claimed_assurance": {"const": "CHECKED"},
            "scope": {"const": data["required_scope"]},
            "completeness": {
                "enum": sorted({str(data["expected_completeness"]), "COMPLETE"})
            },
            "evidence": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "sha256"],
                    "properties": {
                        "path": {"const": "evidence/certificate.json"},
                        "sha256": {
                            "type": "string",
                            "pattern": "^sha256:[0-9a-f]{64}$",
                        },
                    },
                },
            },
            "limitations": _limitations_schema(data),
        },
        "$defs": {
            "rational": rational_schema,
            "polynomial": polynomial_schema,
            "map": map_schema,
            "grid": grid_schema,
            "point": point_schema,
        },
    }


def public_contract(data: dict[str, object]) -> PublicContract:
    """Build the canonical public protocol and its deterministic projection."""

    legacy_schema = submission_schema_parts(data)
    properties = legacy_schema["properties"]
    declaration = {
        "schema_version": "1",
        "task_id": data["task_id"],
        "submission_path": "/app/submission.json",
        "assurance_ceiling": "CHECKED",
        "allowed_assurance": ["UNVERIFIED", "COMPUTED", "CHECKED"],
        "allowed_completeness": properties["completeness"]["enum"],
        "conclusion": properties["conclusion"],
        "scope": {"type": "string", "const": data["required_scope"]},
        "evidence": {
            "min_items": 1,
            "max_items": 1,
            "allowed_paths": ["evidence/certificate.json"],
            "digest_pattern": "^sha256:[0-9a-f]{64}$",
            "media_types": ["application/json"],
        },
        "required_artifact_filenames": ["evidence/certificate.json"],
        "public_notes": (
            "Write evidence/certificate.json as a JSON wrapper with exactly "
            'these fields: schema_version (the string \\"1\\"), task_id '
            f'(the string \\"{data["task_id"]}\\"), result (an exact copy of '
            "the submission result object), scope (an exact copy of the "
            "submission scope), completeness (an exact copy of the submission "
            "completeness), and limitations (an exact copy of the submission "
            "limitations). Bind that exact regular file by SHA-256. The "
            "verifier independently checks the mathematics, input and artifact "
            "identities, declared scope, completeness, and assurance."
        ),
        "submission_result": properties["result"],
        "limitations": properties["limitations"],
        "schema_definitions": legacy_schema["$defs"],
    }
    draft = PublicContract.model_validate(declaration)
    declaration["submission_schema"] = json.loads(render_submission_schema(draft))
    return PublicContract.model_validate(declaration)


def render_task(
    data: dict[str, object], verifier: bytes, support: bytes
) -> dict[Path, bytes]:
    slug = str(data["case_id"])
    task = DATASET / slug
    input_content = json_bytes(data)
    fixture_digest = sha256_bytes(input_content)
    submission, evidence = solution(data)
    contract = public_contract(data)
    contract_value = contract.model_dump(mode="json", exclude_none=True)
    schema_text = render_submission_schema(contract)
    checksum = verifier_bundle_checksum_bytes(verifier, support)
    description = (
        f"Assess one exact polynomial-map claim in the {data['family']} pilot family."
    )
    task_toml = f'''schema_version = "1.4"
artifacts = ["/app/submission.json", "/app/evidence"]

[task]
name = "jacobian/{slug}"
version = "1.0.0"
description = "{description}"
keywords = ["polynomial", "map", "symbolic-coordination", "exact-certificate"]

[metadata]
evaluation_kind = "workflow"
domain = "mathematical-sciences"
field = "algebra"
primary_domain = "algebra"
assurance_ceiling = "CHECKED"
answer_visibility = "hidden-at-runtime"
provenance_class = "deterministic-hand-auditable-pilot"
fixture_digest = "{fixture_digest}"
required_provider = "core"
author_name = "Jacobian contributors"
difficulty = "medium"
category = "mathematics"
tags = ["polynomial-map", "coordination", "offline", "clean-room-verifier"]
case_version = "{CASE_VERSION}"
generator_version = "{GENERATOR_VERSION}"
derivation = "Issue #477 deterministic hand-auditable polynomial-map pilot fixture."

[agent]
timeout_sec = 600.0

[verifier]
timeout_sec = 120.0
environment_mode = "separate"

[environment]
network_mode = "no-network"
cpus = 1
memory_mb = 1024
storage_mb = 4096

[verifier.environment]
network_mode = "no-network"
cpus = 1
memory_mb = 1024
storage_mb = 4096
'''.encode()
    readme = f"""# jacobian/{slug}

{description}

## Case

- family: `{data["family"]}`
- case version: `{CASE_VERSION}`
- generator: `{GENERATOR_VERSION}`
- fixture digest: `{fixture_digest}`
- assurance ceiling: `CHECKED`
- note: {data["case_note"]}

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
capability or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
scope, input and artifact bindings, evidence digest, and assurance. `VERIFIED`
is unauthorized in this pilot and receives zero reward.
""".encode()
    instruction_base = f"""# Exact polynomial-map claim assessment

Assess the `{data["family"]}` claim frozen in `input.json` under exact rational
polynomial semantics. Supplied candidates, provider statuses, partial direction
checks, and search records are evidence to audit, not authority. Return the
terminal certificate described by `submission_schema.json`, bind it to the
exact claim, map, subject, semantics, scope, and checker identities in the
input, and write the mirrored JSON certificate to
`evidence/certificate.json` with its SHA-256 digest.

For an inverse claim, the certificate must expose both ordered composition
residual families. A Keller-condition certificate licenses only its exact
constant nonzero Jacobian claim. A bounded search licenses only an exact
collision witness, complete declared-grid exhaustion, or an honest
non-conclusion matching timeout or incomplete execution. Any mathematically
valid collision witness in the declared grid is acceptable.

The `conclusion` field is determined by the terminal `verdict`:

| Verdict | Conclusion |
|---|---|
| `VALID_TWO_SIDED_INVERSE` | `TRUE` |
| `INVALID_INVERSE_CANDIDATE` | `FALSE` |
| `KELLER_CONDITION_ONLY` | `TRUE` |
| `NOT_KELLER` | `FALSE` |
| `COLLISION_FOUND` | `TRUE` |
| `NO_COLLISION_IN_DECLARED_GRID` | `FALSE` |
| `UNKNOWN` | `UNKNOWN` |

Use any mathematical method. No external service or special tool is required.
"""
    instruction = render_instruction(contract, instruction_base).encode()
    environment_docker = f"FROM {IMAGE}\nCOPY input.json submission_schema.json /app/\nWORKDIR /app\n".encode()
    tests_docker = f'''FROM {IMAGE}
LABEL jacobian.checksum="{checksum}"
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
COPY verifier.py verifier_support.py public_contract.json input.json test.sh /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
'''.encode()
    solve_sh = b"""#!/bin/sh
set -eu
mkdir -p /app/evidence
cp /solution/submission.json /app/submission.json
cp /solution/certificate.json /app/evidence/certificate.json
cp /solution/answer.txt /app/answer.txt
"""
    member = f'''schema_version = "2"
task_id = "{slug}"
task_name = "jacobian/{slug}"
evaluation_kind = "workflow"
domain = "mathematical-sciences"
field = "algebra"
primary_domain = "algebra"
provenance_class = "deterministic-hand-auditable-pilot"
provenance_ref = "authored:symbolic-coordination-v1/{CASE_VERSION}#{slug}"
assurance_ceiling = "CHECKED"
required_provider = "core"
environment_profile = "core-python-minimal-verifier"
verifier_contract_version = "1"
evaluation_owner = "jacobian/symbolic-coordination-v1"
'''.encode()
    return {
        task / "README.md": readme,
        task / "instruction.md": instruction,
        task / "task.toml": task_toml,
        task / "environment/Dockerfile": environment_docker,
        task / "environment/input.json": input_content,
        task / "environment/submission_schema.json": schema_text.encode(),
        task / "solution/answer.txt": (
            str(submission["result"]["verdict"]) + "\n"
        ).encode(),
        task / "solution/certificate.json": evidence,
        task / "solution/submission.json": json_bytes(submission),
        task / "solution/solve.sh": solve_sh,
        task / "tests/Dockerfile": tests_docker,
        task / "tests/input.json": input_content,
        task / "tests/public_contract.json": (
            json.dumps(contract_value, indent=2, sort_keys=True) + "\n"
        ).encode(),
        task / "tests/test.sh": b"#!/bin/sh\nset -eu\nexec python /tests/verifier.py\n",
        task / "tests/verifier.py": verifier,
        task / "tests/verifier_support.py": support,
        DATASET / "members" / f"{slug}.toml": member,
    }


def expected_files() -> tuple[dict[Path, bytes], dict[str, object]]:
    verifier = VERIFIER_TEMPLATE.read_bytes()
    support = TEMPLATE_SUPPORT.read_bytes()
    files: dict[Path, bytes] = {}
    manifest_cases = []
    for raw_case in cases():
        data = bind_case(raw_case)
        rendered = render_task(data, verifier, support)
        overlap = files.keys() & rendered.keys()
        if overlap:
            raise AssertionError(f"duplicate generated paths: {sorted(overlap)}")
        files.update(rendered)
        manifest_cases.append(
            {
                "task_id": data["case_id"],
                "family": data["family"],
                "fixture_sha256": sha256_bytes(
                    rendered[DATASET / str(data["case_id"]) / "environment/input.json"]
                ),
                "subject_sha256": data["bindings"]["subject_sha256"],
            }
        )
    manifest = {
        "schema_version": "1",
        "dataset_id": "jacobian/symbolic-coordination-v1",
        "case_version": CASE_VERSION,
        "generator_version": GENERATOR_VERSION,
        "source_issue": "https://github.com/morluto/jacobian/issues/477",
        "generation": "deterministic-no-random-seed",
        "case_count": len(manifest_cases),
        "cases": manifest_cases,
    }
    files[DATASET / "pilot-manifest.json"] = json_bytes(manifest)
    return files, manifest


def run(check: bool) -> int:
    files, manifest = expected_files()
    failures = []
    if check:
        for path, expected in sorted(files.items()):
            if not path.is_file():
                failures.append(f"missing: {path.relative_to(ROOT)}")
            elif path.read_bytes() != expected:
                failures.append(f"drift: {path.relative_to(ROOT)}")
        expected_members = {f"{case['task_id']}.toml" for case in manifest["cases"]}
        actual_members = {path.name for path in (DATASET / "members").glob("*.toml")}
        for extra in sorted(actual_members - expected_members):
            failures.append(
                f"unexpected member: benchmarks/datasets/symbolic-coordination-v1/members/{extra}"
            )
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
        print(
            f"symbolic-coordination-v1: {manifest['case_count']} generated cases are current"
        )
        return 0
    for path, content in sorted(files.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    for case in manifest["cases"]:
        for relative in ("solution/solve.sh", "tests/test.sh"):
            (DATASET / str(case["task_id"]) / relative).chmod(0o755)
    print(f"rendered {manifest['case_count']} symbolic-coordination-v1 cases")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return run(args.check)


if __name__ == "__main__":
    raise SystemExit(main())
