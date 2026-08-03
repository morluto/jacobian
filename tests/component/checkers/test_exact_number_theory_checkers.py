from __future__ import annotations

import copy
from collections.abc import Callable
from itertools import product
from typing import Any

import pytest
from tests.component.checkers.exact_domain_checker_support import (
    _NUMBER_THEORY_CASES,
    _request,
)
from tests.unit.contracts.artifacts import canonical_digest as _digest

from jacobian_checkers.exact_domain_operations import (
    check_integer_powerful_number,
    check_integer_prime_factorization,
    check_modular_polynomial_residue_image,
)


def _modular_checker_request() -> dict[str, Any]:
    return copy.deepcopy(
        next(
            checker_request
            for checker, checker_request in _NUMBER_THEORY_CASES
            if checker is check_modular_polynomial_residue_image
        )
    )


def test_modular_residue_checker_reports_exhaustive_integer_replay() -> None:
    decision = check_modular_polynomial_residue_image(_modular_checker_request())

    assert decision["accepted"] is True
    assert decision["arithmetic"] == "EXACT_INTEGER"
    assert decision["method"] == "EXHAUSTIVE_FINITE"
    assert decision["coverage"] == "EXHAUSTIVE"


def test_modular_residue_checker_accepts_exact_assignment_bound() -> None:
    assignments = [list(values) for values in product(range(16), repeat=3)]
    residues = [
        assignment[0] * assignment[1] * assignment[2] % 16 for assignment in assignments
    ]
    image = sorted(set(residues))
    first_assignments: dict[int, list[int]] = {}
    for assignment, residue in zip(assignments, residues, strict=True):
        first_assignments.setdefault(residue, assignment)
    checker_request = _request(
        "modular.polynomial_residue_image.compute",
        "modular.polynomial-residue-image.flint-replay",
        {
            "modulus": 16,
            "variables": [
                {"name": "x", "residues": list(range(16))},
                {"name": "y", "residues": list(range(16))},
                {"name": "z", "residues": list(range(16))},
            ],
            "terms": [{"coefficient": "1", "exponents": [1, 1, 1]}],
        },
        {
            "semantics_version": "modular-polynomial-residue-image.v1",
            "modulus": 16,
            "variable_order": ["x", "y", "z"],
            "domains": [list(range(16)), list(range(16)), list(range(16))],
            "normalized_terms": [{"coefficient": 1, "exponents": [1, 1, 1]}],
            "enumeration_scope": "COMPLETE_DECLARED_CARTESIAN_PRODUCT",
            "total_assignments": 4_096,
            "image": image,
            "residue_counts": [
                {"residue": residue, "count": residues.count(residue)}
                for residue in image
            ],
            "witnesses": [
                {
                    "residue": residue,
                    "assignment": first_assignments[residue],
                }
                for residue in image
            ],
            "table": [
                {"assignment": assignment, "residue": residue}
                for assignment, residue in zip(assignments, residues, strict=True)
            ],
        },
    )

    decision = check_modular_polynomial_residue_image(checker_request)

    assert decision["accepted"] is True
    assert decision["method"] == "EXHAUSTIVE_FINITE"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda result: result["table"].pop(),
        lambda result: result["table"][1].update(residue=3),
        lambda result: result["residue_counts"][1].update(count=2),
        lambda result: result["witnesses"][2].update(assignment=[2]),
        lambda result: result.update(variable_order=["y"]),
        lambda result: result["normalized_terms"][0].update(coefficient=3),
    ),
    ids=(
        "partial-table",
        "wrong-evaluation",
        "wrong-multiplicity",
        "nonfirst-witness",
        "wrong-variable-order",
        "wrong-normalization",
    ),
)
def test_modular_residue_checker_rejects_one_obligation_mutation(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    checker_request = _modular_checker_request()
    mutate(checker_request["candidate"]["payload"])
    checker_request["candidate"]["payload_digest"] = _digest(
        checker_request["candidate"]["payload"]
    )

    decision = check_modular_polynomial_residue_image(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_modular_residue_checker_rejects_wrong_bound_scope() -> None:
    checker_request = _modular_checker_request()
    checker_request["scope"] = {
        "assignment_count": 7,
        "enumeration": "COMPLETE_DECLARED_CARTESIAN_PRODUCT",
    }

    decision = check_modular_polynomial_residue_image(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize(
    ("value", "factors"),
    (
        ("1", []),
        ("-1", []),
        ("2", [{"prime": "2", "power": 1}]),
        (
            "-360",
            [
                {"prime": "2", "power": 3},
                {"prime": "3", "power": 2},
                {"prime": "5", "power": 1},
            ],
        ),
    ),
)
def test_prime_factorization_checker_accepts_exact_boundaries(
    value: str,
    factors: list[dict[str, object]],
) -> None:
    checker_request = _request(
        "integer.compute.prime_factorization",
        "integer.prime-factorization.flint-replay",
        {"value": value, "resource_budget": {"wall_seconds": 5}},
        {"factors": factors},
    )

    decision = check_integer_prime_factorization(checker_request)

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda factors: factors.__setitem__(
            slice(None),
            [{"prime": "6", "power": 1}, {"prime": "60", "power": 1}],
        ),
        lambda factors: factors.pop(),
        lambda factors: factors.append({"prime": "5", "power": 1}),
        lambda factors: factors[0].update(power=2),
        lambda factors: factors.reverse(),
        lambda factors: factors[0].update(prime="-2"),
    ),
    ids=(
        "composite-bases",
        "missing-factor",
        "duplicate-base",
        "wrong-power",
        "noncanonical-order",
        "negative-base",
    ),
)
def test_prime_factorization_checker_rejects_false_or_noncanonical_factors(
    mutate: Callable[[list[dict[str, object]]], object],
) -> None:
    checker_request = _request(
        "integer.compute.prime_factorization",
        "integer.prime-factorization.flint-replay",
        {"value": "360", "resource_budget": {"wall_seconds": 5}},
        {
            "factors": [
                {"prime": "2", "power": 3},
                {"prime": "3", "power": 2},
                {"prime": "5", "power": 1},
            ]
        },
    )
    mutate(checker_request["candidate"]["payload"]["factors"])
    checker_request["candidate"]["payload_digest"] = _digest(
        checker_request["candidate"]["payload"]
    )

    decision = check_integer_prime_factorization(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_prime_factorization_checker_rejects_zero_source() -> None:
    checker_request = _request(
        "integer.compute.prime_factorization",
        "integer.prime-factorization.flint-replay",
        {"value": "0", "resource_budget": {"wall_seconds": 5}},
        {"factors": []},
    )

    assert check_integer_prime_factorization(checker_request)["accepted"] is False


@pytest.mark.parametrize(
    ("value", "result"),
    (
        (
            "1",
            {
                "semantics_version": (
                    "powerful-number.prime-exponents-at-least-two.v1"
                ),
                "is_powerful": True,
                "factors": [],
                "violating_primes": [],
            },
        ),
        (
            "72",
            {
                "semantics_version": (
                    "powerful-number.prime-exponents-at-least-two.v1"
                ),
                "is_powerful": True,
                "factors": [
                    {"prime": "2", "power": 3},
                    {"prime": "3", "power": 2},
                ],
                "violating_primes": [],
            },
        ),
        (
            "12",
            {
                "semantics_version": (
                    "powerful-number.prime-exponents-at-least-two.v1"
                ),
                "is_powerful": False,
                "factors": [
                    {"prime": "2", "power": 2},
                    {"prime": "3", "power": 1},
                ],
                "violating_primes": ["3"],
            },
        ),
    ),
)
def test_powerful_number_checker_accepts_exact_decisions(
    value: str,
    result: dict[str, object],
) -> None:
    checker_request = _request(
        "integer.decide.powerful",
        "integer.powerful.flint-replay",
        {"value": value, "resource_budget": {"wall_seconds": 5}},
        result,
    )

    decision = check_integer_powerful_number(checker_request)

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda result: result.update(is_powerful=False),
        lambda result: result["violating_primes"].append("2"),
        lambda result: result["factors"].pop(),
        lambda result: result.update(
            is_powerful=False,
            factors=[
                {"prime": "2", "power": 1},
                {"prime": "6", "power": 2},
            ],
            violating_primes=["2"],
        ),
        lambda result: result.update(semantics_version="powerful-number.v2"),
        lambda result: result["factors"].reverse(),
    ),
    ids=(
        "wrong-decision",
        "wrong-violations",
        "incomplete-factorization",
        "composite-factor-base",
        "wrong-semantics",
        "noncanonical-factor-order",
    ),
)
def test_powerful_number_checker_rejects_false_or_rebound_results(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    checker_request = _request(
        "integer.decide.powerful",
        "integer.powerful.flint-replay",
        {"value": "72", "resource_budget": {"wall_seconds": 5}},
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": True,
            "factors": [
                {"prime": "2", "power": 3},
                {"prime": "3", "power": 2},
            ],
            "violating_primes": [],
        },
    )
    result = checker_request["candidate"]["payload"]
    mutate(result)
    checker_request["candidate"]["payload_digest"] = _digest(result)

    decision = check_integer_powerful_number(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize("value", ("0", "-72"))
def test_powerful_number_checker_rejects_nonpositive_source(value: str) -> None:
    checker_request = _request(
        "integer.decide.powerful",
        "integer.powerful.flint-replay",
        {"value": value, "resource_budget": {"wall_seconds": 5}},
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": True,
            "factors": [],
            "violating_primes": [],
        },
    )

    assert check_integer_powerful_number(checker_request)["accepted"] is False
