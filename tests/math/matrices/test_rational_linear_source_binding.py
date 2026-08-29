"""Regression tests binding rational solution and inconsistency results to their source (#2294)."""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from itertools import islice
from math import log10
from typing import Any, cast

import pytest
from pydantic import ValidationError
from sympy import primerange
from tests.support.rationals import rational_payload as q

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.rational_linear._models import (
    MAX_LINEAR_DIMENSION,
    LinearRationalInconsistencyFindRequest,
    LinearRationalInconsistencyResult,
    LinearRationalSolutionFindRequest,
    LinearRationalSolutionResult,
    LinearRationalSystem,
)
from jacobian.math.matrices.rational_linear._tools import (
    compute_rational_inconsistency,
    compute_rational_solution,
)
from jacobian.math.matrices.values import (
    RationalMatrix,
    SparseRationalMatrix,
    dense_rational_matrix_from_sparse,
    sparse_rational_matrix_from_dense,
)


def _q(value: Fraction) -> dict[str, str]:
    return q(value.numerator, value.denominator)


def _system(
    variables: list[str],
    entries: list[list[Fraction]],
    rhs: list[Fraction],
) -> dict[str, object]:
    return {
        "variables": variables,
        "coefficients": {
            "row_count": len(entries),
            "column_count": len(variables),
            "entries": [
                {"row": row, "column": column, "value": _q(value)}
                for row, values in enumerate(entries)
                for column, value in enumerate(values)
                if value
            ],
        },
        "rhs": [_q(value) for value in rhs],
    }


def _unique_system() -> dict[str, object]:
    return _system(
        ["x", "y"],
        [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(-1)]],
        [Fraction(5), Fraction(1)],
    )


def _inconsistent_system() -> dict[str, object]:
    return _system(
        ["x", "y"],
        [[Fraction(1), Fraction(1)], [Fraction(1), Fraction(1)]],
        [Fraction(0), Fraction(1)],
    )


def test_sparse_rational_matrix_round_trips_through_its_dense_owner() -> None:
    dense = RationalMatrix.model_validate(
        {
            "entries": [
                [_q(Fraction(0)), _q(Fraction(2)), _q(Fraction(0))],
                [_q(Fraction(0)), _q(Fraction(0)), _q(Fraction(0))],
            ]
        }
    )
    sparse = sparse_rational_matrix_from_dense(dense)
    restored = SparseRationalMatrix.model_validate(sparse.model_dump(mode="json"))

    assert restored.row_count == 2
    assert restored.column_count == 3
    assert tuple((entry.row, entry.column) for entry in restored.entries) == ((0, 1),)
    assert dense_rational_matrix_from_sparse(restored) == dense


def _underdetermined_system() -> dict[str, object]:
    return _system(
        ["x", "y", "z"],
        [[Fraction(1), Fraction(1), Fraction(0)]],
        [Fraction(1)],
    )


def _mutable(dumped: dict[str, Any]) -> dict[str, Any]:
    """JSON round-trip so nested tuple payloads become mutable lists."""

    return cast(dict[str, Any], json.loads(json.dumps(dumped)))


def test_producer_results_retain_their_source_system() -> None:
    """Every outcome retains the exact declared system, including degenerate shapes."""

    for payload in (
        _unique_system(),
        _inconsistent_system(),
        _underdetermined_system(),
    ):
        request = LinearRationalSolutionFindRequest.model_validate({"system": payload})
        solution = compute_rational_solution(request)
        assert solution.system == request.system

        dual_request = LinearRationalInconsistencyFindRequest.model_validate(
            {"system": payload}
        )
        inconsistency = compute_rational_inconsistency(dual_request)
        assert inconsistency.system == dual_request.system


def test_solution_result_replays_against_the_source() -> None:
    """The admitted solution satisfies A x = b exactly on the retained system."""

    system = LinearRationalSystem.model_validate(_unique_system())
    result = compute_rational_solution(LinearRationalSolutionFindRequest(system=system))

    assert result.status == "SOLUTION"
    assert result.values is not None
    assert len(result.values) == len(system.variables)
    components = [value.as_fraction() for value in result.values]
    coefficient_map = {
        (item.row, item.column): item.value.as_fraction()
        for item in system.coefficients.entries
    }
    for row, bound in zip(
        range(system.coefficients.row_count),
        (value.as_fraction() for value in system.rhs),
        strict=True,
    ):
        residual = sum(
            coefficient_map.get((row, column), Fraction(0)) * component
            for column, component in enumerate(components)
        )
        assert residual == bound


def test_inconsistent_result_replays_witness_relations() -> None:
    """The separating witness annihilates every column with a nonzero pairing."""

    system = LinearRationalSystem.model_validate(_inconsistent_system())
    result = compute_rational_inconsistency(
        LinearRationalInconsistencyFindRequest(system=system)
    )

    assert result.status == "INCONSISTENT"
    assert result.left_witness is not None
    assert result.rhs_pairing is not None
    assert len(result.left_witness) == len(system.rhs)
    coordinates = [value.as_fraction() for value in result.left_witness]
    coefficient_map = {
        (item.row, item.column): item.value.as_fraction()
        for item in system.coefficients.entries
    }
    for column in range(len(system.variables)):
        assert (
            sum(
                coefficient_map.get((row, column), Fraction(0)) * coordinate
                for row, coordinate in enumerate(coordinates)
            )
            == 0
        )
    pairing = sum(
        bound.as_fraction() * coordinate
        for bound, coordinate in zip(system.rhs, coordinates, strict=True)
    )
    assert pairing == result.rhs_pairing.as_fraction()
    assert pairing != 0


@pytest.mark.parametrize("payload", (_unique_system(), _inconsistent_system()))
def test_serialized_results_round_trip(payload: dict[str, object]) -> None:
    """Producer output validates through the JSON wire shape."""

    solution = compute_rational_solution(
        LinearRationalSolutionFindRequest.model_validate({"system": payload})
    )
    assert (
        LinearRationalSolutionResult.model_validate_json(solution.model_dump_json())
        == solution
    )

    inconsistency = compute_rational_inconsistency(
        LinearRationalInconsistencyFindRequest.model_validate({"system": payload})
    )
    assert (
        LinearRationalInconsistencyResult.model_validate_json(
            inconsistency.model_dump_json()
        )
        == inconsistency
    )


def test_solution_result_rejects_wrong_coordinate_count() -> None:
    """Mutated values or a mutated source fail the A x = b replay."""

    dumped = _mutable(
        compute_rational_solution(
            LinearRationalSolutionFindRequest.model_validate(
                {"system": _unique_system()}
            )
        ).model_dump()
    )
    assert dumped["status"] == "SOLUTION"

    dropped_value = copy.deepcopy(dumped)
    dropped_value["values"] = [dumped["values"][0]]
    with pytest.raises(ValidationError):
        LinearRationalSolutionResult.model_validate(dropped_value)


def test_inconsistent_result_rejects_wrong_witness_shape() -> None:

    dumped = _mutable(
        compute_rational_inconsistency(
            LinearRationalInconsistencyFindRequest.model_validate(
                {"system": _inconsistent_system()}
            )
        ).model_dump()
    )
    assert dumped["status"] == "INCONSISTENT"
    dropped_coordinate = copy.deepcopy(dumped)
    dropped_coordinate["left_witness"] = [dumped["left_witness"][0]]
    with pytest.raises(ValidationError):
        LinearRationalInconsistencyResult.model_validate(dropped_coordinate)


def test_consistent_outcome_rejects_a_bare_witness() -> None:
    """A consistent outcome cannot carry an inconsistency witness."""

    consistent = _mutable(
        compute_rational_inconsistency(
            LinearRationalInconsistencyFindRequest.model_validate(
                {"system": _unique_system()}
            )
        ).model_dump()
    )
    assert consistent["status"] == "CONSISTENT"
    assert consistent["left_witness"] is None
    assert consistent["rhs_pairing"] is None

    bare_witness = copy.deepcopy(consistent)
    bare_witness["left_witness"] = [_q(Fraction(1)), _q(Fraction(-1))]
    bare_witness["rhs_pairing"] = _q(Fraction(1))
    with pytest.raises(ValidationError):
        LinearRationalInconsistencyResult.model_validate(bare_witness)


def test_negative_solution_status_is_structural_not_a_backend_replay() -> None:
    """A no-solution conclusion is constructed by the bounded owner kernel."""

    dumped = _mutable(
        compute_rational_solution(
            LinearRationalSolutionFindRequest.model_validate(
                {"system": _unique_system()}
            )
        ).model_dump()
    )
    assert dumped["status"] == "SOLUTION"

    flipped = copy.deepcopy(dumped)
    flipped["status"] = "INCONSISTENT"
    flipped["values"] = None
    assert LinearRationalSolutionResult.model_validate(flipped).status == "INCONSISTENT"

    identity = copy.deepcopy(dumped)
    identity["system"] = _system(
        ["x", "y"],
        [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]],
        [Fraction(0), Fraction(0)],
    )
    identity["status"] = "INCONSISTENT"
    identity["values"] = None
    assert (
        LinearRationalSolutionResult.model_validate(identity).status == "INCONSISTENT"
    )


def test_negative_inconsistency_status_is_structural_not_a_backend_replay() -> None:
    """A no-witness conclusion is constructed by the bounded owner kernel."""

    consistent = _mutable(
        compute_rational_inconsistency(
            LinearRationalInconsistencyFindRequest.model_validate(
                {"system": _unique_system()}
            )
        ).model_dump()
    )
    assert consistent["status"] == "CONSISTENT"

    contradictory = copy.deepcopy(consistent)
    contradictory["system"] = _system(
        ["x"],
        [[Fraction(1)], [Fraction(1)]],
        [Fraction(0), Fraction(1)],
    )
    assert (
        LinearRationalInconsistencyResult.model_validate(contradictory).status
        == "CONSISTENT"
    )


def test_negative_outcomes_round_trip_on_their_true_sources() -> None:
    """Genuine no-witness outcomes replay successfully against their own system."""

    solution = compute_rational_solution(
        LinearRationalSolutionFindRequest.model_validate(
            {"system": _inconsistent_system()}
        )
    )
    assert solution.status == "INCONSISTENT"
    assert (
        LinearRationalSolutionResult.model_validate_json(solution.model_dump_json())
        == solution
    )

    inconsistency = compute_rational_inconsistency(
        LinearRationalInconsistencyFindRequest.model_validate(
            {"system": _underdetermined_system()}
        )
    )
    assert inconsistency.status == "CONSISTENT"
    assert (
        LinearRationalInconsistencyResult.model_validate_json(
            inconsistency.model_dump_json()
        )
        == inconsistency
    )


def test_polynomial_coordinate_composition_reconstructs_the_target() -> None:
    """A Ringel-style span solve stays bound: A c = t reconstructs the target."""

    # Declared generator functionals in monomial coordinates, one per column.
    generators = (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(1)),
    )
    target = (Fraction(3), Fraction(2))
    payload = _system(
        ["c1", "c2", "c3"],
        [list(column) for column in zip(*generators, strict=True)],
        list(target),
    )

    result = compute_rational_solution(
        LinearRationalSolutionFindRequest.model_validate({"system": payload})
    )
    assert result.status == "SOLUTION"
    assert result.system == LinearRationalSystem.model_validate(payload)
    weights = [value.as_fraction() for value in result.values] if result.values else []
    reconstructed = tuple(
        sum(
            weight * generator[index]
            for weight, generator in zip(weights, generators, strict=True)
        )
        for index in range(len(target))
    )
    assert reconstructed == target


def test_separating_functional_annihilates_generators_but_not_the_target() -> None:
    """A Ringel-style separation stays bound: y^T A = 0 while y^T b != 0."""

    generators = (
        (Fraction(1), Fraction(0)),
        (Fraction(2), Fraction(0)),
    )
    target = (Fraction(0), Fraction(1))
    payload = _system(
        ["c1", "c2"],
        [list(column) for column in zip(*generators, strict=True)],
        list(target),
    )

    result = compute_rational_inconsistency(
        LinearRationalInconsistencyFindRequest.model_validate({"system": payload})
    )
    assert result.status == "INCONSISTENT"
    assert result.left_witness is not None
    coordinates = [value.as_fraction() for value in result.left_witness]
    for generator in generators:
        assert sum(y * g for y, g in zip(coordinates, generator, strict=True)) == 0
    assert sum(y * t for y, t in zip(coordinates, target, strict=True)) != 0


def test_sparse_diagonal_system_scales_beyond_the_dense_32_axis() -> None:
    dimension = 128
    payload = {
        "variables": [f"x_{index}" for index in range(dimension)],
        "coefficients": {
            "row_count": dimension,
            "column_count": dimension,
            "entries": [
                {"row": index, "column": index, "value": _q(Fraction(index + 1))}
                for index in range(dimension)
            ],
        },
        "rhs": [_q(Fraction(index + 1)) for index in range(dimension)],
    }
    result = compute_rational_solution(
        LinearRationalSolutionFindRequest.model_validate({"system": payload})
    )
    assert result.status == "SOLUTION"
    assert result.values == tuple(
        CanonicalRational(num="1", den="1") for _ in range(dimension)
    )


def test_sparse_kernel_agrees_with_dense_sympy_on_the_overlap() -> None:
    from sympy import Matrix, Rational

    payload = _unique_system()
    result = compute_rational_solution(
        LinearRationalSolutionFindRequest.model_validate({"system": payload})
    )
    dense_solution, parameters = Matrix([[2, 1], [1, -1]]).gauss_jordan_solve(
        Matrix([5, 1])
    )
    assert parameters.rows == 0
    assert result.values is not None
    assert all(isinstance(value, Rational) for value in dense_solution)
    assert tuple(value.as_fraction() for value in result.values) == tuple(
        Fraction(int(value.p), int(value.q)) for value in dense_solution
    )


def test_result_sensitive_admission_keeps_a_wide_one_row_system() -> None:
    dimension = 1_024
    payload = {
        "variables": [f"x_{index}" for index in range(dimension)],
        "coefficients": {
            "row_count": 1,
            "column_count": dimension,
            "entries": [{"row": 0, "column": dimension - 1, "value": _q(Fraction(1))}],
        },
        "rhs": [_q(Fraction(7))],
    }
    result = compute_rational_solution(
        LinearRationalSolutionFindRequest.model_validate({"system": payload})
    )
    assert result.values is not None
    assert len(result.values) == dimension
    assert result.values[-1] == CanonicalRational(num="7", den="1")
    assert all(value.num == "0" for value in result.values[:-1])


@pytest.mark.parametrize(
    "coefficients",
    (
        [
            {"row": 0, "column": 0, "value": _q(Fraction(1))},
            {"row": 0, "column": 0, "value": _q(Fraction(2))},
        ],
        [{"row": 0, "column": 0, "value": _q(Fraction(0))}],
    ),
)
def test_sparse_coefficients_reject_noncanonical_storage(
    coefficients: list[dict[str, object]],
) -> None:
    with pytest.raises(ValidationError):
        LinearRationalSystem.model_validate(
            {
                "variables": ["x"],
                "coefficients": {
                    "row_count": 1,
                    "column_count": 1,
                    "entries": coefficients,
                },
                "rhs": [_q(Fraction(0))],
            }
        )


def test_sparse_work_budget_rejects_large_fill_envelope() -> None:
    dimension = 500
    system = LinearRationalSystem.model_validate(
        {
            "variables": [f"x_{index}" for index in range(dimension)],
            "coefficients": {
                "row_count": dimension,
                "column_count": dimension,
                "entries": [],
            },
            "rhs": [_q(Fraction(0)) for _ in range(dimension)],
        }
    )
    with pytest.raises(OperationDomainValidationError, match="scalar-work budget"):
        compute_rational_solution(LinearRationalSolutionFindRequest(system=system))


def test_result_deserialization_does_not_repeat_semantic_admission() -> None:
    dimension = 500
    system = LinearRationalSystem.model_validate(
        {
            "variables": [f"x_{index}" for index in range(dimension)],
            "coefficients": {
                "row_count": dimension,
                "column_count": dimension,
                "entries": [],
            },
            "rhs": [_q(Fraction(0)) for _ in range(dimension)],
        }
    )
    restored = LinearRationalSolutionResult.model_validate(
        {"system": system.model_dump(mode="json"), "status": "INCONSISTENT"}
    )
    assert restored.system == system


def test_sparse_result_height_rejects_before_backend_expansion() -> None:
    denominators = tuple(islice(primerange(1_000_000_000, 2_000_000_000), 64))
    system = LinearRationalSystem.model_validate(
        {
            "variables": [f"x_{index}" for index in range(64)],
            "coefficients": {
                "row_count": 64,
                "column_count": 64,
                "entries": [
                    {
                        "row": row,
                        "column": column,
                        "value": _q(Fraction(1, denominators[column])),
                    }
                    for row in range(64)
                    for column in range(64)
                ],
            },
            "rhs": [_q(Fraction(0)) for _ in range(64)],
        }
    )
    with pytest.raises(OperationDomainValidationError, match="result-height bound"):
        compute_rational_solution(LinearRationalSolutionFindRequest(system=system))


def _primes_whose_product_exceeds_result_height() -> tuple[int, ...]:
    """Distinct small primes whose primorial exceeds the canonical height cap."""

    primes: list[int] = []
    log10_product = 0.0
    for prime in primerange(2, 200_000):
        primes.append(int(prime))
        log10_product += log10(prime)
        if (
            log10_product > MAX_CANONICAL_RATIONAL_DIGITS
            and len(primes) <= MAX_LINEAR_DIMENSION
        ):
            return tuple(primes)
    raise AssertionError("could not assemble a dual-height-exceeding prime list")


def _tall_reciprocal_prime_system() -> dict[str, object]:
    """One-variable system ``(1/p_i) x = 1/p_i``; the exact solution is ``x = 1``."""

    primes = _primes_whose_product_exceeds_result_height()
    return {
        "variables": ["x"],
        "coefficients": {
            "row_count": len(primes),
            "column_count": 1,
            "entries": [
                {"row": index, "column": 0, "value": _q(Fraction(1, prime))}
                for index, prime in enumerate(primes)
            ],
        },
        "rhs": [_q(Fraction(1, prime)) for prime in primes],
    }


def test_solution_admission_excludes_dual_witness_height() -> None:
    """A tall reciprocal-prime identity is admitted for the solution postcondition."""

    system = LinearRationalSystem.model_validate(_tall_reciprocal_prime_system())
    result = compute_rational_solution(LinearRationalSolutionFindRequest(system=system))

    assert result.status == "SOLUTION"
    assert result.values == (CanonicalRational(num="1", den="1"),)


def test_inconsistency_admission_still_enforces_witness_height() -> None:
    """The same dual primorial still rejects the left-witness postcondition."""

    system = LinearRationalSystem.model_validate(_tall_reciprocal_prime_system())
    with pytest.raises(OperationDomainValidationError, match="result-height bound"):
        compute_rational_inconsistency(
            LinearRationalInconsistencyFindRequest(system=system)
        )


def test_witness_admission_excludes_primal_solution_height() -> None:
    """A wide reciprocal-prime row is admitted for the inconsistency postcondition."""

    primes = _primes_whose_product_exceeds_result_height()
    payload = {
        "variables": [f"x_{index}" for index in range(len(primes))],
        "coefficients": {
            "row_count": 1,
            "column_count": len(primes),
            "entries": [
                {"row": 0, "column": index, "value": _q(Fraction(1, prime))}
                for index, prime in enumerate(primes)
            ],
        },
        "rhs": [_q(Fraction(1, primes[0]))],
    }
    system = LinearRationalSystem.model_validate(payload)
    result = compute_rational_inconsistency(
        LinearRationalInconsistencyFindRequest(system=system)
    )

    assert result.status == "CONSISTENT"
    with pytest.raises(OperationDomainValidationError, match="result-height bound"):
        compute_rational_solution(LinearRationalSolutionFindRequest(system=system))
