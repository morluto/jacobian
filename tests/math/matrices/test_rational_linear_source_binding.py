"""Regression tests binding rational solution and inconsistency results to their source (#2294)."""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from typing import Any, cast

import pytest
from pydantic import ValidationError
from tests.support.rationals import rational_payload as q

from jacobian.math.matrices.rational_linear._models import (
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

pytestmark = pytest.mark.requires_backend("flint")


def _q(value: Fraction) -> dict[str, str]:
    return q(value.numerator, value.denominator)


def _system(
    variables: list[str],
    entries: list[list[Fraction]],
    rhs: list[Fraction],
) -> dict[str, object]:
    return {
        "variables": variables,
        "coefficients": {"entries": [[_q(value) for value in row] for row in entries]},
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
    for row, bound in zip(
        system.coefficients.entries,
        (value.as_fraction() for value in system.rhs),
        strict=True,
    ):
        residual = sum(
            coefficient.as_fraction() * component
            for coefficient, component in zip(row, components, strict=True)
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
    for column in range(len(system.coefficients.entries[0])):
        assert (
            sum(
                row[column].as_fraction() * coordinate
                for row, coordinate in zip(
                    system.coefficients.entries,
                    coordinates,
                    strict=True,
                )
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
