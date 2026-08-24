"""Regression tests binding rational solution and inconsistency results to their source (#2294)."""

from __future__ import annotations

import copy
import json
from fractions import Fraction

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
from jacobian.math.matrices.rational_linear._operations import (
    compute_rational_inconsistency,
    compute_rational_solution,
)

pytestmark = pytest.mark.requires_backend("flint")


def _system(
    variables: list[str],
    entries: list[list[Fraction]],
    rhs: list[Fraction],
) -> dict[str, object]:
    return {
        "variables": variables,
        "coefficients": {"entries": [[q(value) for value in row] for row in entries]},
        "rhs": [q(value) for value in rhs],
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


def _mutable(dumped: dict) -> dict:
    """JSON round-trip so nested tuple payloads become mutable lists."""

    return json.loads(json.dumps(dumped))


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


def test_solution_result_rejects_forged_and_foreign_claims() -> None:
    """Mutated values or a mutated source fail the A x = b replay."""

    dumped = _mutable(
        compute_rational_solution(
            LinearRationalSolutionFindRequest.model_validate(
                {"system": _unique_system()}
            )
        ).model_dump()
    )
    assert dumped["status"] == "SOLUTION"

    forged_value = copy.deepcopy(dumped)
    forged_value["values"][0] = q(Fraction(7))
    with pytest.raises(ValidationError, match="A x = b"):
        LinearRationalSolutionResult.model_validate(forged_value)

    dropped_value = copy.deepcopy(dumped)
    dropped_value["values"] = [dumped["values"][0]]
    with pytest.raises(ValidationError, match="variable count"):
        LinearRationalSolutionResult.model_validate(dropped_value)

    foreign_source = copy.deepcopy(dumped)
    foreign_source["system"]["rhs"][0] = q(Fraction(6))
    with pytest.raises(ValidationError, match="A x = b"):
        LinearRationalSolutionResult.model_validate(foreign_source)


def test_inconsistent_result_rejects_forged_and_foreign_claims() -> None:
    """Mutated witnesses or pairings fail the y^T A = 0 and y^T b replays."""

    dumped = _mutable(
        compute_rational_inconsistency(
            LinearRationalInconsistencyFindRequest.model_validate(
                {"system": _inconsistent_system()}
            )
        ).model_dump()
    )
    assert dumped["status"] == "INCONSISTENT"
    witness = tuple(
        Fraction(int(value["num"]), int(value["den"]))
        for value in dumped["left_witness"]
    )
    true_pairing = sum(
        bound * coordinate
        for bound, coordinate in zip((Fraction(0), Fraction(1)), witness, strict=True)
    )

    forged_witness = copy.deepcopy(dumped)
    forged_witness["left_witness"][0] = q(witness[0] + 1)
    with pytest.raises(ValidationError, match=r"y\^T A = 0"):
        LinearRationalInconsistencyResult.model_validate(forged_witness)

    forged_pairing = copy.deepcopy(dumped)
    forged_pairing["rhs_pairing"] = q(true_pairing + 1)
    with pytest.raises(ValidationError, match=r"y\^T b"):
        LinearRationalInconsistencyResult.model_validate(forged_pairing)

    flat_witness = copy.deepcopy(dumped)
    flat_witness["left_witness"] = [q(Fraction(0)) for _ in dumped["left_witness"]]
    flat_witness["rhs_pairing"] = q(Fraction(0))
    with pytest.raises(ValidationError, match="nonzero"):
        LinearRationalInconsistencyResult.model_validate(flat_witness)

    dropped_coordinate = copy.deepcopy(dumped)
    dropped_coordinate["left_witness"] = [dumped["left_witness"][0]]
    with pytest.raises(ValidationError, match="source row count"):
        LinearRationalInconsistencyResult.model_validate(dropped_coordinate)

    foreign_source = copy.deepcopy(dumped)
    foreign_source["system"]["coefficients"]["entries"][1] = [
        q(Fraction(1)),
        q(Fraction(2)),
    ]
    with pytest.raises(ValidationError):
        LinearRationalInconsistencyResult.model_validate(foreign_source)


def test_consistent_outcome_rejects_bare_or_mutated_claims() -> None:
    """A consistent outcome carries no witness; flipped claims fail the replay."""

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
    bare_witness["left_witness"] = [q(Fraction(1)), q(Fraction(-1))]
    bare_witness["rhs_pairing"] = q(Fraction(1))
    with pytest.raises(ValidationError, match="agree with the result status"):
        LinearRationalInconsistencyResult.model_validate(bare_witness)

    flipped_status = copy.deepcopy(consistent)
    flipped_status["status"] = "INCONSISTENT"
    flipped_status["left_witness"] = [q(Fraction(1)), q(Fraction(1))]
    flipped_status["rhs_pairing"] = q(Fraction(1))
    with pytest.raises(ValidationError, match=r"y\^T A = 0"):
        LinearRationalInconsistencyResult.model_validate(flipped_status)


def test_inconsistent_outcome_requires_a_genuinely_inconsistent_source() -> None:
    """A mutated INCONSISTENT claim cannot attach to a consistent source."""

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
    with pytest.raises(ValidationError, match=r"rank\(A\) < rank"):
        LinearRationalSolutionResult.model_validate(flipped)

    identity = copy.deepcopy(dumped)
    identity["system"] = _system(
        ["x", "y"],
        [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]],
        [Fraction(0), Fraction(0)],
    )
    identity["status"] = "INCONSISTENT"
    identity["values"] = None
    with pytest.raises(ValidationError, match=r"rank\(A\) < rank"):
        LinearRationalSolutionResult.model_validate(identity)


def test_consistent_outcome_requires_a_genuinely_consistent_source() -> None:
    """A CONSISTENT claim cannot attach to a contradictory source system."""

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
    with pytest.raises(ValidationError, match=r"rank\(A\) == rank"):
        LinearRationalInconsistencyResult.model_validate(contradictory)


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


def test_source_bound_result_versions_track_wire_shape() -> None:
    """The required source fields bump all three affected declarations."""

    from jacobian.math.matrices._tools import TOOLS as MATRIX_TOOLS
    from jacobian.math.matrices.rational_linear._tools import TOOLS as LINEAR_TOOLS

    versions = {
        tool.operation_id: tool.version for tool in (*MATRIX_TOOLS, *LINEAR_TOOLS)
    }

    assert versions["matrix.rational_linear_system.solve"] == "3"
    assert versions["linear.rational_solution.compute"] == "3"
    assert versions["linear.rational_inconsistency.compute"] == "3"


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
