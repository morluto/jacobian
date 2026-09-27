import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieAlgebraElement,
    LieGeneratedIdealRequest,
    LieIdeal,
    LieIdealRequest,
    LieQuotientRequest,
    LieSubalgebraRequest,
)
from jacobian.math.lie_algebras.operations import (
    check_ideal,
    check_subalgebra,
    lie_generated_ideal,
    lie_quotient,
)

HEISENBERG = FiniteDimensionalLieAlgebra.model_validate(
    {
        "basis": ["x", "y", "z"],
        "structure_constants": [
            {"i": 0, "j": 1, "k": 2, "coefficient": {"num": 1, "den": 1}}
        ],
    }
)


def _element(*coordinates: int) -> LieAlgebraElement:
    return LieAlgebraElement.model_validate(
        {
            "basis": HEISENBERG.basis,
            "coordinates": [{"num": value, "den": 1} for value in coordinates],
        }
    )


def _rows(value: LieIdeal) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(entry.as_fraction() for entry in row) for row in value.generators.entries
    )


def test_heisenberg_ideal_closure_and_existing_consumers() -> None:
    ideal = lie_generated_ideal(HEISENBERG, (_element(1, 0, 0),))

    assert ideal.algebra == HEISENBERG
    assert ideal.basis == HEISENBERG.basis
    assert _rows(ideal) == (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    assert check_ideal(HEISENBERG, ideal).is_ideal
    assert check_subalgebra(HEISENBERG, ideal).is_subalgebra
    quotient = lie_quotient(HEISENBERG, ideal, ("q",))
    assert quotient.ideal.basis == ideal.basis
    assert quotient.ideal.generators == ideal.generators
    assert quotient.quotient.structure_constants == ()
    assert LieIdeal.model_validate(ideal.model_dump()) == ideal

    wire_value = ideal.model_dump()
    assert LieIdealRequest.model_validate(
        {"algebra": HEISENBERG.model_dump(), "candidate": wire_value}
    )
    assert LieSubalgebraRequest.model_validate(
        {"algebra": HEISENBERG.model_dump(), "candidate": wire_value}
    )
    assert LieQuotientRequest.model_validate(
        {
            "algebra": HEISENBERG.model_dump(),
            "ideal": wire_value,
            "quotient_basis": ["q"],
        }
    )
    assert check_ideal(HEISENBERG, wire_value).is_ideal
    assert check_subalgebra(HEISENBERG, wire_value).is_subalgebra
    assert lie_quotient(HEISENBERG, wire_value, ("q",)).quotient == quotient.quotient


def test_empty_generators_return_the_source_bound_zero_ideal() -> None:
    ideal = lie_generated_ideal(HEISENBERG, [])

    assert ideal.algebra == HEISENBERG
    assert ideal.generators.row_count == 0
    assert ideal.generators.column_count == len(HEISENBERG.basis)
    assert check_ideal(HEISENBERG, ideal).is_ideal


def test_rational_generator_coordinates_produce_exact_rref_rows() -> None:
    rational_x = LieAlgebraElement.model_validate(
        {
            "basis": HEISENBERG.basis,
            "coordinates": [
                {"num": 2, "den": 3},
                {"num": 0, "den": 1},
                {"num": 0, "den": 1},
            ],
        }
    )
    ideal = lie_generated_ideal(HEISENBERG, [rational_x])

    assert _rows(ideal) == (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )


def test_basis_generators_return_the_full_ideal() -> None:
    ideal = lie_generated_ideal(
        HEISENBERG,
        [_element(1, 0, 0), _element(0, 1, 0), _element(0, 0, 1)],
    )

    assert ideal.generators.row_count == len(HEISENBERG.basis)
    assert check_ideal(HEISENBERG, ideal).is_ideal


def test_empty_generators_on_a_six_dimensional_algebra_return_zero_ideal() -> None:
    algebra = FiniteDimensionalLieAlgebra.model_validate(
        {"basis": [f"e{i}" for i in range(6)], "structure_constants": []}
    )

    ideal = lie_generated_ideal(algebra, [])

    assert ideal.algebra == algebra
    assert ideal.generators.row_count == 0
    assert ideal.generators.column_count == 6
    assert check_ideal(algebra, ideal).is_ideal


def test_six_dimensional_single_generator_ideal_closes_exactly() -> None:
    algebra = FiniteDimensionalLieAlgebra.model_validate(
        {
            "basis": [f"e{i}" for i in range(6)],
            "structure_constants": [
                {"i": 0, "j": 1, "k": 1, "coefficient": {"num": 1, "den": 1}}
            ],
        }
    )
    generator = LieAlgebraElement.model_validate(
        {
            "basis": [f"e{i}" for i in range(6)],
            "coordinates": [
                {"num": 1 if position == 0 else 0, "den": 1} for position in range(6)
            ],
        }
    )

    ideal = lie_generated_ideal(algebra, [generator])

    assert _rows(ideal) == (
        (Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
    )
    assert check_ideal(algebra, ideal).is_ideal


def test_operation_rejects_generators_on_a_different_axis() -> None:
    alien = LieAlgebraElement.model_validate(
        {"basis": ["a", "b", "c"], "coordinates": [{"num": 1, "den": 1}] * 3}
    )
    with pytest.raises(
        OperationDomainValidationError, match="ideal generators must use"
    ) as exc_info:
        lie_generated_ideal(HEISENBERG, [alien])
    assert (
        exc_info.value.errors()[0]["type"]
        == "lie_algebra.generated_ideal_generator_basis"
    )


def test_source_bound_ideal_rejects_another_algebra_with_same_axis() -> None:
    ideal = lie_generated_ideal(HEISENBERG, [_element(1, 0, 0)])
    same_basis_different_bracket = FiniteDimensionalLieAlgebra.model_validate(
        {"basis": ["x", "y", "z"], "structure_constants": []}
    )

    with pytest.raises(OperationDomainValidationError) as exc_info:
        check_ideal(same_basis_different_bracket, ideal)
    assert exc_info.value.errors()[0]["type"] == "lie_algebra.candidate_source"


def test_request_rejects_generators_on_another_basis_axis() -> None:
    foreign = LieAlgebraElement.model_validate(
        {
            "basis": ["a", "b", "c"],
            "coordinates": [{"num": 1, "den": 1}] * 3,
        }
    )
    with pytest.raises(ValidationError, match="ideal generators must use"):
        LieGeneratedIdealRequest(algebra=HEISENBERG, generators=(foreign,))


def test_catalog_declares_generated_ideal_value() -> None:
    from jacobian.math.lie_algebras._tools import TOOLS

    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "lie_algebra.ideal.generated.compute"
    )
    assert operation.result_type is LieIdeal
    request = operation.request_type.model_validate_json(
        json.dumps(operation.examples[0].input)
    )
    ideal = operation.run(request)
    assert _rows(ideal) == (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
