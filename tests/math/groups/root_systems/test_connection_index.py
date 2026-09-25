"""Exact weight/root quotient invariants for finite Cartan data."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.groups.root_systems._models import CartanMatrix
from jacobian.math.groups.root_systems.connection_index._models import (
    RootSystemConnectionIndexResult,
)
from jacobian.math.groups.root_systems.connection_index.operations import (
    root_system_connection_index,
)
from jacobian.math.matrices.values import IntegerMatrix

_OPERATION_ID = "root_system.connection_index.compute"

Matrix = tuple[tuple[int, ...], ...]


def _cartan(rows: Matrix) -> CartanMatrix:
    rank = len(rows)
    return CartanMatrix(
        matrix=IntegerMatrix(
            row_count=rank,
            column_count=rank,
            entries=rows,
        ),
        simple_root_axis=tuple(range(rank)),
    )


@pytest.mark.parametrize(
    ("rows", "expected_index", "expected_factors"),
    (
        (((2,),), 2, (2,)),  # A1
        (((2, -1), (-1, 2)), 3, (3,)),  # A2
        (((2, -1), (-2, 2)), 2, (2,)),  # B2
        (((2, -2), (-1, 2)), 2, (2,)),  # C2
        (
            (
                (2, -1, 0, 0),
                (-1, 2, -1, -1),
                (0, -1, 2, 0),
                (0, -1, 0, 2),
            ),
            4,
            (2, 2),
        ),  # D4
        (
            (
                (2, -1, 0, 0, 0),
                (-1, 2, -1, 0, 0),
                (0, -1, 2, -1, -1),
                (0, 0, -1, 2, 0),
                (0, 0, -1, 0, 2),
            ),
            4,
            (4,),
        ),  # D5
        (
            tuple(
                tuple(
                    2 if row == column else -1 if abs(row - column) == 1 else 0
                    for column in range(8)
                )
                for row in range(8)
            ),
            9,
            (9,),
        ),  # A8 exercises the maximum admitted rank
        (((2, -3), (-1, 2)), 1, ()),  # G2
    ),
    ids=("a1", "a2", "b2", "c2", "d4", "d5", "a8-rank-bound", "g2"),
)
def test_connection_index_matches_known_root_lattice_quotients(
    rows: Matrix,
    expected_index: int,
    expected_factors: tuple[int, ...],
) -> None:
    result = root_system_connection_index(_cartan(rows))

    assert result.connection_index == expected_index
    assert result.invariant_factors == expected_factors
    assert result.root_to_weight.entries == rows
    assert result.quotient == "WEIGHT_LATTICE_MOD_ROOT_LATTICE"
    assert (
        RootSystemConnectionIndexResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_reducible_root_quotient_canonicalizes_component_product() -> None:
    # P/Q for A1 x A2 is Z/2 x Z/3, canonically Z/6.
    result = root_system_connection_index(_cartan(((2, 0, 0), (0, 2, -1), (0, -1, 2))))

    assert result.connection_index == 6
    assert result.invariant_factors == (6,)


def test_result_model_rejects_inconsistent_index_and_basis_map() -> None:
    result = root_system_connection_index(_cartan(((2, -1), (-1, 2))))
    payload = json.loads(result.model_dump_json())
    payload["connection_index"] = "2"
    with pytest.raises(ValidationError, match="product of quotient invariant factors"):
        RootSystemConnectionIndexResult.model_validate_json(json.dumps(payload))

    payload = json.loads(result.model_dump_json())
    payload["root_to_weight"]["entries"][0][0] = "1"
    with pytest.raises(ValidationError, match="equal the retained Cartan matrix"):
        RootSystemConnectionIndexResult.model_validate_json(json.dumps(payload))


def test_non_finite_cartan_data_are_rejected() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        root_system_connection_index(_cartan(((2, -1, -1), (-1, 2, -1), (-1, -1, 2))))
    assert error.value.errors()[0]["type"] == "root_system.finite_type"


def test_raw_rank_bound_is_checked_before_smith_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = tuple(
        tuple(2 if row == column else 0 for column in range(9)) for row in range(9)
    )
    forged = CartanMatrix.model_construct(
        matrix=IntegerMatrix.model_construct(
            row_count=9,
            column_count=9,
            entries=rows,
        ),
        simple_root_axis=tuple(range(9)),
    )

    def forbidden_smith(_matrix: IntegerMatrix) -> object:
        pytest.fail("Smith expansion ran before the Cartan rank bound")

    monkeypatch.setattr(
        "jacobian.math.groups.root_systems.connection_index.operations.smith_normal_form_result",
        forbidden_smith,
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        root_system_connection_index(forged)
    assert (
        error.value.errors()[0]["type"] == "root_system.connection_index.rank_exceeded"
    )


def test_catalog_operation_runs_example() -> None:
    catalog = Catalog.open()
    operation = catalog.operation(_OPERATION_ID)
    assert operation is not None
    result = invoke_operation(_OPERATION_ID, operation.examples[0].input, catalog)
    assert result.output["invariant_factors"] == ["3"]
    assert result.output["connection_index"] == "3"
