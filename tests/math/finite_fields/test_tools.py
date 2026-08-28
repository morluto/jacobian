import pytest
from pydantic import ValidationError

from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.finite_fields import (
    Axis,
    AxisBoundMatrix,
    FiniteDimensionalSubspace,
    FiniteFieldPresentation,
    FiniteLinearMap,
    FiniteMapTable,
    ProjectiveLine,
    RankResult,
    direction_rank_ledger,
    element,
    finite_field,
    finite_map_table,
    finite_polynomial,
    finite_polynomial_map,
    projective_line,
)
from jacobian.math.finite_fields._models import (
    DirectionRankLedgerRequest,
    FiniteMapTableRequest,
    ProjectiveLineRequest,
)
from jacobian.math.finite_fields._tools import TOOLS


def test_bundle_declares_atomic_inline_typed_operations() -> None:
    bundle = TOOLS

    assert tuple(operation.operation_id for operation in bundle) == (
        "finite_field.projective_line.enumerate",
        "finite_field.restrict_scalars.compute",
        "finite_field.linear_map.rank.compute",
        "finite_field.direction_rank_ledger.compute",
        "finite_field.orbit_distribution.compute",
        "finite_field.polynomial_map.table.compute",
        "finite_field.polynomial_map.fibers.compute",
        "finite_field.polynomial_map.collision.analyze",
        "finite_field.polynomial_map.permutation.analyze",
    )
    projective, restrict_operation, rank_operation, _, _, table, _, _, _ = bundle
    for operation in bundle:
        assert isinstance(operation, MathTool)
        assert not hasattr(operation, "provider_binding")
    assert projective.request_type is ProjectiveLineRequest
    assert projective.result_type is ProjectiveLine
    assert restrict_operation.result_type is FiniteLinearMap
    assert rank_operation.result_type is RankResult
    assert table.result_type is FiniteMapTable


def test_projective_enumeration_refuses_large_output_before_allocation() -> None:
    request = ProjectiveLineRequest(
        presentation=FiniteFieldPresentation(
            characteristic=2,
            modulus_coefficients=(1, 1, 1),
        ),
        axis=Axis(name="large", labels=tuple(f"x{index}" for index in range(7))),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        projective_line(request.presentation, request.axis)
    assert (
        error.value.errors()[0]["type"]
        == "finite_field.projective_line_two_coordinate_axis"
    )


def test_finite_map_table_refuses_excessive_polynomial_work() -> None:
    presentation = finite_field(2, (1, 1, 0, 1, 1, 0, 0, 0, 1))
    one = element(presentation, (1,) + (0,) * 7)
    request = FiniteMapTableRequest(
        polynomial_map=finite_polynomial_map(
            finite_polynomial(presentation, (one,) * 512)
        )
    )
    with pytest.raises(OperationDomainValidationError) as error:
        finite_map_table(request.polynomial_map)
    assert error.value.errors() == (
        {
            "loc": ("polynomial_map",),
            "type": "finite_field.finite_map_exceeds_operation_work_budget",
            "msg": "finite map exceeds the operation work budget",
        },
    )


def test_direction_rank_ledger_refuses_excessive_aggregate_work() -> None:
    presentation = finite_field(2, (1, 1, 1))
    row_axis = Axis(name="rows", labels=("r0", "r1"))
    column_axis = Axis(
        name="columns",
        labels=tuple(f"c{index}" for index in range(64)),
    )
    basis_axis = Axis(
        name="basis",
        labels=tuple(f"B{index}" for index in range(64)),
    )
    zero = element(presentation, (0, 0))
    one = element(presentation, (1, 0))
    basis = tuple(
        AxisBoundMatrix(
            presentation=presentation,
            row_axis=row_axis,
            column_axis=column_axis,
            entries=(
                tuple(one if column == index else zero for column in range(64)),
                (zero,) * 64,
            ),
        )
        for index in range(64)
    )
    request = DirectionRankLedgerRequest(
        subspace=FiniteDimensionalSubspace(
            presentation=presentation,
            basis_axis=basis_axis,
            basis=basis,
        ),
        directions=projective_line(presentation, row_axis),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        direction_rank_ledger(request.subspace, request.directions)
    assert (
        error.value.errors()[0]["type"]
        == "finite_field.direction_rank_ledger_exceeds_operation_work_budget"
    )


def test_oversized_presentation_rejects_during_request_parsing() -> None:
    with pytest.raises(ValidationError) as error:
        ProjectiveLineRequest(
            presentation=FiniteFieldPresentation(
                characteristic=99991,
                modulus_coefficients=(1, 0, 1),
            ),
            axis=Axis(name="rows", labels=("r1", "r2")),
        )
    assert (
        error.value.errors()[0]["type"]
        == "finite_field.characteristic_exceeds_supported_field_order_bound"
    )


def test_oversized_axis_rejects_during_request_parsing() -> None:
    with pytest.raises(ValidationError) as error:
        ProjectiveLineRequest(
            presentation=FiniteFieldPresentation(
                characteristic=2,
                modulus_coefficients=(1, 1, 1),
            ),
            axis=Axis(name="large", labels=tuple(f"x{i}" for i in range(257))),
        )
    assert (
        error.value.errors()[0]["type"]
        == "finite_field.axis_exceeds_supported_label_bound"
    )
