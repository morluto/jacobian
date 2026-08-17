import pytest
from pydantic import ValidationError

from jacobian.catalog.models import MathTool
from jacobian.math.finite_fields import (
    Axis,
    AxisBoundMatrix,
    CollisionResult,
    DirectionRankLedger,
    FiberPartition,
    FiniteDimensionalSubspace,
    FiniteFieldPresentation,
    FiniteLinearMap,
    FiniteMapTable,
    OrbitDistribution,
    PermutationResult,
    ProjectiveLine,
    RankResult,
    element,
    finite_field,
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
    (
        projective,
        restrict_operation,
        rank_operation,
        ledger,
        orbit,
        table,
        fibers,
        collision,
        permutation,
    ) = bundle
    for operation in bundle:
        assert isinstance(operation, MathTool)
        assert not hasattr(operation, "provider_binding")
    assert projective.request_type is ProjectiveLineRequest
    assert projective.result_type is ProjectiveLine
    assert restrict_operation.result_type is FiniteLinearMap
    assert rank_operation.result_type is RankResult
    assert ledger.result_type is DirectionRankLedger
    assert orbit.result_type is OrbitDistribution
    assert table.result_type is FiniteMapTable
    assert fibers.result_type is FiberPartition
    assert collision.result_type is CollisionResult
    assert permutation.result_type is PermutationResult


def test_projective_enumeration_refuses_large_output_before_allocation() -> None:
    with pytest.raises(ValidationError, match="two-coordinate axis"):
        ProjectiveLineRequest(
            presentation=FiniteFieldPresentation(
                characteristic=2,
                modulus_coefficients=(1, 1, 1),
            ),
            axis=Axis(name="large", labels=tuple(f"x{index}" for index in range(7))),
        )


def test_finite_map_table_refuses_excessive_polynomial_work() -> None:
    operation = TOOLS[5]
    presentation = finite_field(2, (1, 1, 0, 1, 1, 0, 0, 0, 1))
    one = element(presentation, (1,) + (0,) * 7)
    request = FiniteMapTableRequest(
        polynomial_map=finite_polynomial_map(
            finite_polynomial(presentation, (one,) * 512)
        )
    )

    with pytest.raises(ValueError, match="finite map work"):
        operation.run(request)


def test_direction_rank_ledger_refuses_excessive_aggregate_work() -> None:
    operation = TOOLS[3]
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

    with pytest.raises(ValueError, match="direction-rank work"):
        operation.run(request)


def test_oversized_presentation_rejects_during_request_parsing() -> None:
    with pytest.raises(ValidationError, match="field-order bound"):
        ProjectiveLineRequest(
            presentation=FiniteFieldPresentation(
                characteristic=99991,
                modulus_coefficients=(1, 0, 1),
            ),
            axis=Axis(name="rows", labels=("r1", "r2")),
        )


def test_oversized_axis_rejects_during_request_parsing() -> None:
    with pytest.raises(ValidationError, match="label bound"):
        ProjectiveLineRequest(
            presentation=FiniteFieldPresentation(
                characteristic=2,
                modulus_coefficients=(1, 1, 1),
            ),
            axis=Axis(name="large", labels=tuple(f"x{i}" for i in range(257))),
        )
