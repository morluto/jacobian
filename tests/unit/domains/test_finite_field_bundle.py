import pytest
from pydantic import ValidationError

from jacobian.domains.finite_fields import build_finite_field_bundle
from jacobian.domains.finite_fields.contracts import (
    DirectionRankLedgerRequest,
    FiniteMapTableRequest,
    ProjectiveLineRequest,
)
from jacobian.math.finite_fields import (
    Axis,
    AxisBoundMatrix,
    CollisionCertificate,
    DirectionRankLedger,
    FiberPartition,
    FiniteDimensionalSubspace,
    FiniteFieldPresentation,
    FiniteLinearMap,
    FiniteMapTable,
    FinitePolynomialMap,
    OrbitDistribution,
    PermutationCertificate,
    ProjectiveLine,
    ProjectivePoint,
    RankResult,
    element,
    finite_field,
    finite_polynomial,
    finite_polynomial_map,
    projective_line,
)
from jacobian.operation_execution import execute_operation
from jacobian.operations import NonConclusion


def test_bundle_declares_atomic_port_bound_operations() -> None:
    bundle = build_finite_field_bundle()

    assert bundle.capability_ids == (
        "finite_field.projective_line.enumerate",
        "finite_field.restrict_scalars.compute",
        "finite_field.linear_map.rank.compute",
        "finite_field.direction_rank_ledger.compute",
        "finite_field.orbit_distribution.compute",
        "finite_field.polynomial_map.table.compute",
        "finite_field.polynomial_map.fibers.compute",
        "finite_field.polynomial_map.collision.compute",
        "finite_field.polynomial_map.permutation.compute",
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
    ) = bundle.capabilities
    assert projective.output_ports[0].value_type is ProjectiveLine
    assert tuple(port.value_type for port in restrict_operation.input_ports) == (
        FiniteDimensionalSubspace,
        ProjectivePoint,
    )
    assert restrict_operation.output_ports[0].value_type is FiniteLinearMap
    assert tuple(port.value_type for port in rank_operation.input_ports) == (
        ProjectivePoint,
        FiniteLinearMap,
    )
    assert rank_operation.output_ports[0].value_type is RankResult
    assert tuple(port.value_type for port in ledger.input_ports) == (
        FiniteDimensionalSubspace,
        ProjectiveLine,
    )
    assert ledger.output_ports[0].value_type is DirectionRankLedger
    assert orbit.input_ports[0].value_type is DirectionRankLedger
    assert orbit.output_ports[0].value_type is OrbitDistribution
    assert table.input_ports[0].value_type is FinitePolynomialMap
    assert table.output_ports[0].value_type is FiniteMapTable
    assert fibers.input_ports[0].value_type is FiniteMapTable
    assert fibers.output_ports[0].value_type is FiberPartition
    assert collision.output_ports[0].value_type is CollisionCertificate
    assert permutation.output_ports[0].value_type is PermutationCertificate
    for consumer in (fibers, collision, permutation):
        assert consumer.spec.preflight is not None
        assert consumer.provider_binding.runtime is not None
        assert "finite-map-replay" in consumer.provider_binding.runtime.features


def test_projective_enumeration_refuses_large_output_before_allocation() -> None:
    operation = build_finite_field_bundle().capabilities[0]
    request = ProjectiveLineRequest(
        presentation=FiniteFieldPresentation(
            characteristic=2,
            modulus_coefficients=(1, 1, 1),
        ),
        axis=Axis(name="large", labels=tuple(f"x{index}" for index in range(7))),
    )

    terminal = execute_operation(operation.spec, request)

    assert isinstance(terminal, NonConclusion)
    assert terminal.diagnostic.code == "RESOURCE_LIMIT_EXCEEDED"


def test_finite_map_table_refuses_excessive_polynomial_work() -> None:
    operation = build_finite_field_bundle().capabilities[5]
    presentation = finite_field(2, (1, 1, 0, 1, 1, 0, 0, 0, 1))
    one = element(presentation, (1,) + (0,) * 7)
    request = FiniteMapTableRequest(
        polynomial_map=finite_polynomial_map(
            finite_polynomial(presentation, (one,) * 512)
        )
    )

    terminal = execute_operation(operation.spec, request)

    assert isinstance(terminal, NonConclusion)
    assert terminal.diagnostic.code == "RESOURCE_LIMIT_EXCEEDED"


def test_direction_rank_ledger_refuses_excessive_aggregate_work() -> None:
    operation = build_finite_field_bundle().capabilities[3]
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

    terminal = execute_operation(operation.spec, request)

    assert isinstance(terminal, NonConclusion)
    assert terminal.diagnostic.code == "RESOURCE_LIMIT_EXCEEDED"


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
