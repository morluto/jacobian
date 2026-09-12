import pytest
from pydantic import ValidationError

from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.finite_fields import (
    Axis,
    AxisBoundMatrix,
    FiniteDimensionalSubspace,
    FiniteFieldElement,
    FiniteFieldPresentation,
    FiniteLinearMap,
    FiniteMapTable,
    HomogeneousFixedSubspace,
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
    FinitePolynomialEvaluationRequest,
    PaleyTournamentRequest,
    ProjectiveLineRequest,
)
from jacobian.math.finite_fields._tools import TOOLS


def test_bundle_declares_atomic_inline_typed_operations() -> None:
    bundle = TOOLS

    assert tuple(operation.operation_id for operation in bundle) == (
        "finite_field.projective_line.enumerate",
        "finite_field.matrix.rank.compute",
        "finite_field.restrict_scalars.compute",
        "finite_field.linear_map.rank.compute",
        "finite_field.direction_rank_ledger.compute",
        "finite_field.orbit_distribution.compute",
        "finite_field.polynomial.evaluate.compute",
        "finite_field.polynomial_map.table.compute",
        "finite_field.polynomial_map.fibers.compute",
        "finite_field.polynomial_map.collision.analyze",
        "finite_field.polynomial_map.permutation.analyze",
        "finite_field.paley_tournament.construct",
        "finite_field.prime_linear_action.homogeneous_fixed_subspace.compute",
    )
    (
        projective,
        _,
        restrict_operation,
        rank_operation,
        _,
        _,
        point_evaluation,
        table,
        _,
        _,
        _,
        paley,
        fixed,
    ) = bundle
    for operation in bundle:
        assert isinstance(operation, MathTool)
        assert not hasattr(operation, "provider_binding")
    assert projective.request_type is ProjectiveLineRequest
    assert projective.result_type is ProjectiveLine
    assert restrict_operation.result_type is FiniteLinearMap
    assert rank_operation.result_type is RankResult
    assert table.result_type is FiniteMapTable
    assert point_evaluation.request_type is FinitePolynomialEvaluationRequest
    assert point_evaluation.result_type is FiniteFieldElement
    assert paley.request_type is PaleyTournamentRequest
    assert fixed.result_type is HomogeneousFixedSubspace


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
            row_axis=row_axis,
            column_axis=column_axis,
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


def test_axis_beyond_shared_matrix_bound_rejects_during_request_parsing() -> None:
    with pytest.raises(ValidationError) as error:
        ProjectiveLineRequest(
            presentation=FiniteFieldPresentation(
                characteristic=2,
                modulus_coefficients=(1, 1, 1),
            ),
            axis=Axis(name="large", labels=tuple(f"x{i}" for i in range(1025))),
        )
    assert (
        error.value.errors()[0]["type"]
        == "finite_field.axis_exceeds_supported_label_bound"
    )


def test_point_evaluation_has_no_complete_field_enumeration_factor() -> None:
    presentation = finite_field(2, (1, 1, 0, 1, 1, 0, 0, 0, 1))
    one = element(presentation, (1,) + (0,) * 7)
    polynomial = finite_polynomial(presentation, (one,) * 512)
    request = FinitePolynomialEvaluationRequest(
        polynomial=polynomial,
        value=element(presentation, (0,) * 8),
    )
    # 512 coefficients * degree 8 is admitted; the complete table's extra
    # factor of |F| is intentionally not charged by this operation.
    operation = next(
        operation
        for operation in TOOLS
        if operation.operation_id == "finite_field.polynomial.evaluate.compute"
    )
    restored_request = operation.request_type.model_validate_json(
        request.model_dump_json(), strict=True
    )
    result = operation.run(restored_request)
    assert result == one
    assert (
        FiniteFieldElement.model_validate_json(result.model_dump_json(), strict=True)
        == result
    )


def test_point_evaluation_rejects_mismatched_parent() -> None:
    field = finite_field(2, (1, 1, 1))
    other = finite_field(3, (0, 1))
    polynomial = finite_polynomial(
        field, (element(field, (1, 0)), element(field, (1, 0)))
    )
    request = FinitePolynomialEvaluationRequest(
        polynomial=polynomial,
        value=element(other, (0,)),
    )
    operation = next(
        operation
        for operation in TOOLS
        if operation.operation_id == "finite_field.polynomial.evaluate.compute"
    )
    with pytest.raises(OperationDomainValidationError) as error:
        operation.run(request)
    assert error.value.errors()[0]["type"] == (
        "finite_field.finite_polynomial_evaluation_parent_mismatch"
    )


def test_point_evaluation_preserves_extension_field_coordinates() -> None:
    field = finite_field(23, (1, 18, 1), generator="v")
    coefficients = tuple(
        element(field, (coefficient, 0)) for coefficient in (0, 18, 16, 16, 18)
    )
    request = FinitePolynomialEvaluationRequest(
        polynomial=finite_polynomial(field, coefficients, variable="X"),
        value=element(field, (0, 1)),
    )
    operation = next(
        operation
        for operation in TOOLS
        if operation.operation_id == "finite_field.polynomial.evaluate.compute"
    )
    assert operation.run(request).coordinates == (1, 22)


@pytest.mark.parametrize("constant", [0, 1])
def test_point_evaluation_handles_zero_and_constant_polynomials_and_matches_table(
    constant: int,
) -> None:
    field = finite_field(2, (1, 1, 1))
    zero = element(field, (0, 0))
    one = element(field, (1, 0))
    polynomial = finite_polynomial(field, (zero if constant == 0 else one,))
    value = element(field, (0, 1))
    operation = next(
        operation
        for operation in TOOLS
        if operation.operation_id == "finite_field.polynomial.evaluate.compute"
    )

    result = operation.run(
        FinitePolynomialEvaluationRequest(polynomial=polynomial, value=value)
    )
    table = finite_map_table(finite_polynomial_map(polynomial))
    table_value = next(target for source, target in table.entries if source == value)

    assert result == (zero if constant == 0 else one)
    assert result == table_value
