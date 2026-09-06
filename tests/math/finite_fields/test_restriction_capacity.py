"""Restriction output axes use the prime-field matrix capacity."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields import (
    Axis,
    AxisBoundMatrix,
    FiniteDimensionalSubspace,
    ProjectivePoint,
    element,
    finite_field,
    linear_map_rank,
    restrict_scalars,
)


def _source(
    columns: int, modulus: tuple[int, ...]
) -> tuple[FiniteDimensionalSubspace, ProjectivePoint]:
    field = finite_field(2, modulus)
    one = element(field, (1,) + (0,) * (field.degree - 1))
    rows = Axis(name="rows", labels=("r",))
    matrix = AxisBoundMatrix(
        presentation=field,
        row_axis=rows,
        column_axis=Axis(name="columns", labels=tuple(f"c{i}" for i in range(columns))),
        entries=((one,) * columns,),
    )
    subspace = FiniteDimensionalSubspace(
        presentation=field,
        row_axis=matrix.row_axis,
        column_axis=matrix.column_axis,
        basis_axis=Axis(name="basis", labels=("B",)),
        basis=(matrix,),
    )
    direction = ProjectivePoint(presentation=field, axis=rows, coordinates=(one,))
    return subspace, direction


@pytest.mark.parametrize(
    ("columns", "modulus"),
    [(129, (1, 1, 1)), (128, (1, 1, 0, 1, 1, 0, 0, 0, 1))],
)
def test_restricted_axes_reach_the_matrix_capacity(
    columns: int, modulus: tuple[int, ...]
) -> None:
    source, direction = _source(columns, modulus)
    result = restrict_scalars(source, direction)
    degree = source.presentation.degree
    assert len(result.target_axis.labels) == columns * degree
    assert result.matrix.entries == tuple(
        (int(i % degree == 0),) for i in range(columns * degree)
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result
    ranked = linear_map_rank(source, direction)
    assert ranked.rank == 1
    assert ranked.linear_map == result
    assert type(ranked).model_validate_json(ranked.model_dump_json()) == ranked


def test_source_admission_rejects_expansion_beyond_matrix_capacity() -> None:
    with pytest.raises(OperationDomainValidationError, match="restriction output"):
        restrict_scalars(*_source(129, (1, 1, 0, 1, 1, 0, 0, 0, 1)))


def test_larger_axis_does_not_expand_extension_matrix_source_capacity() -> None:
    with pytest.raises(ValueError, match="matrix axes exceed"):
        _source(257, (1, 1, 1))
