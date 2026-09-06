"""Zero matrix subspaces retain their ambient axes through restriction."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields import (
    Axis,
    AxisBoundMatrix,
    FiniteDimensionalSubspace,
    FiniteFieldPresentation,
    ProjectivePoint,
    direction_rank_ledger,
    finite_field,
    linear_map_rank,
    orbit_distribution,
    projective_line,
    restrict_scalars,
)

pytestmark = pytest.mark.requires_backend("flint")


@pytest.mark.parametrize("column_count", [0, 2])
def test_zero_subspace_composes_through_restriction_and_rank(column_count: int) -> None:
    field = finite_field(2, (1, 1, 1))
    rows = Axis(name="rows", labels=("x", "y"))
    columns = Axis(name="columns", labels=tuple(f"c{i}" for i in range(column_count)))
    subspace = FiniteDimensionalSubspace(
        presentation=field,
        row_axis=rows,
        column_axis=columns,
        basis_axis=Axis(name="basis", labels=()),
        basis=(),
    )
    restored = FiniteDimensionalSubspace.model_validate_json(subspace.model_dump_json())
    assert restored == subspace
    assert restored.row_axis == rows
    assert restored.column_axis == columns
    directions = projective_line(field, rows)
    for direction in directions.points:
        restricted = restrict_scalars(restored, direction)
        assert restricted.source_axis == subspace.basis_axis
        assert restricted.target_axis.labels == tuple(
            f"{label}:{basis}"
            for label in columns.labels
            for basis in field.ordered_basis
        )
        assert restricted.matrix.entries == ((),) * (column_count * field.degree)
        assert restricted.matrix.columns == 0
        ranked = linear_map_rank(restored, direction)
        assert ranked.rank == 0
        assert type(ranked).model_validate_json(ranked.model_dump_json()) == ranked
    ledger = direction_rank_ledger(restored, directions)
    ledger = type(ledger).model_validate_json(ledger.model_dump_json())
    assert all(entry.rank == 0 for entry in ledger.entries)
    distribution = orbit_distribution(ledger)
    assert distribution.counts == ((1, 5 * (1 + 4**column_count)),)
    assert (
        type(distribution).model_validate_json(distribution.model_dump_json())
        == distribution
    )


def test_empty_subspace_digest_retains_ambient_axes() -> None:
    field = finite_field(2, (1, 1, 1))
    subspaces = tuple(
        FiniteDimensionalSubspace(
            presentation=field,
            row_axis=Axis(name="rows", labels=rows),
            column_axis=Axis(name="columns", labels=columns),
            basis_axis=Axis(name="basis", labels=()),
            basis=(),
        )
        for rows, columns in [((), ()), ((), ("c",)), (("r",), ())]
    )
    assert len({value.digest for value in subspaces}) == 3
    for value in subspaces:
        assert type(value).model_validate_json(value.model_dump_json()) == value


def test_subspace_checks_declared_ambient_axes_not_only_agreement_between_matrices() -> (
    None
):
    field = finite_field(2, (1, 1, 1))
    rows = Axis(name="rows", labels=("r",))
    columns = Axis(name="columns", labels=())
    matrix = AxisBoundMatrix(
        presentation=field, row_axis=rows, column_axis=columns, entries=((),)
    )
    with pytest.raises(ValueError, match="share their parent and axes"):
        FiniteDimensionalSubspace(
            presentation=field,
            row_axis=Axis(name="rows", labels=("other",)),
            column_axis=columns,
            basis_axis=Axis(name="basis", labels=("B",)),
            basis=(matrix,),
        )


def test_empty_basis_still_requires_an_admitted_field() -> None:
    field = FiniteFieldPresentation(characteristic=4, modulus_coefficients=(0, 1))
    rows = Axis(name="rows", labels=("r",))
    subspace = FiniteDimensionalSubspace(
        presentation=field,
        row_axis=rows,
        column_axis=Axis(name="columns", labels=()),
        basis_axis=Axis(name="basis", labels=()),
        basis=(),
    )
    direction = ProjectivePoint.model_validate(
        {
            "presentation": field,
            "axis": rows,
            "coordinates": [{"presentation": field, "coordinates": (1,)}],
        }
    )
    with pytest.raises(
        OperationDomainValidationError, match="characteristic must be prime"
    ):
        restrict_scalars(subspace, direction)
