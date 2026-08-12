from __future__ import annotations

import pytest

from jacobian.math.finite_fields import (
    Axis,
    AxisBoundMatrix,
    FiniteDimensionalSubspace,
    FiniteFieldElement,
    FiniteFieldPresentation,
    FiniteLinearMap,
    ProjectivePoint,
    RankResult,
)
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix


def _presentation(*, generator: str = "a") -> FiniteFieldPresentation:
    return FiniteFieldPresentation(
        characteristic=2,
        modulus_coefficients=(1, 1, 0, 1),
        generator=generator,
    )


def _element(
    presentation: FiniteFieldPresentation,
    coordinates: tuple[int, int, int],
) -> FiniteFieldElement:
    return FiniteFieldElement(presentation=presentation, coordinates=coordinates)


def test_presentation_identity_binds_modulus_generator_basis_and_encoding() -> None:
    presentation = _presentation()

    assert presentation.degree == 3
    assert presentation.order == 8
    assert presentation.ordered_basis == ("1", "a", "a^2")
    assert presentation.digest == _presentation().digest
    assert presentation.digest != _presentation(generator="z").digest
    assert (
        FiniteFieldPresentation.model_validate(presentation.model_dump(mode="json"))
        == presentation
    )


def test_presentation_rejects_reducible_or_noncanonical_moduli() -> None:
    with pytest.raises(ValueError, match="irreducible"):
        FiniteFieldPresentation(characteristic=2, modulus_coefficients=(0, 0, 1))
    with pytest.raises(ValueError, match="canonical"):
        FiniteFieldPresentation(characteristic=2, modulus_coefficients=(1, 3, 1))


def test_values_reject_same_shape_substitutions_with_wrong_parent_or_axis() -> None:
    presentation = _presentation()
    other_presentation = _presentation(generator="z")
    row_axis = Axis(name="rows", labels=("r1", "r2"))
    column_axis = Axis(name="columns", labels=("c1", "c2"))
    wrong_axis = Axis(name="other rows", labels=("r1", "r2"))
    zero = _element(presentation, (0, 0, 0))
    one = _element(presentation, (1, 0, 0))

    with pytest.raises(ValueError, match="presentation"):
        AxisBoundMatrix(
            presentation=presentation,
            row_axis=row_axis,
            column_axis=column_axis,
            entries=((one, zero), (zero, _element(other_presentation, (1, 0, 0)))),
        )
    with pytest.raises(ValueError, match="normalized"):
        ProjectivePoint(
            presentation=presentation,
            axis=row_axis,
            coordinates=(_element(presentation, (0, 1, 0)), zero),
        )
    matrix = FiniteLinearMap(
        source_axis=column_axis,
        target_axis=row_axis,
        matrix=PrimeFieldMatrix(2, ((1, 0), (0, 1)), 2),
    )
    with pytest.raises(ValueError, match="target axis"):
        FiniteLinearMap(
            source_axis=column_axis,
            target_axis=wrong_axis,
            matrix=PrimeFieldMatrix(2, ((1, 0),), 2),
        )
    point = ProjectivePoint(
        presentation=presentation,
        axis=row_axis,
        coordinates=(one, zero),
    )
    assert RankResult(direction=point, linear_map=matrix, rank=2).rank == 2


def test_subspace_rejects_dependent_basis_matrices() -> None:
    presentation = _presentation()
    rows = Axis(name="rows", labels=("r1", "r2"))
    columns = Axis(name="columns", labels=("c1", "c2"))
    basis_axis = Axis(name="basis", labels=("B1", "B2"))
    zero = _element(presentation, (0, 0, 0))
    one = _element(presentation, (1, 0, 0))
    matrix = AxisBoundMatrix(
        presentation=presentation,
        row_axis=rows,
        column_axis=columns,
        entries=((one, zero), (zero, zero)),
    )

    with pytest.raises(ValueError, match="independent"):
        FiniteDimensionalSubspace(
            presentation=presentation,
            basis_axis=basis_axis,
            basis=(matrix, matrix),
        )


def test_rank_result_rejects_a_map_over_the_wrong_prime_field() -> None:
    presentation = _presentation()
    rows = Axis(name="rows", labels=("r1", "r2"))
    basis_axis = Axis(name="basis", labels=("B1",))
    zero = _element(presentation, (0, 0, 0))
    one = _element(presentation, (1, 0, 0))
    direction = ProjectivePoint(
        presentation=presentation,
        axis=rows,
        coordinates=(one, zero),
    )
    wrong_prime_map = FiniteLinearMap(
        source_axis=basis_axis,
        target_axis=Axis(name="target", labels=("t1", "t2")),
        matrix=PrimeFieldMatrix(3, ((1,), (0,)), 1),
    )

    with pytest.raises(ValueError, match="prime field"):
        RankResult(direction=direction, linear_map=wrong_prime_map, rank=1)


def test_presentation_rejects_oversized_characteristic_before_primality() -> None:
    with pytest.raises(ValueError, match="field-order bound"):
        FiniteFieldPresentation(
            characteristic=99991,
            modulus_coefficients=(1, 0, 1),
        )


def test_presentation_rejects_oversized_field_order_before_irreducibility() -> None:
    with pytest.raises(ValueError, match="field order"):
        FiniteFieldPresentation(
            characteristic=2,
            modulus_coefficients=(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        )


def test_presentation_rejects_oversized_modulus_length() -> None:
    with pytest.raises(ValueError, match="length"):
        FiniteFieldPresentation(
            characteristic=2,
            modulus_coefficients=(1,) + (0,) * 17 + (1,),
        )


def test_axis_rejects_oversized_label_set() -> None:
    with pytest.raises(ValueError, match="label bound"):
        Axis(name="large", labels=tuple(f"x{i}" for i in range(257)))


def test_subspace_rejects_oversized_rank_matrix_before_allocation() -> None:
    presentation = _presentation()
    row_axis = Axis(name="rows", labels=tuple(f"r{i}" for i in range(16)))
    column_axis = Axis(name="columns", labels=tuple(f"c{i}" for i in range(16)))
    zero = _element(presentation, (0, 0, 0))
    matrix = AxisBoundMatrix(
        presentation=presentation,
        row_axis=row_axis,
        column_axis=column_axis,
        entries=((zero,) * 16,) * 16,
    )

    with pytest.raises(ValueError, match="rank matrix"):
        FiniteDimensionalSubspace(
            presentation=presentation,
            basis_axis=Axis(
                name="basis",
                labels=tuple(f"b{i}" for i in range(86)),
            ),
            basis=(matrix,) * 86,
        )
