"""Exact homogeneous fixed spaces for prime-field linear actions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields import (
    Axis,
    HomogeneousFixedSubspace,
    PrimeFieldLinearAction,
    homogeneous_fixed_subspace,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _swap_action() -> PrimeFieldLinearAction:
    return PrimeFieldLinearAction(
        variable_axis=Axis(name="polynomial_variables", labels=("x", "y")),
        generator_matrices=(
            PrimeFieldMatrix(prime=3, entries=((0, 1), (1, 0)), columns=2),
        ),
    )


def _anwar_q_action() -> PrimeFieldLinearAction:
    """The D8 action on Q in arXiv:2607.18585v2, equation (4)."""

    return PrimeFieldLinearAction(
        variable_axis=Axis(name="Q", labels=tuple(f"x{i}" for i in range(5))),
        generator_matrices=(
            PrimeFieldMatrix(
                prime=2,
                entries=(
                    (1, 1, 0, 0, 0),
                    (0, 1, 1, 0, 0),
                    (0, 0, 1, 0, 0),
                    (0, 0, 0, 1, 1),
                    (0, 0, 0, 0, 1),
                ),
                columns=5,
            ),
            PrimeFieldMatrix(
                prime=2,
                entries=(
                    (1, 1, 1, 0, 1),
                    (0, 1, 0, 0, 0),
                    (0, 0, 1, 0, 0),
                    (0, 0, 0, 1, 1),
                    (0, 0, 0, 0, 1),
                ),
                columns=5,
            ),
        ),
    )


def _apply_matrix(
    matrix: tuple[tuple[int, ...], ...], vector: tuple[int, ...], prime: int
) -> tuple[int, ...]:
    return tuple(
        sum(coefficient * value for coefficient, value in zip(row, vector, strict=True))
        % prime
        for row in matrix
    )


def test_quadratic_swap_has_a_canonical_reduced_fixed_basis() -> None:
    result = homogeneous_fixed_subspace(_swap_action(), 2)

    assert result.monomial_basis == ((2, 0), (1, 1), (0, 2))
    # Rows encode x²+y² and xy in canonical RREF coefficient form.
    assert result.basis_matrix.entries == ((1, 0, 1), (0, 1, 0))
    assert result.fixed_dimension == 2


def test_nonsymmetric_generator_uses_matrix_columns_as_variable_images() -> None:
    action = PrimeFieldLinearAction(
        variable_axis=Axis(name="polynomial_variables", labels=("x", "y")),
        # x -> x and y -> x+y, so the degree-one fixed space is <x>.
        generator_matrices=(
            PrimeFieldMatrix(prime=3, entries=((1, 1), (0, 1)), columns=2),
        ),
    )

    result = homogeneous_fixed_subspace(action, 1)

    assert result.monomial_basis == ((1, 0), (0, 1))
    assert result.basis_matrix.entries == ((1, 0),)


def test_degree_zero_is_the_constant_fixed_space() -> None:
    result = homogeneous_fixed_subspace(_swap_action(), 0)

    assert result.monomial_basis == ((0, 0),)
    assert result.basis_matrix.entries == ((1,),)
    assert result.fixed_dimension == 1


def test_zero_fixed_space_retains_the_ambient_monomial_axis() -> None:
    action = PrimeFieldLinearAction(
        variable_axis=Axis(name="polynomial_variables", labels=("x",)),
        generator_matrices=(PrimeFieldMatrix(prime=3, entries=((2,),), columns=1),),
    )

    result = homogeneous_fixed_subspace(action, 1)

    assert result.monomial_basis == ((1,),)
    assert result.basis_matrix.entries == ()
    assert result.basis_matrix.columns == 1
    assert result.fixed_dimension == 0


def test_multiple_generators_compute_their_simultaneous_fixed_space() -> None:
    action = PrimeFieldLinearAction(
        variable_axis=Axis(name="polynomial_variables", labels=("x", "y")),
        generator_matrices=(
            PrimeFieldMatrix(prime=5, entries=((0, 1), (1, 0)), columns=2),
            PrimeFieldMatrix(prime=5, entries=((2, 0), (0, 3)), columns=2),
        ),
    )

    result = homogeneous_fixed_subspace(action, 2)

    assert result.basis_matrix.entries == ((0, 1, 0),)


def test_source_d8_action_reproduces_every_reported_fixed_dimension() -> None:
    # The dimensions are the degree-zero-through-seven ledger cited in #1264
    # for the paper's five-variable U calculation, where k[U] = Sym(Q).
    assert tuple(
        homogeneous_fixed_subspace(_anwar_q_action(), degree).fixed_dimension
        for degree in range(8)
    ) == (1, 2, 4, 7, 15, 23, 37, 53)


def test_every_returned_row_is_fixed_by_every_induced_generator() -> None:
    from jacobian.math.finite_fields.operations import _induced_action_matrix

    result = homogeneous_fixed_subspace(_swap_action(), 3)
    for generator in result.action.generator_matrices:
        induced = _induced_action_matrix(
            result.action, result.monomial_basis, generator
        )
        for basis_row in result.basis_matrix.entries:
            assert _apply_matrix(induced, basis_row, result.action.prime) == basis_row


def test_result_round_trips_and_its_action_composes_unchanged() -> None:
    produced = homogeneous_fixed_subspace(_swap_action(), 2)
    decoded = HomogeneousFixedSubspace.model_validate_json(produced.model_dump_json())
    consumed = homogeneous_fixed_subspace(decoded.action, decoded.degree)

    assert decoded == produced
    assert consumed == produced


def test_singular_generator_is_rejected_by_operation_admission() -> None:
    action = PrimeFieldLinearAction(
        variable_axis=Axis(name="polynomial_variables", labels=("x", "y")),
        generator_matrices=(
            PrimeFieldMatrix(prime=3, entries=((1, 0), (0, 0)), columns=2),
        ),
    )

    with pytest.raises(OperationDomainValidationError, match="invertible"):
        homogeneous_fixed_subspace(action, 2)


def test_oversized_homogeneous_basis_is_rejected_before_expansion() -> None:
    identity = tuple(
        tuple(int(row == column) for column in range(8)) for row in range(8)
    )
    action = PrimeFieldLinearAction(
        variable_axis=Axis(
            name="polynomial_variables", labels=tuple(f"x{index}" for index in range(8))
        ),
        generator_matrices=(PrimeFieldMatrix(prime=3, entries=identity, columns=8),),
    )

    with pytest.raises(OperationDomainValidationError, match="monomial basis"):
        homogeneous_fixed_subspace(action, 5)


@pytest.mark.parametrize("degree", [-1, 65, True])
def test_native_api_rejects_degree_outside_the_typed_request_domain(
    degree: object,
) -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        homogeneous_fixed_subspace(_swap_action(), degree)  # type: ignore[arg-type]

    assert error.value.errors()[0]["type"] == (
        "finite_field.fixed_subspace_degree_bound"
    )


def test_raw_action_rejects_oversized_prime_and_matrix_before_nested_work() -> None:
    with pytest.raises(ValidationError, match="word-safe backend bound"):
        PrimeFieldLinearAction.model_validate(
            {
                "variable_axis": {
                    "name": "polynomial_variables",
                    "labels": ["x"],
                },
                "generator_matrices": [
                    {"prime": 10**200, "entries": [[1]], "columns": 1}
                ],
            }
        )


def test_raw_result_binds_basis_prime_before_nested_matrix_parsing() -> None:
    produced = homogeneous_fixed_subspace(_swap_action(), 2).model_dump(mode="json")
    produced["basis_matrix"]["prime"] = 5

    with pytest.raises(ValidationError) as error:
        HomogeneousFixedSubspace.model_validate(produced)

    assert error.value.errors()[0]["type"] == (
        "finite_field.fixed_subspace_basis_parent"
    )
