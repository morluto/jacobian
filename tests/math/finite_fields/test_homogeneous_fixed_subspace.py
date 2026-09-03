"""Exact homogeneous fixed spaces for prime-field linear actions."""

from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields import (
    HomogeneousFixedSubspace,
    PrimeFieldActionAxis,
    PrimeFieldLinearAction,
    homogeneous_fixed_subspace,
)
from jacobian.math.finite_fields._models import HomogeneousFixedSubspaceRequest
from jacobian.math.finite_fields._tools import TOOLS
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _swap_action() -> PrimeFieldLinearAction:
    return PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(
            name="polynomial_variables", labels=("x", "y")
        ),
        generator_matrices=(
            PrimeFieldMatrix(prime=3, entries=((0, 1), (1, 0)), columns=2),
        ),
    )


def _anwar_q_action_data() -> dict[str, object]:
    return {
        "variable_axis": {"name": "Q", "labels": [f"x{i}" for i in range(5)]},
        "generator_matrices": [
            {
                "prime": 2,
                "entries": [
                    [1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                ],
                "columns": 5,
            },
            {
                "prime": 2,
                "entries": [
                    [1, 1, 1, 0, 1],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                ],
                "columns": 5,
            },
        ],
    }


def _anwar_q_action() -> PrimeFieldLinearAction:
    """The D8 action on Q in arXiv:2607.18585v2, equation (4)."""

    return PrimeFieldLinearAction.model_validate(_anwar_q_action_data())


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
        variable_axis=PrimeFieldActionAxis(
            name="polynomial_variables", labels=("x", "y")
        ),
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
        variable_axis=PrimeFieldActionAxis(name="polynomial_variables", labels=("x",)),
        generator_matrices=(PrimeFieldMatrix(prime=3, entries=((2,),), columns=1),),
    )

    result = homogeneous_fixed_subspace(action, 1)

    assert result.monomial_basis == ((1,),)
    assert result.basis_matrix.entries == ()
    assert result.basis_matrix.columns == 1
    assert result.fixed_dimension == 0


def test_singular_one_variable_generator_is_rejected_by_scalar_admission() -> None:
    action = PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(name="polynomial_variables", labels=("x",)),
        generator_matrices=(PrimeFieldMatrix(prime=3, entries=((0,),), columns=1),),
    )

    with pytest.raises(OperationDomainValidationError, match="invertible"):
        homogeneous_fixed_subspace(action, 1)


def test_multiple_generators_compute_their_simultaneous_fixed_space() -> None:
    action = PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(
            name="polynomial_variables", labels=("x", "y")
        ),
        generator_matrices=(
            PrimeFieldMatrix(prime=5, entries=((0, 1), (1, 0)), columns=2),
            PrimeFieldMatrix(prime=5, entries=((2, 0), (0, 3)), columns=2),
        ),
    )

    result = homogeneous_fixed_subspace(action, 2)

    assert result.basis_matrix.entries == ((0, 1, 0),)


def test_repeated_generators_are_canonicalized_without_changing_the_fixed_space() -> (
    None
):
    generator = PrimeFieldMatrix(prime=5, entries=((0, 1), (1, 0)), columns=2)
    repeated = PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(
            name="polynomial_variables", labels=("x", "y")
        ),
        generator_matrices=tuple(generator for _ in range(1_025)),
    )
    canonical = PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(
            name="polynomial_variables", labels=("x", "y")
        ),
        generator_matrices=(generator,),
    )

    assert repeated.generator_matrices == (generator,)
    assert homogeneous_fixed_subspace(repeated, 2) == homogeneous_fixed_subspace(
        canonical, 2
    )


def test_source_d8_action_reproduces_every_reported_fixed_dimension() -> None:
    # The dimensions are the degree-zero-through-seven ledger cited in #1264
    # for the paper's five-variable U calculation, where k[U] = Sym(Q).
    assert tuple(
        homogeneous_fixed_subspace(_anwar_q_action(), degree).fixed_dimension
        for degree in range(8)
    ) == (1, 2, 4, 7, 15, 23, 37, 53)


def test_source_d8_degree_seven_crosses_the_declared_operation_boundary() -> None:
    request = HomogeneousFixedSubspaceRequest.model_validate(
        {"action": _anwar_q_action_data(), "degree": 7}
    )
    operation = next(
        operation
        for operation in TOOLS
        if operation.operation_id
        == "finite_field.prime_linear_action.homogeneous_fixed_subspace.compute"
    )

    result = operation.run(request)

    assert isinstance(result, HomogeneousFixedSubspace)
    assert result.fixed_dimension == 53


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
        variable_axis=PrimeFieldActionAxis(
            name="polynomial_variables", labels=("x", "y")
        ),
        generator_matrices=(
            PrimeFieldMatrix(prime=3, entries=((1, 0), (0, 0)), columns=2),
        ),
    )

    with pytest.raises(OperationDomainValidationError, match="invertible"):
        homogeneous_fixed_subspace(action, 2)


def test_oversized_homogeneous_basis_is_rejected_by_derived_axis_bound() -> None:
    identity = tuple(
        tuple(int(row == column) for column in range(8)) for row in range(8)
    )
    action = PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(
            name="polynomial_variables", labels=tuple(f"x{index}" for index in range(8))
        ),
        generator_matrices=(PrimeFieldMatrix(prime=3, entries=identity, columns=8),),
    )

    with pytest.raises(OperationDomainValidationError, match="equations exceed"):
        homogeneous_fixed_subspace(action, 6)


def test_stacked_equation_axis_is_rejected_before_polynomial_expansion() -> None:
    generators = tuple(
        PrimeFieldMatrix(
            prime=5,
            entries=(
                (1, index % 5, (index // 5) % 5),
                (0, 1, (index // 25) % 5),
                (0, 0, 1),
            ),
            columns=3,
        )
        for index in range(16)
    )
    action = PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(
            name="polynomial_variables", labels=("x", "y", "z")
        ),
        generator_matrices=generators,
    )

    with pytest.raises(OperationDomainValidationError) as error:
        homogeneous_fixed_subspace(action, 14)

    assert error.value.errors()[0]["type"] == (
        "finite_field.fixed_subspace_equation_axis_bound"
    )


@pytest.mark.parametrize("degree", [-1, True])
def test_native_api_rejects_nonnegative_integer_degree_domain_violations(
    degree: object,
) -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        homogeneous_fixed_subspace(_swap_action(), degree)  # type: ignore[arg-type]

    assert error.value.errors()[0]["type"] == (
        "finite_field.fixed_subspace_degree_bound"
    )


def test_one_variable_degree_is_bounded_by_derived_work_not_fixed_cap() -> None:
    action = PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(name="polynomial_variables", labels=("x",)),
        generator_matrices=(PrimeFieldMatrix(prime=3, entries=((1,),), columns=1),),
    )

    result = homogeneous_fixed_subspace(action, 65)

    assert result.monomial_basis == ((65,),)
    assert result.basis_matrix.entries == ((1,),)


def test_one_variable_basis_does_not_materialize_degree_sized_separator_pool() -> None:
    action = PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(name="polynomial_variables", labels=("x",)),
        generator_matrices=(PrimeFieldMatrix(prime=3, entries=((1,),), columns=1),),
    )

    result = homogeneous_fixed_subspace(action, 10_000_000)

    assert result.monomial_basis == ((10_000_000,),)
    assert result.basis_matrix.entries == ((1,),)


def test_nine_variable_degree_one_action_uses_derived_admission() -> None:
    variable_count = 9
    identity = tuple(
        tuple(int(row == column) for column in range(variable_count))
        for row in range(variable_count)
    )
    action = PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(
            name="polynomial_variables",
            labels=tuple(f"x{index}" for index in range(variable_count)),
        ),
        generator_matrices=(
            PrimeFieldMatrix(prime=3, entries=identity, columns=variable_count),
        ),
    )

    result = homogeneous_fixed_subspace(action, 1)

    assert result.fixed_dimension == variable_count


def test_action_axis_uses_the_shared_matrix_axis_bound() -> None:
    variable_count = 257
    identity = tuple(
        tuple(int(row == column) for column in range(variable_count))
        for row in range(variable_count)
    )

    action = PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(
            name="polynomial_variables",
            labels=tuple(f"x{index}" for index in range(variable_count)),
        ),
        generator_matrices=(
            PrimeFieldMatrix(prime=3, entries=identity, columns=variable_count),
        ),
    )

    assert len(action.variable_axis.labels) == variable_count


def test_seventeen_one_variable_generators_use_derived_admission() -> None:
    action = PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(name="polynomial_variables", labels=("x",)),
        generator_matrices=tuple(
            PrimeFieldMatrix(prime=19, entries=((value,),), columns=1)
            for value in range(1, 18)
        ),
    )

    result = homogeneous_fixed_subspace(action, 1)

    assert result.fixed_dimension == 0


def test_huge_one_variable_degree_uses_logarithmic_scalar_powering() -> None:
    action = PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(name="polynomial_variables", labels=("x",)),
        generator_matrices=(PrimeFieldMatrix(prime=3, entries=((1,),), columns=1),),
    )

    result = homogeneous_fixed_subspace(action, 1_100_000_000)

    assert result.monomial_basis == ((1_100_000_000,),)
    assert result.basis_matrix.entries == ((1,),)


def test_native_action_preserves_the_exact_prime_fallback() -> None:
    prime = 2_147_483_659
    action = PrimeFieldLinearAction.model_validate(
        {
            "variable_axis": {"name": "polynomial_variables", "labels": ["x"]},
            "generator_matrices": [{"prime": prime, "entries": [[1]], "columns": 1}],
        }
    )

    result = homogeneous_fixed_subspace(action, 1)

    assert result.action.prime == prime
    assert result.basis_matrix.entries == ((1,),)


def test_native_action_rejects_a_prime_the_worker_cannot_serialize() -> None:
    prior_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        prime = 2**2_203 - 1
        action = PrimeFieldLinearAction(
            variable_axis=PrimeFieldActionAxis(
                name="polynomial_variables", labels=("x",)
            ),
            generator_matrices=(
                PrimeFieldMatrix(prime=prime, entries=((1,),), columns=1),
            ),
        )

        with pytest.raises(
            OperationDomainValidationError, match="worker JSON integer serialization"
        ):
            homogeneous_fixed_subspace(action, 1)
    finally:
        sys.set_int_max_str_digits(prior_limit)


def test_catalog_action_rejects_non_word_safe_prime_before_nested_work() -> None:
    with pytest.raises(ValidationError, match="word-safe backend bound"):
        HomogeneousFixedSubspaceRequest.model_validate(
            {
                "action": {
                    "variable_axis": {
                        "name": "polynomial_variables",
                        "labels": ["x"],
                    },
                    "generator_matrices": [
                        {"prime": 10**200, "entries": [[1]], "columns": 1}
                    ],
                },
                "degree": 1,
            }
        )


def test_catalog_action_rejects_non_word_safe_canonical_action() -> None:
    action = PrimeFieldLinearAction(
        variable_axis=PrimeFieldActionAxis(name="polynomial_variables", labels=("x",)),
        generator_matrices=(
            PrimeFieldMatrix(
                prime=2_147_483_659,
                entries=((1,),),
                columns=1,
            ),
        ),
    )

    with pytest.raises(ValidationError, match="word-safe backend bound"):
        HomogeneousFixedSubspaceRequest(action=action, degree=1)


def test_raw_action_malformed_matrix_row_is_a_validation_error() -> None:
    with pytest.raises(ValidationError):
        HomogeneousFixedSubspaceRequest.model_validate(
            {
                "action": {
                    "variable_axis": {
                        "name": "polynomial_variables",
                        "labels": ["x"],
                    },
                    "generator_matrices": [{"prime": 3, "entries": [1], "columns": 1}],
                },
                "degree": 1,
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
