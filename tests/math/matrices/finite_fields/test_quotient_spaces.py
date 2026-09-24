"""Exact quotient-space values over bounded prime fields."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.finite_fields._tools import (
    compute_quotient_space,
    project_quotient_vector,
)
from jacobian.math.matrices.finite_fields.quotient_spaces import (
    PrimeFieldQuotientRequest,
    PrimeFieldQuotientSpace,
    PrimeFieldSubspace,
    PrimeFieldVectorProjectionRequest,
)


def _quotient(prime: int, n: int, generators: tuple[tuple[int, ...], ...]):
    return compute_quotient_space(
        PrimeFieldQuotientRequest(
            subspace=PrimeFieldSubspace(
                prime=prime, ambient_dimension=n, generators=generators
            )
        )
    )


def _mat_vec(matrix: tuple[tuple[int, ...], ...], vector: tuple[int, ...], p: int):
    return tuple(
        sum(a * b for a, b in zip(row, vector, strict=True)) % p for row in matrix
    )


def _span(generators: tuple[tuple[int, ...], ...], p: int):
    coefficients = range(p)
    vectors = {tuple(0 for _ in (generators[0] if generators else ()))}
    for generator in generators:
        vectors = {
            tuple(
                (entry + scalar * value) % p
                for entry, value in zip(vector, generator, strict=True)
            )
            for vector in vectors
            for scalar in coefficients
        }
    return vectors


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_projection_has_denominator_kernel_and_coordinates_in_returned_basis(
    prime: int,
):
    quotient = _quotient(
        prime,
        3,
        ((1, 1, 0), (prime - 1, prime - 1, 0), (0, 1, 1)),
    )
    assert len(quotient.quotient_basis) == 1
    for generator in quotient.source.generators:
        assert _mat_vec(quotient.projection.entries, generator, prime) == (0,)
    for representative in quotient.quotient_basis:
        assert _mat_vec(quotient.projection.entries, representative, prime) == (1,)

    vector = (2 % prime, 1 % prime, 2 % prime)
    projected = project_quotient_vector(
        PrimeFieldVectorProjectionRequest(quotient=quotient, vector=vector)
    )
    assert projected.quotient == quotient
    assert projected.coordinates == _mat_vec(quotient.projection.entries, vector, prime)


def test_quotient_and_projected_vector_roundtrip_unchanged_through_json():
    quotient = _quotient(3, 2, ((1, 1),))
    decoded_quotient = PrimeFieldQuotientSpace.model_validate_json(
        quotient.model_dump_json()
    )
    assert decoded_quotient == quotient
    projected = project_quotient_vector(
        PrimeFieldVectorProjectionRequest(quotient=decoded_quotient, vector=(2, 0))
    )
    assert projected.coordinates == (2,)
    assert projected.quotient == quotient


def test_zero_subspace_is_identity_quotient_and_full_subspace_is_zero_quotient():
    identity = _quotient(5, 2, ())
    assert identity.quotient_basis == ((1, 0), (0, 1))
    assert identity.projection.entries == ((1, 0), (0, 1))
    vector = (4, 3)
    assert (
        project_quotient_vector(
            PrimeFieldVectorProjectionRequest(quotient=identity, vector=vector)
        ).coordinates
        == vector
    )

    zero = _quotient(5, 2, ((1, 0), (0, 1)))
    assert zero.quotient_basis == ()
    assert zero.projection.entries == ()
    assert zero.projection.columns == 2
    projected_zero = project_quotient_vector(
        PrimeFieldVectorProjectionRequest(quotient=zero, vector=(4, 3))
    )
    assert projected_zero.coordinates == ()
    assert projected_zero.quotient == zero


def test_zero_ambient_axis_retains_empty_projection_shape():
    zero = _quotient(2, 0, ())
    assert zero.quotient_basis == ()
    assert zero.projection.entries == ()
    assert zero.projection.columns == 0
    assert (
        project_quotient_vector(
            PrimeFieldVectorProjectionRequest(quotient=zero, vector=())
        ).coordinates
        == ()
    )


def test_quotient_projection_is_invariant_under_adding_denominator_vectors():
    quotient = _quotient(7, 3, ((1, 2, 0),))
    vector = (3, 4, 5)
    representative_in_same_class = tuple(
        (entry + scale * basis_entry) % 7
        for entry, basis_entry in zip(vector, (1, 2, 0), strict=True)
        for scale in [3]
    )
    first = project_quotient_vector(
        PrimeFieldVectorProjectionRequest(quotient=quotient, vector=vector)
    )
    second = project_quotient_vector(
        PrimeFieldVectorProjectionRequest(
            quotient=quotient, vector=representative_in_same_class
        )
    )
    assert first.coordinates == second.coordinates


def test_projection_classes_match_exhaustive_cosets_in_gf2_cubed():
    generators = ((1, 1, 0), (0, 1, 1), (1, 0, 1))
    quotient = _quotient(2, 3, generators)
    denominator = _span(generators, 2)
    vectors = tuple(
        (first, second, third)
        for first in range(2)
        for second in range(2)
        for third in range(2)
    )
    projected = {
        vector: project_quotient_vector(
            PrimeFieldVectorProjectionRequest(quotient=quotient, vector=vector)
        ).coordinates
        for vector in vectors
    }
    for left in vectors:
        for right in vectors:
            difference = tuple((a - b) % 2 for a, b in zip(left, right, strict=True))
            assert (projected[left] == projected[right]) == (difference in denominator)


def test_projection_rejects_wrong_parent_axis_and_noncanonical_values():
    quotient = _quotient(3, 2, ((1, 0),))
    with pytest.raises(ValidationError):
        PrimeFieldVectorProjectionRequest(quotient=quotient, vector=(0,))
    with pytest.raises(ValidationError):
        PrimeFieldVectorProjectionRequest(quotient=quotient, vector=(3, 0))
    with pytest.raises(ValidationError):
        PrimeFieldQuotientSpace.model_validate(
            {
                "source": {
                    "prime": 3,
                    "ambient_dimension": 2,
                    "generators": [[1, 0]],
                },
                "quotient_basis": [[0, 1]],
                "projection": {
                    "prime": 5,
                    "entries": [[0, 1]],
                    "columns": 2,
                },
            }
        )


def test_projection_rechecks_caller_authored_quotient_relation():
    quotient = _quotient(3, 2, ((1, 1),))
    forged_payload = quotient.model_dump(mode="json")
    forged_payload["projection"]["entries"] = [[1, 0]]
    forged = PrimeFieldQuotientSpace.model_validate(forged_payload)
    with pytest.raises(OperationDomainValidationError):
        project_quotient_vector(
            PrimeFieldVectorProjectionRequest(quotient=forged, vector=(2, 1))
        )


def test_quotient_operation_rejects_a_composite_characteristic():
    with pytest.raises(OperationDomainValidationError):
        _quotient(4, 1, ())


def test_quotient_admission_counts_parented_output_coordinates():
    with pytest.raises(ValidationError):
        PrimeFieldQuotientRequest(
            subspace=PrimeFieldSubspace(prime=2, ambient_dimension=724, generators=())
        )


def test_quotient_space_tools_are_published_with_parented_value_contracts():
    tools = {tool.operation_id: tool for tool in BUILTIN_TOOLS}
    assert (
        tools["prime_field.vector_space.quotient.compute"].result_type
        is PrimeFieldQuotientSpace
    )
    assert (
        tools["prime_field.vector_space.quotient.project"].result_type.__name__
        == "PrimeFieldQuotientVector"
    )
