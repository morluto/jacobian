"""Catalog membership check for the finite-field quotient-space operations."""

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.matrices import finite_fields
from jacobian.math.matrices.finite_fields._tools import compute_quotient_space
from jacobian.math.matrices.finite_fields.operations import (
    project_quotient_vector,
)
from jacobian.math.matrices.finite_fields.quotient_spaces import (
    PrimeFieldQuotientRequest,
    PrimeFieldQuotientSpace,
    PrimeFieldSubspace,
)


def _quotient(
    prime: int, n: int, generators: tuple[tuple[int, ...], ...]
) -> PrimeFieldQuotientSpace:
    return compute_quotient_space(
        PrimeFieldQuotientRequest(
            subspace=PrimeFieldSubspace(
                prime=prime, ambient_dimension=n, generators=generators
            )
        )
    )


def _mat_vec(
    matrix: tuple[tuple[int, ...], ...], vector: tuple[int, ...], p: int
) -> tuple[int, ...]:
    return tuple(
        sum(a * b for a, b in zip(row, vector, strict=True)) % p for row in matrix
    )


def _span(generators: tuple[tuple[int, ...], ...], p: int) -> set[tuple[int, ...]]:
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


def test_quotient_construction_is_the_only_published_quotient_operation() -> None:
    tools = {tool.operation_id: tool for tool in BUILTIN_TOOLS}
    assert (
        tools["prime_field.vector_space.quotient.compute"].result_type
        is PrimeFieldQuotientSpace
    )
    # Projection is a cheap deterministic map of an existing public result, so
    # it stays a native package export instead of a discovery entry.
    assert "prime_field.vector_space.quotient.project" not in tools
    assert finite_fields.project_quotient_vector is project_quotient_vector
