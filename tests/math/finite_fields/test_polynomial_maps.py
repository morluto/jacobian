from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.finite_fields import (
    CollisionResult,
    FiberPartition,
    FiniteMapTable,
    FinitePolynomialMap,
    PermutationResult,
    analyze_collisions,
    analyze_permutation,
    element,
    fiber_partition,
    finite_field,
    finite_map_table,
    finite_polynomial,
    finite_polynomial_map,
)
from jacobian.math.finite_fields._tools import TOOLS

pytestmark = pytest.mark.requires_backend("flint")


def _map(*exponents: int) -> FinitePolynomialMap:
    presentation = finite_field(2, (1, 1, 1))
    zero = element(presentation, (0, 0))
    one = element(presentation, (1, 0))
    coefficients = tuple(one if power in exponents else zero for power in range(4))
    return finite_polynomial_map(finite_polynomial(presentation, coefficients))


def test_complete_table_and_fibers_reuse_exact_slice_a_field_identity() -> None:
    polynomial_map = _map(3)

    table = finite_map_table(polynomial_map)
    partition = fiber_partition(table)
    collision = analyze_collisions(table)

    assert len(table.entries) == polynomial_map.domain.order == 4
    assert all(
        source.presentation is polynomial_map.domain for source, _ in table.entries
    )
    assert all(
        target.presentation is polynomial_map.codomain for _, target in table.entries
    )
    assert sorted(len(sources) for _, sources in partition.fibers) == [1, 3]
    assert collision.left != collision.right
    assert (
        next(target for source, target in table.entries if source == collision.left)
        == collision.image
    )
    assert type(table).model_validate(table.model_dump(mode="json")) == table


def test_frobenius_map_is_a_permutation() -> None:
    table = finite_map_table(_map(2))

    result = analyze_permutation(table)

    assert result.status == "PERMUTATION"
    assert len(result.inverse_entries) == 4
    assert {target.digest for _, target in table.entries} == {
        source.digest for source, _ in result.inverse_entries
    }


def test_slice_b_values_reject_wrong_parent_incomplete_table_and_forged_fiber() -> None:
    polynomial_map = _map(3)
    table = finite_map_table(polynomial_map)
    other = finite_field(2, (1, 1, 1), generator="z")
    wrong_polynomial = finite_polynomial(
        other,
        (element(other, (0, 0)), element(other, (1, 0))),
    )

    with pytest.raises(ValueError, match="one exact field presentation"):
        FinitePolynomialMap(
            domain=polynomial_map.domain,
            codomain=polynomial_map.codomain,
            polynomial=wrong_polynomial,
        )
    with pytest.raises(ValueError, match="complete domain"):
        FiniteMapTable(map=polynomial_map, entries=table.entries[:-1])
    with pytest.raises(ValueError, match="canonical domain order"):
        FiniteMapTable(map=polynomial_map, entries=tuple(reversed(table.entries)))
    with pytest.raises(ValueError, match="partition"):
        FiberPartition(
            table=table,
            fibers=((table.entries[0][1], (table.entries[0][0],)),),
        )


def test_certificates_reject_values_not_bound_to_the_exact_table() -> None:
    collision_table = finite_map_table(_map(3))
    permutation_table = finite_map_table(_map(2))
    collision = analyze_collisions(collision_table)
    permutation = analyze_permutation(permutation_table)

    with pytest.raises(ValueError, match="exact bound table"):
        CollisionResult(
            table=collision_table,
            status="COLLISION",
            left=collision.left,
            right=collision.right,
            image=collision_table.entries[0][1],
        )
    with pytest.raises(ValueError, match="exact permutation"):
        PermutationResult(
            table=permutation_table,
            status="PERMUTATION",
            inverse_entries=tuple(reversed(permutation.inverse_entries)),
        )


def test_table_consumers_reject_unevaluated_targets() -> None:
    identity_table = finite_map_table(_map(1))
    zero = identity_table.entries[0][1]
    with pytest.raises(ValidationError, match="bound polynomial"):
        FiniteMapTable(
            map=identity_table.map,
            entries=tuple((source, zero) for source, _ in identity_table.entries),
        )


def test_slice_b_reuses_one_table_for_fiber_and_certificate_handoff() -> None:
    polynomial_map = _map(3)
    _, _, _, _, _, table_operation, fiber_operation, collision_operation, _ = TOOLS

    table = table_operation.run(
        table_operation.request_type.model_validate({"polynomial_map": polynomial_map})
    )

    partition = fiber_operation.run(
        fiber_operation.request_type.model_validate({"table": table})
    )
    collision = collision_operation.run(
        collision_operation.request_type.model_validate({"table": table})
    )

    assert partition.table is table
    assert collision.table is table
