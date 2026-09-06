from __future__ import annotations

from collections.abc import Callable

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields import (
    FiniteMapTable,
    FinitePolynomialMap,
    analyze_collisions,
    analyze_permutation,
    element,
    evaluate_finite_polynomial,
    fiber_partition,
    finite_field,
    finite_map_table,
    finite_polynomial,
    finite_polynomial_map,
    verify_collisions,
    verify_fiber_partition,
    verify_permutation,
)

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
    assert tuple(target for _source, target in table.entries) == tuple(
        evaluate_finite_polynomial(polynomial_map.polynomial, source)
        for source, _target in table.entries
    )


def test_frobenius_map_is_a_permutation() -> None:
    table = finite_map_table(_map(2))

    result = analyze_permutation(table)

    assert result.status == "PERMUTATION"
    assert len(result.inverse_entries) == 4
    assert {target.digest for _, target in table.entries} == {
        source.digest for source, _ in result.inverse_entries
    }
    assert {source.digest: target.digest for source, target in table.entries} == {
        target.digest: source.digest for target, source in result.inverse_entries
    }
    assert type(result).model_validate(result.model_dump(mode="json")) == result


def test_polynomial_normalization_checks_parent_of_discarded_zero() -> None:
    field = finite_field(2, (1, 1, 1))
    other = finite_field(3, (0, 1))
    one = element(field, (1, 0))
    foreign_zero = element(other, (0,))
    with pytest.raises(ValueError, match="coefficients must share their parent"):
        finite_polynomial(field, (one, foreign_zero))
    zero = element(field, (0, 0))
    polynomial = finite_polynomial(field, (one, zero, zero))
    assert polynomial.coefficients == (one,)
    assert (
        type(polynomial).model_validate_json(polynomial.model_dump_json()) == polynomial
    )


def test_slice_b_values_reject_wrong_parent_and_incomplete_table() -> None:
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


def test_fibers_and_collisions_preserve_the_table_defining_invariants() -> None:
    table = finite_map_table(_map(3))
    partition = fiber_partition(table)
    collision = analyze_collisions(table)

    assert all(
        sources == tuple(source for source, target in table.entries if target == image)
        for image, sources in partition.fibers
    )
    assert collision.left is not None
    assert collision.right is not None
    assert collision.image is not None
    assert all(
        target == collision.image
        for source, target in table.entries
        if source in (collision.left, collision.right)
    )
    assert type(table).model_validate(table.model_dump(mode="json")) == table
    assert (
        type(partition).model_validate(partition.model_dump(mode="json")) == partition
    )
    assert (
        type(collision).model_validate(collision.model_dump(mode="json")) == collision
    )


def test_slice_b_reuses_one_table_for_fiber_and_certificate_handoff() -> None:
    polynomial_map = _map(3)
    table = finite_map_table(polynomial_map)
    partition = fiber_partition(table)
    collision = analyze_collisions(table)

    assert partition.table is table
    assert collision.table is table


@pytest.mark.parametrize(
    "consumer", [fiber_partition, analyze_collisions, analyze_permutation]
)
@pytest.mark.parametrize("mutation", ["target", "polynomial"])
def test_consumers_authenticate_the_supplied_polynomial_table(
    consumer: Callable[[FiniteMapTable], object],
    mutation: str,
) -> None:
    table = finite_map_table(_map(2))
    payload = table.model_dump(mode="json")
    if mutation == "target":
        payload["entries"][1][1] = payload["entries"][0][1]
    else:
        payload["map"]["polynomial"]["coefficients"] = [payload["entries"][0][0]]
    candidate = FiniteMapTable.model_validate(payload)

    with pytest.raises(OperationDomainValidationError) as error:
        consumer(candidate)
    assert error.value.errors()[0]["type"] == (
        "finite_field.finite_map_table_targets_match_bound_polynomial"
    )


@pytest.mark.parametrize(
    "consumer", [fiber_partition, analyze_collisions, analyze_permutation]
)
def test_table_authentication_rejects_unadmitted_evaluation_work(
    consumer: Callable[[FiniteMapTable], object],
) -> None:
    field = finite_field(2, (1, 1, 0, 1, 1, 0, 0, 0, 1))
    one = element(field, (1,) + (0,) * 7)
    table = finite_map_table(finite_polynomial_map(finite_polynomial(field, (one,))))
    candidate_map = finite_polynomial_map(finite_polynomial(field, (one,) * 512))
    candidate = FiniteMapTable(map=candidate_map, entries=table.entries)

    with pytest.raises(OperationDomainValidationError) as error:
        consumer(candidate)
    assert error.value.errors()[0]["type"] == (
        "finite_field.finite_map_exceeds_operation_work_budget"
    )


@pytest.mark.parametrize("exponents", [(), (2,), (3,)])
def test_serialized_producer_tables_enter_every_consumer(
    exponents: tuple[int, ...],
) -> None:
    table = finite_map_table(_map(*exponents))
    candidate = FiniteMapTable.model_validate_json(table.model_dump_json())
    for consumer in (fiber_partition, analyze_collisions, analyze_permutation):
        result = consumer(candidate)
        assert result == consumer(table)
        assert result.table is candidate


def test_serialized_fiber_partition_is_explicitly_verifiable() -> None:
    partition = fiber_partition(finite_map_table(_map(3)))
    decoded = type(partition).model_validate_json(partition.model_dump_json())
    assert verify_fiber_partition(decoded)

    forged = partition.model_dump(mode="json")
    forged["fibers"][1][1] = [forged["fibers"][1][1][0]]
    forged_decoded = type(partition).model_validate(forged)
    assert not verify_fiber_partition(forged_decoded)


def test_serialized_collision_is_explicitly_verifiable() -> None:
    collision = analyze_collisions(finite_map_table(_map(3)))
    decoded = type(collision).model_validate_json(collision.model_dump_json())
    assert verify_collisions(decoded)

    forged = collision.model_dump(mode="json")
    forged["image"] = forged["table"]["entries"][0][1]
    forged_decoded = type(collision).model_validate(forged)
    assert not verify_collisions(forged_decoded)


def test_serialized_permutation_is_explicitly_verifiable() -> None:
    permutation = analyze_permutation(finite_map_table(_map(2)))
    decoded = type(permutation).model_validate_json(permutation.model_dump_json())
    assert verify_permutation(decoded)

    forged = permutation.model_dump(mode="json")
    forged["inverse_entries"][0][1] = forged["inverse_entries"][1][1]
    forged_decoded = type(permutation).model_validate(forged)
    assert not verify_permutation(forged_decoded)
