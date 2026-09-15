"""Semigroup constructions slice (#3719)."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_semigroups._models import FiniteSemigroup
from jacobian.math.finite_semigroups.operations import (
    adjoin_identity,
    adjoin_zero,
    karoubi_projection,
    opposite_semigroup,
    product_semigroup,
    rees_quotient,
)


def _z3() -> FiniteSemigroup:
    return FiniteSemigroup(
        elements=("0", "1", "2"),
        multiplication=(("0", "1", "2"), ("1", "2", "0"), ("2", "0", "1")),
    )


def test_opposite_is_involution() -> None:
    source = _z3()
    opposite = opposite_semigroup(source).opposite
    assert (
        opposite_semigroup(
            FiniteSemigroup(
                elements=opposite.elements, multiplication=opposite.multiplication
            )
        ).opposite
        == source
    )


def test_product_projections_replay() -> None:
    source = _z3()
    result = product_semigroup(source, source)
    assert len(result.product.elements) == 9
    left = dict(result.left_projection)
    right = dict(result.right_projection)
    for label in result.product.elements:
        assert "×" in label  # noqa: RUF001 - product label separator
    # Projection of a product is componentwise.
    first = result.product.elements[4]
    assert left[first] in source.elements and right[first] in source.elements


def test_adjunctions_extend_and_embed() -> None:
    source = _z3()
    adjoined = adjoin_identity(source)
    assert len(adjoined.result.elements) == 4
    assert tuple(source for source, _ in adjoined.embedding) == source.elements
    zeroed = adjoin_zero(source)
    assert len(zeroed.result.elements) == 4
    zero = zeroed.result.elements[-1]
    idx = {label: i for i, label in enumerate(zeroed.result.elements)}
    for element in zeroed.result.elements:
        assert zeroed.result.multiplication[idx[zero]][idx[element]] == zero
        assert zeroed.result.multiplication[idx[element]][idx[zero]] == zero


def test_rees_collapse_and_projection() -> None:
    source = _z3()
    result = rees_quotient(source, ("0", "1", "2"))
    assert len(result.quotient.elements) == 1
    assert all(target == result.quotient.elements[0] for _, target in result.projection)
    with pytest.raises(OperationDomainValidationError, match="two-sided"):
        rees_quotient(source, ("1",))


def test_karoubi_envelope_of_group() -> None:
    result = karoubi_projection(_z3())
    assert result.objects == ("0",)
    assert len(result.category.morphisms) == 3
    assert result.category.objects == ("0",)
