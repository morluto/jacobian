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


# ---------------------------------------------------------------------------
# Structural laws: involution and projection functoriality (Workstream C)
# ---------------------------------------------------------------------------


def _left_zero_band() -> FiniteSemigroup:
    """Two-element left-zero band: x * y = x (associative, noncommutative)."""
    return FiniteSemigroup(
        elements=("a", "b"),
        multiplication=(("a", "a"), ("b", "b")),
    )


def _right_zero_band() -> FiniteSemigroup:
    """Two-element right-zero band: x * y = y (associative, noncommutative)."""
    return FiniteSemigroup(
        elements=("a", "b"),
        multiplication=(("a", "b"), ("a", "b")),
    )


def _multiply(semigroup: FiniteSemigroup, left: str, right: str) -> str:
    index = {label: position for position, label in enumerate(semigroup.elements)}
    return semigroup.multiplication[index[left]][index[right]]


class TestInvolutionLaws:
    def test_opposite_transposes_and_involutes_noncommutative_band(self) -> None:
        source = _left_zero_band()
        opposite = opposite_semigroup(source).opposite
        assert opposite.elements == source.elements
        for i in range(2):
            for j in range(2):
                assert opposite.multiplication[i][j] == source.multiplication[j][i]
        # The band is noncommutative, so the opposite is genuinely different.
        assert opposite.multiplication != source.multiplication
        redoubled = opposite_semigroup(
            FiniteSemigroup(
                elements=opposite.elements, multiplication=opposite.multiplication
            )
        ).opposite
        assert redoubled == source

    def test_opposite_commutes_with_direct_product(self) -> None:
        left = _left_zero_band()
        right = _right_zero_band()
        opposite_of_product = opposite_semigroup(
            product_semigroup(left, right).product
        ).opposite
        product_of_opposites = product_semigroup(
            opposite_semigroup(left).opposite, opposite_semigroup(right).opposite
        ).product
        assert opposite_of_product == product_of_opposites


class TestProjectionFunctoriality:
    def test_product_projections_preserve_multiplication(self) -> None:
        left = _left_zero_band()
        right = _right_zero_band()
        result = product_semigroup(left, right)
        left_map = dict(result.left_projection)
        right_map = dict(result.right_projection)
        labels = result.product.elements
        for first in labels:
            for second in labels:
                product_cell = _multiply(result.product, first, second)
                assert left_map[product_cell] == _multiply(
                    left, left_map[first], left_map[second]
                )
                assert right_map[product_cell] == _multiply(
                    right, right_map[first], right_map[second]
                )

    def test_adjunction_embeddings_preserve_multiplication(self) -> None:
        source = _left_zero_band()
        for adjoined in (adjoin_identity(source), adjoin_zero(source)):
            embedding = dict(adjoined.embedding)
            for first in source.elements:
                for second in source.elements:
                    assert (
                        _multiply(adjoined.result, embedding[first], embedding[second])
                        == embedding[_multiply(source, first, second)]
                    )

    def test_rees_projection_preserves_multiplication(self) -> None:
        source = adjoin_zero(_left_zero_band()).result
        result = rees_quotient(source, ("0",))
        projection = dict(result.projection)
        # The ideal collapses to a single class.
        assert projection["0"] == result.quotient.elements[-1]
        for first in source.elements:
            for second in source.elements:
                assert projection[_multiply(source, first, second)] == _multiply(
                    result.quotient, projection[first], projection[second]
                )


class TestForgedStructuralClaims:
    def test_forged_opposite_cell_breaks_transposition(self) -> None:
        source = _left_zero_band()
        genuine = opposite_semigroup(source).opposite
        rows = [list(row) for row in genuine.multiplication]
        rows[0][1] = "a" if rows[0][1] == "b" else "b"
        forged = FiniteSemigroup(
            elements=genuine.elements,
            multiplication=tuple(tuple(row) for row in rows),
        )
        assert forged != genuine
        assert any(
            forged.multiplication[i][j] != source.multiplication[j][i]
            for i in range(2)
            for j in range(2)
        )

    def test_swapped_product_projections_break_functoriality(self) -> None:
        left = _left_zero_band()
        right = _right_zero_band()
        result = product_semigroup(left, right)
        left_map = dict(result.left_projection)
        right_map = dict(result.right_projection)
        # The projections genuinely differ on mixed pairs.
        assert any(
            left_map[label] != right_map[label] for label in result.product.elements
        )
        # A swapped (weakened) projection claim fails preservation somewhere.
        assert any(
            right_map[_multiply(result.product, first, second)]
            != _multiply(right, left_map[first], left_map[second])
            for first in result.product.elements
            for second in result.product.elements
        )

    def test_forged_identity_embedding_breaks_preservation(self) -> None:
        source = _left_zero_band()
        adjoined = adjoin_identity(source)
        fresh = adjoined.result.elements[-1]
        forged_embedding = dict(adjoined.embedding)
        forged_embedding[source.elements[0]] = fresh
        assert any(
            _multiply(
                adjoined.result, forged_embedding[first], forged_embedding[second]
            )
            != forged_embedding[_multiply(source, first, second)]
            for first in source.elements
            for second in source.elements
        )

    def test_forged_zero_embedding_breaks_preservation(self) -> None:
        source = _left_zero_band()
        adjoined = adjoin_zero(source)
        fresh = adjoined.result.elements[-1]
        forged_embedding = dict(adjoined.embedding)
        forged_embedding[source.elements[0]] = fresh
        assert any(
            _multiply(
                adjoined.result, forged_embedding[first], forged_embedding[second]
            )
            != forged_embedding[_multiply(source, first, second)]
            for first in source.elements
            for second in source.elements
        )

    def test_forged_rees_projection_breaks_preservation(self) -> None:
        source = adjoin_zero(_left_zero_band()).result
        result = rees_quotient(source, ("0",))
        forged_projection = dict(result.projection)
        forged_projection["a"] = forged_projection["0"]
        assert any(
            forged_projection[_multiply(source, first, second)]
            != _multiply(
                result.quotient, forged_projection[first], forged_projection[second]
            )
            for first in source.elements
            for second in source.elements
        )
