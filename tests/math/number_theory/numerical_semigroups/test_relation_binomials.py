"""Native relation encodings do not assert generation of a toric kernel."""

from __future__ import annotations

from typing import Any

import pytest
from sympy import Poly, symbols

from jacobian.math.number_theory.numerical_semigroups import (
    MinimalPresentationRelation,
    RelationBinomial,
    RelationBinomialsResult,
    relation_binomials,
)


def test_empty_and_doubled_families_are_only_relation_encodings() -> None:
    empty = relation_binomials((3, 5), ())
    assert empty == RelationBinomialsResult(minimal_generators=(3, 5), binomials=())
    doubled = relation_binomials(
        (3, 5), (MinimalPresentationRelation(first=(10, 0), second=(0, 6)),)
    )
    assert doubled == RelationBinomialsResult(
        minimal_generators=(3, 5),
        binomials=(RelationBinomial(left_exponents=(10, 0), right_exponents=(0, 6)),),
    )
    assert (
        RelationBinomialsResult.model_validate_json(
            doubled.model_dump_json(), strict=True
        )
        == doubled
    )

    # Independent counterexample: X^5-Y^3 is in the kernel X->t^3,Y->t^5,
    # but is nonzero and not in the principal ideal (X^10-Y^6). A nonzero
    # polynomial multiple of the doubled relation has total degree >=10.
    x, y, t = symbols("x y t")
    kernel_generator = Poly(x**5 - y**3, x, y, domain="QQ")
    binomial = doubled.binomials[0]
    doubled_polynomial = Poly.from_dict(
        {
            binomial.left_exponents: binomial.left_coefficient,
            binomial.right_exponents: binomial.right_coefficient,
        },
        x,
        y,
        domain="QQ",
    )
    assert kernel_generator.as_expr().subs({x: t**3, y: t**5}) == 0
    assert not kernel_generator.is_zero  # Not in the ideal of the empty family.
    assert doubled_polynomial == kernel_generator * Poly(x**5 + y**3, x, y)
    assert kernel_generator.total_degree() < doubled_polynomial.total_degree()
    assert kernel_generator.rem(doubled_polynomial) == kernel_generator


@pytest.mark.parametrize("generators", ((5, 3), (3, 5, 8), (3, 5, 5)))
def test_conversion_never_reinterprets_a_noncanonical_generator_axis(
    generators: tuple[int, ...],
) -> None:
    relation = MinimalPresentationRelation(first=(5, 0), second=(0, 3))
    with pytest.raises(
        ValueError, match=r"strictly increasing|minimal generating system"
    ):
        relation_binomials(generators, (relation,))


@pytest.mark.parametrize(
    "relation,reason",
    (
        (
            MinimalPresentationRelation(first=(1, 0), second=(0, 1)),
            "same semigroup degree",
        ),
        (
            MinimalPresentationRelation(first=(1,), second=(0,)),
            "coordinates must match",
        ),
        (
            MinimalPresentationRelation.model_construct(first=(-5, 0), second=(0, -3)),
            "nonnegative integers",
        ),
        (
            MinimalPresentationRelation.model_construct(first=(True, 0), second=(0, 1)),
            "nonnegative integers",
        ),
        (
            MinimalPresentationRelation.model_construct(first=(5, 0), second=(5, 0)),
            "must be distinct",
        ),
        (
            MinimalPresentationRelation.model_construct(first=[5, 0], second=(0, 3)),
            "coordinates must match",
        ),
        (
            MinimalPresentationRelation.model_construct(second=(0, 3)),
            "coordinates must match",
        ),
    ),
)
def test_relation_shape_and_source_degree_are_checked(
    relation: MinimalPresentationRelation, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        relation_binomials((3, 5), (relation,))


@pytest.mark.parametrize("generators", ((), (0, 1), (3, 6), (2, 501), (True, 5)))
def test_native_conversion_rejects_invalid_or_unbounded_sources(
    generators: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        relation_binomials(generators, ())


@pytest.mark.parametrize(
    "relations", (None, [], ({"first": (5, 0), "second": (0, 3)},))
)
def test_native_conversion_requires_relation_values(relations: Any) -> None:
    with pytest.raises(ValueError, match="relation"):
        relation_binomials((3, 5), relations)


def test_relation_orientation_order_and_duplicates_are_retained() -> None:
    first = MinimalPresentationRelation(first=(5, 0), second=(0, 3))
    second = MinimalPresentationRelation(first=(0, 3), second=(5, 0))
    result = relation_binomials((3, 5), (first, second, first))
    assert result.minimal_generators == (3, 5)
    assert [(b.left_exponents, b.right_exponents) for b in result.binomials] == [
        ((5, 0), (0, 3)),
        ((0, 3), (5, 0)),
        ((5, 0), (0, 3)),
    ]
