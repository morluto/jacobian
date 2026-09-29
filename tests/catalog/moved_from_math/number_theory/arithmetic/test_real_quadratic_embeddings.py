"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/number_theory/arithmetic/test_real_quadratic_embeddings.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

from jacobian._exact import CanonicalRational
from jacobian.math.number_theory.algebraic_numbers.quadratic import (
    RealQuadraticEmbeddingProfile,
    RealQuadraticValue,
    real_quadratic_embeddings,
)


def _r(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(numerator, denominator)


def _element(
    rational_part: int,
    radical_coefficient: int,
    radicand: int,
) -> RealQuadraticValue:
    return RealQuadraticValue(
        rational_part=_r(rational_part),
        radical_coefficient=_r(radical_coefficient),
        radicand=radicand,
    )


def _profile() -> RealQuadraticEmbeddingProfile:
    return real_quadratic_embeddings(_element(3, 2, 2))


def test_embedding_profile_is_served_by_the_public_catalog() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    ids = {tool.operation_id for tool in BUILTIN_TOOLS}

    assert "arithmetic.real_quadratic.embeddings.compute" in ids
    assert "arithmetic.real_quadratic.order.compute" in ids
