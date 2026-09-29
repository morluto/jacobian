"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/affine_semigroups/test_minimal_generators.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from fractions import Fraction
from typing import cast

from jacobian._exact import CanonicalRational
from jacobian.math.affine_semigroups import (
    AffineConfiguration,
    PositiveAffineSemigroup,
)
from jacobian.math.affine_semigroups.atoms import (
    AffineMinimalGenerators,
    AffineMinimalGeneratorsRequest,
)


def _semigroup(vectors: tuple[tuple[int, ...], ...]) -> PositiveAffineSemigroup:
    rows = len(vectors[0])
    configuration = AffineConfiguration(
        row_labels=tuple(f"r{i}" for i in range(rows)),
        generator_labels=tuple(f"g{i}" for i in range(len(vectors))),
        entries=tuple(tuple(vector[row] for vector in vectors) for row in range(rows)),
    )
    return PositiveAffineSemigroup(
        configuration=configuration,
        grading=tuple(
            CanonicalRational.from_fraction(Fraction(1)) for _ in range(rows)
        ),
    )


def _brute_factorizations(
    vectors: tuple[tuple[int, ...], ...], target: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    maxima = tuple(
        max(
            (target[r] // vector[r] for r in range(len(target)) if vector[r] > 0),
            default=0,
        )
        for vector in vectors
    )
    return tuple(
        counts
        for counts in itertools.product(*(range(bound + 1) for bound in maxima))
        if tuple(
            sum(counts[j] * vectors[j][r] for j in range(len(vectors)))
            for r in range(len(target))
        )
        == target
    )


def test_catalog_example_executes_with_the_published_operation_contract() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "affine_semigroup.minimal_generators.compute"
    )
    example = tool.examples[0]
    request_type = cast(type[AffineMinimalGeneratorsRequest], tool.request_type)
    request = request_type.model_validate_json(json.dumps(example.input))

    run = cast(
        Callable[[AffineMinimalGeneratorsRequest], AffineMinimalGenerators],
        tool.run,
    )
    result = run(request)

    assert result.atoms.configuration.columns_vectors == ((1, 0),)
    assert result.source_factorizations == ((1,), (1,), (2,))
