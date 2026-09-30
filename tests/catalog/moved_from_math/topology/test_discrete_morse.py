"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/topology/test_discrete_morse.py``. The preamble is carried over verbatim so the moved tests
resolve every name they use.
"""

from __future__ import annotations

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.topology._models import FiniteSimplicialComplex
from jacobian.math.topology.discrete_morse import (
    MorseMatchingOutcome,
    construct_matching,
)
from jacobian.math.topology.discrete_morse._models import (
    DiscreteMorseMatchingRequest,
    MatchingPair,
)
from jacobian.math.topology.operations import canonicalize

OPERATION_ID = "topology.discrete_morse.matching.construct"


def _request(
    vertices: list[str],
    facets: list[list[str]],
    pairs: list[tuple[list[str], list[str]]],
) -> DiscreteMorseMatchingRequest:
    return DiscreteMorseMatchingRequest.model_validate(
        {
            "complex": {"vertices": vertices, "facets": facets},
            "pairs": [{"face": face, "coface": coface} for face, coface in pairs],
        }
    )


def _args(
    vertices: list[str],
    facets: list[list[str]],
    pairs: list[tuple[list[str], list[str]]],
) -> tuple[FiniteSimplicialComplex, tuple[MatchingPair, ...]]:
    request = DiscreteMorseMatchingRequest.model_validate(
        {
            "complex": {"vertices": vertices, "facets": facets},
            "pairs": [{"face": face, "coface": coface} for face, coface in pairs],
        }
    )
    canonical = canonicalize(request.complex.vertices, request.complex.facets).complex
    return canonical, request.pairs


def _circle_args(
    pairs: list[tuple[list[str], list[str]]],
) -> tuple[FiniteSimplicialComplex, tuple[MatchingPair, ...]]:
    return _args(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]], pairs)


class TestNativeCatalogParity:
    def test_catalog_tool_runs_the_same_kernel(self) -> None:
        tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)
        request = _request(
            ["a", "b", "c"],
            [["a", "b"], ["b", "c"], ["a", "c"]],
            [(["a"], ["a", "b"]), (["c"], ["a", "c"])],
        )
        assert tool.run(request) == construct_matching(
            *_circle_args([(["a"], ["a", "b"]), (["c"], ["a", "c"])])
        )

    def test_published_examples_execute(self) -> None:
        tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)
        outcomes = []
        for example in tool.examples:
            request = DiscreteMorseMatchingRequest.model_validate(example.input)
            outcomes.append(tool.run(request).outcome)
        assert outcomes == [
            MorseMatchingOutcome.ACYCLIC_MATCHING,
            MorseMatchingOutcome.CYCLIC_MATCHING,
        ]
