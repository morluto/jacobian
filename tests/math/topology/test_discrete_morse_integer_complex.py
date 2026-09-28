"""Exact signed Morse-differential checks against simplicial chain data."""

from __future__ import annotations

import json

from jacobian.math.topology.discrete_morse._models import MatchingPair
from jacobian.math.topology.discrete_morse._tools import TOOLS
from jacobian.math.topology.discrete_morse.operations import (
    compute_integer_morse_complex,
)
from jacobian.math.topology.operations import canonicalize


def _complex(vertices, facets):
    return canonicalize(tuple(vertices), tuple(tuple(face) for face in facets)).complex


def test_empty_matching_is_the_oriented_simplicial_chain_complex() -> None:
    source = _complex(("a", "b", "c"), (("a", "b", "c"),))

    result = compute_integer_morse_complex(source, ())

    assert result.chain_complex.coefficient_ring == "ZZ"
    assert result.chain_complex.basis_sizes == (3, 3, 1)
    assert result.chain_complex.differential_matrices == (
        ((-1, -1, 0), (1, 0, -1), (0, 1, 1)),
        ((1,), (-1,), (1,)),
    )


def test_projective_plane_matching_retains_the_order_two_torsion() -> None:
    # The standard six-vertex triangulation of RP^2 has H_1 = Z/2. This
    # explicit acyclic matching reduces it to one critical cell in each
    # dimension; the independently known minimal Morse differential is 2.
    facets = (
        ("0", "1", "2"),
        ("0", "1", "3"),
        ("0", "2", "4"),
        ("0", "3", "5"),
        ("0", "4", "5"),
        ("1", "2", "5"),
        ("1", "3", "4"),
        ("1", "4", "5"),
        ("2", "3", "4"),
        ("2", "3", "5"),
    )
    pairs = tuple(
        MatchingPair(face=face, coface=coface)
        for face, coface in (
            (("1",), ("0", "1")),
            (("1", "2"), ("0", "1", "2")),
            (("1", "3"), ("0", "1", "3")),
            (("1", "5"), ("1", "4", "5")),
            (("2",), ("0", "2")),
            (("2", "3"), ("2", "3", "4")),
            (("2", "4"), ("0", "2", "4")),
            (("2", "5"), ("1", "2", "5")),
            (("3",), ("0", "3")),
            (("3", "4"), ("1", "3", "4")),
            (("3", "5"), ("0", "3", "5")),
            (("4",), ("0", "4")),
            (("4", "5"), ("0", "4", "5")),
            (("5",), ("0", "5")),
        )
    )

    result = compute_integer_morse_complex(_complex(tuple("012345"), facets), pairs)

    assert result.chain_complex.basis_sizes == (1, 1, 1)
    assert result.chain_complex.differential_matrices == (((0,),), ((-2,),))
    assert result.gradient_path_total == 4
    # For this based complex d_1=0 and d_2=-2, so H_1 is Z/2.
    d1, d2 = result.chain_complex.differential_matrices
    assert d1 == ((0,),)
    assert abs(d2[0][0]) == 2


def test_integer_morse_catalog_example_runs() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "topology.discrete_morse.integer_complex.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))

    result = tool.run(request)

    assert result.chain_complex.basis_sizes == (2, 1)
    assert result.chain_complex.differential_matrices == (((-1,), (1,)),)
