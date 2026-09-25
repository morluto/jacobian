"""Exact chain-homotopy witnesses for bounded discrete Morse reductions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import (
    FacesInDimension,
    FiniteSimplicialComplex,
    SimplicialComplexRequest,
)
from jacobian.math.topology.discrete_morse._models import (
    MatchingPair,
    MorseChainContractionRequest,
)
from jacobian.math.topology.discrete_morse._tools import TOOLS
from jacobian.math.topology.discrete_morse.operations import (
    compute_chain_contraction,
    compute_integer_morse_complex,
)
from jacobian.math.topology.operations import canonicalize


def _complex(vertices, facets):
    return canonicalize(tuple(vertices), tuple(tuple(face) for face in facets)).complex


INTERVAL = _complex(("a", "b"), (("a", "b"),))
INTERVAL_MATCH = (MatchingPair(face=("a",), coface=("a", "b")),)
FILLED_TRIANGLE = _complex(("a", "b", "c"), (("a", "b", "c"),))
TRIANGLE_MATCH = tuple(
    MatchingPair(face=face, coface=coface)
    for face, coface in (
        (("b",), ("a", "b")),
        (("c",), ("b", "c")),
        (("a", "c"), ("a", "b", "c")),
    )
)


def test_interval_returns_exact_inclusion_projection_and_chain_homotopy() -> None:
    result = compute_chain_contraction(INTERVAL, INTERVAL_MATCH)

    assert result.source_chain_complex.differential_matrices == (((-1,), (1,)),)
    assert result.morse_chain_complex.basis_sizes == (1, 0)
    assert result.inclusion_matrices == (((0,), (1,)), ((),))
    assert result.projection_matrices == (((1, 1),), ())
    assert result.homotopy_matrices == (((-1, 0),), ())
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result

    payload = result.model_dump(mode="python")
    payload["critical_cells_by_dimension"] = (
        {**payload["critical_cells_by_dimension"][0], "cells": (("x",),)},
        *payload["critical_cells_by_dimension"][1:],
    )
    with pytest.raises(ValidationError, match="critical bases"):
        type(result).model_validate(payload)

    payload = result.model_dump(mode="python")
    payload["pairs"] = ({**payload["pairs"][0], "face": ("x",)},)
    with pytest.raises(ValidationError, match="matching pairs"):
        type(result).model_validate(payload)


def test_empty_matching_is_identity_and_filled_triangle_has_chain_homotopy() -> None:
    identity = compute_chain_contraction(INTERVAL, ())
    assert identity.morse_chain_complex == identity.source_chain_complex
    assert identity.inclusion_matrices == (((1, 0), (0, 1)), ((1,),))
    assert identity.projection_matrices == identity.inclusion_matrices
    assert identity.homotopy_matrices == (((0, 0),), ())

    triangle = compute_chain_contraction(FILLED_TRIANGLE, TRIANGLE_MATCH)
    integer_morse = compute_integer_morse_complex(FILLED_TRIANGLE, TRIANGLE_MATCH)
    assert (
        triangle.critical_cells_by_dimension
        == integer_morse.critical_cells_by_dimension
    )
    assert triangle.morse_chain_complex == integer_morse.chain_complex
    assert triangle.morse_chain_complex.basis_sizes == (1, 0, 0)


def test_projective_plane_contraction_preserves_integral_two_torsion() -> None:
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
    projective_plane = _complex(tuple("012345"), facets)

    contraction = compute_chain_contraction(projective_plane, pairs)
    morse = compute_integer_morse_complex(projective_plane, pairs)

    assert contraction.source_chain_complex.basis_sizes == (6, 15, 10)
    assert contraction.morse_chain_complex == morse.chain_complex
    assert contraction.morse_chain_complex.basis_sizes == (1, 1, 1)
    assert contraction.morse_chain_complex.differential_matrices == (((0,),), ((-2,),))


def test_cyclic_matching_is_rejected_as_a_domain_error() -> None:
    circle = _complex(("a", "b", "c"), (("a", "b"), ("b", "c"), ("a", "c")))
    cyclic = tuple(
        MatchingPair(face=face, coface=coface)
        for face, coface in (
            (("a",), ("a", "b")),
            (("b",), ("b", "c")),
            (("c",), ("a", "c")),
        )
    )
    with pytest.raises(OperationDomainValidationError, match="closed V-path"):
        compute_chain_contraction(circle, cyclic)


def test_tool_catalog_entry_executes_the_same_chain_contraction() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "topology.discrete_morse.chain_contraction.compute"
    )
    request = tool.request_type.model_validate(
        {
            "complex": {"vertices": ["a", "b"], "facets": [["a", "b"]]},
            "pairs": [{"face": ["a"], "coface": ["a", "b"]}],
        }
    )
    assert tool.run(request) == compute_chain_contraction(INTERVAL, INTERVAL_MATCH)


def test_tool_request_candidate_bound_runs_before_complex_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.topology.discrete_morse._tools as tools

    facet = tuple(f"v{i}" for i in range(6))
    request = MorseChainContractionRequest.model_construct(
        complex=SimplicialComplexRequest.model_construct(
            vertices=facet, facets=(facet,) * 9
        ),
        pairs=(),
    )

    def fail_on_canonicalize(*args, **kwargs):
        pytest.fail("complex canonicalization ran before candidate admission")

    monkeypatch.setattr(tools, "canonicalize", fail_on_canonicalize)
    with pytest.raises(OperationResourceAdmissionError, match="512-candidate"):
        tools._run_compute_chain_contraction(request)


def test_direct_model_construct_facet_shape_is_bounded_before_dump() -> None:
    oversized = FiniteSimplicialComplex.model_construct(
        vertices=("a",),
        maximal_simplices=(tuple(f"v{i}" for i in range(100_000)),),
        faces_by_dimension=(),
        dimension=0,
        f_vector=(),
        closure_size=0,
    )
    with pytest.raises(OperationResourceAdmissionError, match="axes exceed"):
        compute_chain_contraction(oversized, ())


def test_direct_model_construct_face_shape_is_bounded_before_work_sizing() -> None:
    oversized = FiniteSimplicialComplex.model_construct(
        vertices=("a",),
        maximal_simplices=(("a",),),
        faces_by_dimension=(
            FacesInDimension.model_construct(dimension=0, faces=(("a",) * 100_000,)),
        ),
        dimension=0,
        f_vector=(1,),
        closure_size=1,
    )
    with pytest.raises(OperationResourceAdmissionError, match="face groups"):
        compute_chain_contraction(oversized, ())
