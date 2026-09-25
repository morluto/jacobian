"""Tests for simplicial complex link operation."""

from itertools import combinations
from typing import TypedDict

from jacobian.math.topology._models import SimplicialComplexRequest, canonical_complex
from jacobian.math.topology._structural import (
    FVectorRequest,
    LinkRequest,
    compute_link,
)


class ComplexWire(TypedDict):
    """Facet-presentation payload at a raw JSON boundary."""

    vertices: list[str]
    facets: list[list[str]]


def _complex(data: ComplexWire) -> SimplicialComplexRequest:
    """Validate a raw facet presentation at the model boundary."""

    return SimplicialComplexRequest.model_validate(data)


def test_link_of_vertex_in_triangle() -> None:
    result = compute_link(
        LinkRequest(
            complex=_complex(
                {"vertices": ["v0", "v1", "v2"], "facets": [["v0", "v1", "v2"]]}
            ),
            simplex=("v0",),
        )
    )
    assert result.link_facets == (("v1", "v2"),)
    assert result.link_is_empty is False


def test_link_of_edge_in_triangle() -> None:
    result = compute_link(
        LinkRequest(
            complex=_complex(
                {"vertices": ["v0", "v1", "v2"], "facets": [["v0", "v1", "v2"]]}
            ),
            simplex=("v0", "v1"),
        )
    )
    assert result.link_facets == (("v2",),)


def test_link_of_facet_is_the_canonical_empty_face_complex() -> None:
    result = compute_link(
        LinkRequest(
            complex=_complex({"vertices": ["a", "b"], "facets": [["a", "b"]]}),
            simplex=("a", "b"),
        )
    )
    assert result.link_facets == ()
    assert result.link_is_empty
    assert result.link_complex == canonical_complex((), ())
    assert result.link_complex.dimension == -1
    assert type(result).model_validate(result.model_dump()) == result


def test_empty_face_link_matches_an_independent_face_set_oracle() -> None:
    source = _complex({"vertices": ["a", "b", "c"], "facets": [["a", "b"], ["b", "c"]]})
    result = compute_link(LinkRequest(complex=source, simplex=()))

    source_faces = {
        tuple(sorted(face))
        for facet in source.facets
        for size in range(1, len(facet) + 1)
        for face in combinations(facet, size)
    }
    oracle_facets = tuple(
        sorted(
            (
                face
                for face in source_faces
                if not any(set(face) < set(other) for other in source_faces)
            ),
            key=lambda face: (-len(face), face),
        )
    )
    assert result.link_facets == oracle_facets
    assert result.link_complex == canonical_complex(source.vertices, source.facets)
    assert not result.link_is_empty


def test_empty_face_link_of_empty_complex_is_empty_complex() -> None:
    result = compute_link(
        LinkRequest(complex=_complex({"vertices": [], "facets": []}), simplex=())
    )
    assert result.link_is_empty
    assert result.link_complex.dimension == -1
    assert result.link_complex.vertices == ()
    assert result.link_complex.maximal_simplices == ()


def test_link_of_vertex_in_discrete_complex() -> None:
    result = compute_link(
        LinkRequest(
            complex=_complex({"vertices": ["v0", "v1"], "facets": [["v0"], ["v1"]]}),
            simplex=("v0",),
        )
    )
    assert result.link_is_empty is True


def test_link_of_face_in_boundary() -> None:
    result = compute_link(
        LinkRequest(
            complex=_complex(
                {
                    "vertices": ["v0", "v1", "v2"],
                    "facets": [["v0", "v1"], ["v1", "v2"], ["v0", "v2"]],
                }
            ),
            simplex=("v0",),
        )
    )
    assert ("v1",) in result.link_facets
    assert ("v2",) in result.link_facets


def test_f_vector_filled_triangle() -> None:
    """Filled triangle has f=(3,3,1) and h=(1,0,0,0)."""
    from jacobian.math.topology._structural import compute_f_vector

    request = FVectorRequest(
        complex=_complex(
            {"vertices": ["v0", "v1", "v2"], "facets": [["v0", "v1", "v2"]]}
        )
    )
    result = compute_f_vector(request)
    assert result.f_vector == (1, 3, 3, 1)
    assert result.h_vector == (1, 0, 0, 0)


def test_f_vector_single_edge() -> None:
    """Single edge has f=(2,1) and h=(1,0,0)."""
    from jacobian.math.topology._structural import compute_f_vector

    request = FVectorRequest(
        complex=_complex({"vertices": ["v0", "v1"], "facets": [["v0", "v1"]]})
    )
    result = compute_f_vector(request)
    assert result.f_vector == (1, 2, 1)
    assert result.h_vector == (1, 0, 0)


def test_f_vector_h_vector_invariant() -> None:
    """h_0 must always be 1 for a non-empty complex."""
    from jacobian.math.topology._structural import compute_f_vector

    request = FVectorRequest(
        complex=_complex(
            {"vertices": ["v0", "v1", "v2"], "facets": [["v0", "v1", "v2"]]}
        )
    )
    result = compute_f_vector(request)
    assert result.h_vector[0] == 1


def test_link_rejects_non_face() -> None:
    """Non-face simplex should be rejected, not silently returned as empty link."""
    try:
        compute_link(
            LinkRequest(
                complex=_complex(
                    {
                        "vertices": ["v0", "v1", "v2", "v3"],
                        "facets": [["v0", "v1"], ["v2", "v3"]],
                    }
                ),
                simplex=("v0", "v2"),
            )
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
