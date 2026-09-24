import itertools

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.operations import canonicalize
from jacobian.math.topology.simplicial_sets import (
    SimplicialComplexPrefixRequest,
    complex_conversion,
    simplicial_set_from_complex,
)
from jacobian.math.topology.simplicial_sets._tools import TOOLS
from jacobian.math.topology.simplicial_sets.maps import normalized_chains


def _complex(vertices, facets):
    return canonicalize(vertices, facets).complex


def _independent_degree(complex_, degree):
    vertex_index = {vertex: index for index, vertex in enumerate(complex_.vertices)}
    faces = {
        frozenset(vertex_index[vertex] for vertex in face)
        for level in complex_.faces_by_dimension
        for face in level.faces
    }
    return {
        tuple(values)
        for values in itertools.combinations_with_replacement(
            range(len(complex_.vertices)), degree + 1
        )
        if frozenset(values) in faces
    }


def _decode(level):
    return tuple(tuple(map(int, label.split(","))) for label in level)


def test_triangle_complex_prefix_matches_all_monotone_face_tuples():
    source = _complex(("a", "b", "c"), (("a", "b", "c"),))
    result = simplicial_set_from_complex(
        SimplicialComplexPrefixRequest(complex=source, max_degree=3)
    )
    value = result.simplicial_set

    for degree, labels in enumerate(value.sets):
        assert set(_decode(labels)) == _independent_degree(source, degree)

    # Face and degeneracy rows agree with literal deletion and repetition on
    # every generated tuple, independently of the kernel's index construction.
    tuples = tuple(_decode(level) for level in value.sets)
    for degree in range(1, value.max_degree + 1):
        for omitted, row in enumerate(value.face_maps[degree - 1]):
            for simplex, target in zip(tuples[degree], row, strict=True):
                assert tuples[degree - 1][target] == (
                    simplex[:omitted] + simplex[omitted + 1 :]
                )
    for degree in range(value.max_degree):
        for repeated, row in enumerate(value.degeneracy_maps[degree]):
            for simplex, target in zip(tuples[degree], row, strict=True):
                expected = (
                    *simplex[: repeated + 1],
                    simplex[repeated],
                    *simplex[repeated + 1 :],
                )
                assert tuples[degree + 1][target] == expected

    # The source-face table is complete and each face is the strictly
    # increasing simplex at the reported nondegenerate index.
    mapped = {
        (entry.dimension, entry.face_index) for entry in result.face_simplex_indices
    }
    expected = {
        (dimension, face_index)
        for dimension, level in enumerate(source.faces_by_dimension[:4])
        for face_index, _face in enumerate(level.faces)
    }
    assert mapped == expected
    for entry in result.face_simplex_indices:
        face = source.faces_by_dimension[entry.dimension].faces[entry.face_index]
        labels = value.sets[entry.dimension]
        assert _decode((labels[entry.simplex_index],))[0] == tuple(
            source.vertices.index(vertex) for vertex in face
        )


def test_conversion_serializes_and_composes_with_normalized_chains():
    source = _complex(("a", "b"), (("a", "b"),))
    result = simplicial_set_from_complex(
        SimplicialComplexPrefixRequest(complex=source, max_degree=2)
    )
    rebuilt = type(result).model_validate_json(result.model_dump_json())
    assert rebuilt == result
    normalized = normalized_chains(rebuilt.simplicial_set)
    assert tuple(map(len, normalized.nondegenerate_bases)) == (2, 1, 0)
    assert normalized.chain_complex.differential_matrices[0] == ((-1,), (1,))
    assert normalized.chain_complex.basis_sizes == (2, 1, 0)
    assert type(normalized).model_validate_json(normalized.model_dump_json()) == normalized


def test_exact_prefix_count_is_admitted_before_materializing_degrees(monkeypatch):
    source = _complex(tuple("abcdef"), (tuple("abcdef"),))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("degree enumeration started before admission")

    monkeypatch.setattr(complex_conversion, "_level_labels", unexpected)
    with pytest.raises(OperationResourceAdmissionError) as exc:
        simplicial_set_from_complex(
            SimplicialComplexPrefixRequest(complex=source, max_degree=4)
        )
    assert (
        exc.value.errors()[0]["type"] == "simplicial_set.complex_prefix_degree_budget"
    )


def test_output_bound_is_checked_before_materializing_degrees(monkeypatch):
    source = _complex(("a", "b"), (("a", "b"),))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("degree enumeration started before output admission")

    monkeypatch.setattr(complex_conversion, "_level_labels", unexpected)
    monkeypatch.setattr(complex_conversion, "MAX_COMPLEX_PREFIX_OUTPUT_BYTES", 0)
    with pytest.raises(OperationResourceAdmissionError) as exc:
        simplicial_set_from_complex(
            SimplicialComplexPrefixRequest(complex=source, max_degree=2)
        )
    assert (
        exc.value.errors()[0]["type"] == "simplicial_set.complex_prefix_output_budget"
    )


def test_prefix_counts_include_degeneracies_but_source_transport_is_degree_limited():
    source = _complex(("a", "b", "c"), (("a", "b", "c"),))
    result = simplicial_set_from_complex(
        SimplicialComplexPrefixRequest(complex=source, max_degree=1)
    )
    assert tuple(map(len, result.simplicial_set.sets)) == (3, 6)
    assert len(result.face_simplex_indices) == 6  # Three vertices and three edges.
    assert {entry.dimension for entry in result.face_simplex_indices} == {0, 1}


def test_forged_inconsistent_source_face_closure_is_rejected():
    source = _complex(("a", "b"), (("a", "b"),))
    forged = source.model_copy(update={"closure_size": 1})
    with pytest.raises((OperationDomainValidationError, ValidationError)):
        simplicial_set_from_complex(
            SimplicialComplexPrefixRequest(complex=forged, max_degree=1)
        )


def test_published_catalog_example_executes_through_owner_manifest():
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id
        == "topology.simplicial_set.from_simplicial_complex.compute"
    )
    example = operation.examples[0]
    result = operation.run(operation.request_type.model_validate(example.input))
    assert result.simplicial_set.max_degree == 2
    assert tuple(map(len, result.simplicial_set.sets)) == (2, 3, 4)
