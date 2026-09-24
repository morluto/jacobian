from __future__ import annotations

from itertools import combinations, pairwise

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.posets.core._models import (
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import materialize_finite_poset
from jacobian.math.topology._models import SimplicialComplexRequest
from jacobian.math.topology.operations import barycentric_subdivision
from jacobian.math.topology.release import (
    FacePosetRequest,
    FacePosetResult,
    OneSkeletonRequest,
    OrderComplexRequest,
    OrderComplexResult,
    face_poset,
    one_skeleton,
    order_complex,
)


def _poset(elements: tuple[str, ...], covers: tuple[tuple[str, str], ...]):
    return materialize_finite_poset(
        elements,
        tuple(PresentationPair(lower=lower, upper=upper) for lower, upper in covers),
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )


def test_order_complex_matches_all_pairwise_comparable_subsets() -> None:
    poset = _poset(
        ("a", "b", "c", "d"),
        (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")),
    )

    result = order_complex(OrderComplexRequest(poset=poset))

    strict = {(pair.lower, pair.upper) for pair in poset.strict_order_pairs}
    expected_faces: set[tuple[str, ...]] = set()
    for size in range(1, len(poset.elements) + 1):
        for subset in combinations(poset.elements, size):
            if all(
                (lower, upper) in strict or (upper, lower) in strict
                for lower, upper in combinations(subset, 2)
            ):
                expected_faces.add(subset)
    expected_facets = {
        face
        for face in expected_faces
        if not any(set(face) < set(other) for other in expected_faces)
    }
    actual_faces = {
        face for group in result.complex.faces_by_dimension for face in group.faces
    }
    assert actual_faces == expected_faces
    assert set(result.complex.maximal_simplices) == expected_facets
    assert result.complex.f_vector == (4, 5, 2)
    assert result.maximal_chains == (("a", "b", "d"), ("a", "c", "d"))
    assert result.vertex_elements == poset.elements


def test_antichain_singleton_and_empty_poset_contract() -> None:
    antichain = order_complex(OrderComplexRequest(poset=_poset(("a", "b", "c"), ())))
    assert antichain.complex.f_vector == (3,)
    assert antichain.complex.maximal_simplices == (("a",), ("b",), ("c",))

    singleton = order_complex(OrderComplexRequest(poset=_poset(("x",), ())))
    assert singleton.complex.vertices == ("x",)
    assert singleton.maximal_chains == (("x",),)

    empty = _poset((), ())
    with pytest.raises(OperationDomainValidationError, match="empty poset"):
        order_complex(OrderComplexRequest(poset=empty))


def test_order_complex_json_composes_with_one_skeleton() -> None:
    poset = _poset(
        ("a", "b", "c", "d"),
        (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")),
    )
    decoded = OrderComplexResult.model_validate_json(
        order_complex(OrderComplexRequest(poset=poset)).model_dump_json()
    )

    graph = one_skeleton(
        OneSkeletonRequest.model_validate_json(
            OneSkeletonRequest(complex=decoded.complex).model_dump_json()
        )
    )

    assert graph.vertex_labels == poset.elements
    assert graph.edge_faces == (
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "d"),
        ("c", "d"),
    )


def test_face_poset_json_composes_through_order_complex_and_barycentric_transport() -> (
    None
):
    source_request = SimplicialComplexRequest.model_validate(
        {"vertices": ["a", "b", "c"], "facets": [["a", "b", "c"]]}
    )
    face_result = FacePosetResult.model_validate_json(
        face_poset(FacePosetRequest(complex=source_request)).model_dump_json()
    )
    assert face_result.poset is not None
    assert len(face_result.faces) == len(face_result.face_element_labels) == 7

    order_result = OrderComplexResult.model_validate_json(
        order_complex(
            OrderComplexRequest.model_validate_json(
                OrderComplexRequest(poset=face_result.poset).model_dump_json()
            )
        ).model_dump_json()
    )
    assert order_result.complex == face_result.order_complex

    subdivision = barycentric_subdivision(face_result.complex)
    assert subdivision.subdivision_complex is not None
    face_to_subdivision_vertex = dict(
        zip(
            subdivision.subdivision_vertex_faces,
            subdivision.subdivision_vertices,
            strict=True,
        )
    )
    labels_to_subdivision_vertices = {
        label: face_to_subdivision_vertex[face]
        for label, face in zip(
            face_result.face_element_labels, face_result.faces, strict=True
        )
    }
    transported_facets = {
        tuple(sorted(labels_to_subdivision_vertices[label] for label in facet))
        for facet in order_result.complex.maximal_simplices
    }
    assert transported_facets == set(subdivision.subdivision_complex.maximal_simplices)

    graph = one_skeleton(
        OneSkeletonRequest.model_validate_json(
            OneSkeletonRequest(complex=order_result.complex).model_dump_json()
        )
    )
    assert graph.vertex_labels == order_result.vertex_elements
    assert len(graph.edge_faces) == len(face_result.poset.strict_order_pairs)


def test_order_complex_bounds_before_chain_materialization() -> None:
    admissible = order_complex(
        OrderComplexRequest(
            poset=_poset(
                tuple(f"v{i}" for i in range(8)),
                tuple((f"v{i}", f"v{i + 1}") for i in range(7)),
            )
        )
    )
    assert admissible.complex.dimension == 7
    assert admissible.complex.closure_size == 255

    too_long = _poset(
        tuple(f"v{i}" for i in range(9)),
        tuple((f"v{i}", f"v{i + 1}") for i in range(8)),
    )
    with pytest.raises(OperationResourceAdmissionError, match="longest chain"):
        order_complex(OrderComplexRequest(poset=too_long))

    boolean_lattice_elements = tuple(f"s{mask:02d}" for mask in range(32))
    boolean_lattice_covers = tuple(
        (f"s{mask:02d}", f"s{mask | (1 << bit):02d}")
        for mask in range(32)
        for bit in range(5)
        if not mask & (1 << bit)
    )
    boolean_lattice = _poset(boolean_lattice_elements, boolean_lattice_covers)
    with pytest.raises(OperationResourceAdmissionError, match="more nonempty chains"):
        order_complex(OrderComplexRequest(poset=boolean_lattice))

    layers = (3, 3, 2, 2, 2, 2)
    layer_elements = tuple(
        f"l{level}e{element}"
        for level, size in enumerate(layers)
        for element in range(size)
    )
    layer_ranges: list[tuple[str, ...]] = []
    for level, size in enumerate(layers):
        layer_ranges.append(tuple(f"l{level}e{element}" for element in range(size)))
    layered_covers = tuple(
        (lower, upper)
        for lower_layer, upper_layer in pairwise(layer_ranges)
        for lower in lower_layer
        for upper in upper_layer
    )
    too_many_facets = _poset(layer_elements, layered_covers)
    with pytest.raises(OperationResourceAdmissionError, match="maximal-chain facets"):
        order_complex(OrderComplexRequest(poset=too_many_facets))


def test_order_complex_rechecks_caller_supplied_poset_claims() -> None:
    poset = _poset(("a", "b"), (("a", "b"),))
    malformed = poset.model_copy(update={"poset_digest": "sha256:" + "0" * 64})

    with pytest.raises(OperationDomainValidationError, match="canonical finite poset"):
        order_complex(OrderComplexRequest(poset=malformed))
