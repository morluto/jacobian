from __future__ import annotations

import json
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
from jacobian.math.combinatorics.posets.core.operations import (
    materialize_finite_poset,
    verify_finite_poset,
)
from jacobian.math.topology._models import (
    ChainCoefficientRing,
    ChainComplexResult,
    HomologyConvention,
    SimplicialComplexRequest,
    canonical_complex,
)
from jacobian.math.topology._structural import FVectorRequest, compute_f_vector
from jacobian.math.topology.operations import (
    barycentric_subdivision,
    chain_complex,
    homology,
    integral_homology,
)
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
    empty_result = order_complex(OrderComplexRequest(poset=empty))
    assert empty_result.complex.vertices == ()
    assert empty_result.complex.maximal_simplices == ()
    assert empty_result.complex.faces_by_dimension == ()
    assert empty_result.complex.f_vector == ()
    assert empty_result.complex.dimension == -1
    assert empty_result.complex.closure_size == 0
    assert empty_result.maximal_chains == ()


def test_empty_order_complex_json_composes_with_downstream_consumers() -> None:
    encoded = order_complex(
        OrderComplexRequest(poset=_poset((), ()))
    ).model_dump_json()
    decoded = OrderComplexResult.model_validate_json(encoded)

    graph = one_skeleton(
        OneSkeletonRequest.model_validate_json(
            OneSkeletonRequest(complex=decoded.complex).model_dump_json()
        )
    )
    subdivision = barycentric_subdivision(decoded.complex)
    assert graph.graph.vertex_count == 0
    assert graph.graph.edges == ()
    assert graph.vertex_labels == ()
    assert subdivision.subdivision_complex is not None
    assert subdivision.subdivision_complex == decoded.complex
    assert subdivision.original_dimension == -1
    assert compute_f_vector(FVectorRequest(complex={"vertices": (), "facets": ()})).f_vector == (1,)

    face_result = FacePosetResult.model_validate_json(
        face_poset(
            FacePosetRequest(
                complex=SimplicialComplexRequest(vertices=(), facets=())
            )
        ).model_dump_json()
    )
    assert face_result.faces == ()
    assert face_result.poset is not None
    assert face_result.order_complex.dimension == -1


def test_empty_complex_composes_through_simplicial_chains_and_homology() -> None:
    complex_ = order_complex(OrderComplexRequest(poset=_poset((), ()))).complex
    reduced_chain = chain_complex(
        complex_,
        ChainCoefficientRing.PRIME_FIELD,
        2,
        HomologyConvention.REDUCED,
    )
    decoded_chain = ChainComplexResult.model_validate_json(
        reduced_chain.model_dump_json()
    )
    assert decoded_chain.canonical_value.basis_sizes == (1, 0)
    assert decoded_chain.canonical_value.degree_min == -1

    reduced_homology = homology(complex_, 2, HomologyConvention.REDUCED)
    assert reduced_homology.dimension_range == (-1, 0)
    assert tuple(group.betti_number for group in reduced_homology.groups) == (1, 0)
    assert type(reduced_homology).model_validate_json(
        reduced_homology.model_dump_json()
    ) == reduced_homology

    ordinary_homology = homology(complex_, 2, HomologyConvention.UNREDUCED)
    assert ordinary_homology.dimension_range == (0, 0)
    assert ordinary_homology.groups[0].betti_number == 0
    assert type(ordinary_homology).model_validate_json(
        ordinary_homology.model_dump_json()
    ) == ordinary_homology

    integral = integral_homology(complex_, HomologyConvention.REDUCED)
    assert integral.homology.homology_groups[0].degree == -1
    assert integral.homology.homology_groups[0].free_rank == 1
    assert type(integral).model_validate_json(integral.model_dump_json()) == integral


def test_order_complex_result_rejects_forged_axis_bindings() -> None:
    result = order_complex(OrderComplexRequest(poset=_poset(("a",), ())))
    forged = result.model_dump()
    forged["vertex_elements"] = ()
    with pytest.raises(ValueError, match="vertex_elements"):
        OrderComplexResult.model_validate(forged)

    forged = result.model_dump()
    forged["complex"].update(
        vertices=[],
        maximal_simplices=[],
        faces_by_dimension=[],
        dimension=-1,
        f_vector=[],
        closure_size=0,
    )
    with pytest.raises(ValueError, match="complex vertices"):
        OrderComplexResult.model_validate(forged)
    with pytest.raises(ValueError, match="complex vertices"):
        OrderComplexResult.model_validate_json(json.dumps(forged))


def test_order_complex_result_rejects_reversed_and_incomparable_chains() -> None:
    comparable = order_complex(OrderComplexRequest(poset=_poset(("a", "b"), (("a", "b"),))))
    forged = comparable.model_dump()
    forged["maximal_chains"] = [["b", "a"]]
    with pytest.raises(ValueError, match="cover relations"):
        OrderComplexResult.model_validate_json(json.dumps(forged))

    antichain = order_complex(OrderComplexRequest(poset=_poset(("a", "b"), ())))
    forged = antichain.model_dump()
    forged["maximal_chains"] = [["a", "b"]]
    forged["complex"] = canonical_complex(("a", "b"), (("a", "b"),)).model_dump()
    with pytest.raises(ValueError, match="cover relations"):
        OrderComplexResult.model_validate(forged)


def test_order_complex_result_rejects_coordinated_forged_poset_covers() -> None:
    result = order_complex(
        OrderComplexRequest(poset=_poset(("a", "b"), (("a", "b"),)))
    )
    forged = result.model_dump()
    forged["poset"]["cover_relations"] = [{"lower": "b", "upper": "a"}]
    forged["maximal_chains"] = [["b", "a"]]
    decoded_poset = type(result.poset).model_validate(forged["poset"])
    assert not verify_finite_poset(decoded_poset)
    with pytest.raises(ValueError, match="canonical finite poset"):
        OrderComplexResult.model_validate_json(json.dumps(forged))


def test_face_poset_result_rejects_forged_positional_labels() -> None:
    result = face_poset(
        FacePosetRequest(
            complex=SimplicialComplexRequest(vertices=("a",), facets=(("a",),))
        )
    )
    forged = result.model_dump()
    forged["face_element_labels"] = ["wrong"]
    with pytest.raises(ValueError, match="face-element labels"):
        FacePosetResult.model_validate(forged)
    with pytest.raises(ValueError, match="face-element labels"):
        FacePosetResult.model_validate_json(json.dumps(forged))


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


def test_face_poset_without_poset_rejects_relations_not_bound_to_order_complex() -> None:
    result = face_poset(
        FacePosetRequest(
            complex=SimplicialComplexRequest(vertices=("a", "b"), facets=(("a", "b"),))
        )
    ).model_copy(update={"poset": None})
    forged = result.model_dump()
    forged["order_relations"] = [[0, 1]]
    with pytest.raises(ValueError, match="order-complex edges"):
        FacePosetResult.model_validate_json(json.dumps(forged))
