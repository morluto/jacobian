from __future__ import annotations

from itertools import combinations, pairwise

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.posets.core._models import (
    FinitePoset,
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import materialize_finite_poset
from jacobian.math.topology.operations import barycentric_subdivision, canonicalize
from jacobian.math.topology.release import (
    OrderComplexRequest,
    OrderComplexResult,
    order_complex,
)


def _poset(
    elements: tuple[str, ...], covers: tuple[tuple[str, str], ...]
) -> FinitePoset:
    return materialize_finite_poset(
        elements,
        tuple(PresentationPair(lower=lower, upper=upper) for lower, upper in covers),
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )


def test_order_complex_rejects_non_request_native_arguments() -> None:
    for invalid in ({"poset": _poset(("a",), ())}, _poset(("a",), ())):
        with pytest.raises(OperationDomainValidationError) as exc_info:
            order_complex(invalid)  # type: ignore[arg-type]
        assert (
            exc_info.value.errors()[0]["type"]
            == "topology.order_complex.invalid_request"
        )


def test_order_complex_revalidates_forged_request_before_dereferencing() -> None:
    for invalid in (
        OrderComplexRequest.model_construct(),
        OrderComplexRequest.model_construct(poset=None),
    ):
        with pytest.raises(OperationDomainValidationError) as error:
            order_complex(invalid)
        assert (
            error.value.errors()[0]["type"] == "topology.order_complex.invalid_request"
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
    with pytest.raises(OperationDomainValidationError) as exc_info:
        order_complex(OrderComplexRequest(poset=empty))
    assert exc_info.value.errors()[0]["type"] == "topology.order_complex.empty_poset"


def test_order_complex_of_a_face_poset_is_the_barycentric_subdivision() -> None:
    """The order complex of the face poset is the barycentric subdivision.

    This is the mathematical reason the operation exists, so it is checked
    against the barycentric subdivision of the same complex rather than against
    a hand-written chain list.
    """
    source = canonicalize(("a", "b", "c"), (("a", "b", "c"),)).complex
    faces = [face for group in source.faces_by_dimension for face in group.faces]
    label_of = {face: f"F{index}" for index, face in enumerate(faces)}
    # Face poset: strict inclusion between faces, covered by the immediately
    # larger faces.
    covers = tuple(
        (label_of[face], label_of[superset])
        for face in faces
        for superset in faces
        if set(face) < set(superset)
        and not any(set(face) < set(middle) < set(superset) for middle in faces)
    )
    face_poset = _poset(tuple(label_of.values()), covers)

    result = order_complex(OrderComplexRequest(poset=face_poset))

    subdivision = barycentric_subdivision(source)
    assert subdivision.subdivision_complex is not None
    assert result.complex.dimension == subdivision.subdivision_complex.dimension
    assert result.complex.f_vector == subdivision.subdivision_complex.f_vector
    # Maximal chains of a face poset are maximal flags, so a subdivided triangle
    # has six of them, not one per face.
    assert len(result.complex.maximal_simplices) == len(
        subdivision.subdivision_complex.maximal_simplices
    )
    assert all(len(chain) == 3 for chain in result.maximal_chains)


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
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        order_complex(OrderComplexRequest(poset=too_long))
    assert (
        exc_info.value.errors()[0]["type"] == "topology.order_complex.dimension_budget"
    )

    boolean_lattice_elements = tuple(f"s{mask:02d}" for mask in range(32))
    boolean_lattice_covers = tuple(
        (f"s{mask:02d}", f"s{mask | (1 << bit):02d}")
        for mask in range(32)
        for bit in range(5)
        if not mask & (1 << bit)
    )
    boolean_lattice = _poset(boolean_lattice_elements, boolean_lattice_covers)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        order_complex(OrderComplexRequest(poset=boolean_lattice))
    assert exc_info.value.errors()[0]["type"] == "topology.order_complex.face_budget"

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
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        order_complex(OrderComplexRequest(poset=too_many_facets))
    assert exc_info.value.errors()[0]["type"] == "topology.order_complex.facet_budget"


def test_order_complex_rechecks_caller_supplied_poset_claims() -> None:
    poset = _poset(("a", "b"), (("a", "b"),))
    malformed = poset.model_copy(update={"poset_digest": "sha256:" + "0" * 64})

    with pytest.raises(OperationDomainValidationError) as exc_info:
        order_complex(OrderComplexRequest(poset=malformed))
    assert exc_info.value.errors()[0]["type"] == "topology.order_complex.invalid_poset"


def test_order_complex_result_survives_json_round_trip() -> None:
    poset = _poset(
        ("a", "b", "c", "d"),
        (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")),
    )
    produced = order_complex(OrderComplexRequest(poset=poset))
    decoded = OrderComplexResult.model_validate_json(produced.model_dump_json())
    assert decoded == produced
    # A decoded result is a canonical value, so it feeds the kernel's own
    # request model without re-deriving the poset.
    assert order_complex(OrderComplexRequest(poset=decoded.poset)) == produced
