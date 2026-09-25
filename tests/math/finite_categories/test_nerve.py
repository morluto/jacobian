"""Independent checks for the finite-category nerve operation."""

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_categories import (
    FiniteCategory,
    MorphismSpec,
    nerve_prefix,
)
from jacobian.math.finite_categories._tools import TOOLS


def _interval() -> FiniteCategory:
    return FiniteCategory(
        objects=("A", "B"),
        morphisms=(
            MorphismSpec(morphism_id="id_A", source="A", target="A"),
            MorphismSpec(morphism_id="id_B", source="B", target="B"),
            MorphismSpec(morphism_id="f", source="A", target="B"),
        ),
        identities=(("A", "id_A"), ("B", "id_B")),
        composition=(
            ("id_A", "id_A", "id_A"),
            ("f", "id_A", "f"),
            ("id_B", "id_B", "id_B"),
            ("id_B", "f", "f"),
        ),
    )


def test_native_nerve_rejects_non_integer_degrees() -> None:
    for degree in (1.0, "1", True):
        with pytest.raises(OperationDomainValidationError):
            nerve_prefix(_interval(), degree)  # type: ignore[arg-type]


def test_interval_nerve_is_a_reusable_prefix_with_canonical_transport() -> None:
    nerve = nerve_prefix(_interval(), 2)
    simplicial_set = nerve.simplicial_set
    assert nerve.category == _interval()
    assert tuple(map(len, simplicial_set.sets)) == (2, 3, 4)
    assert nerve.simplex_morphisms[0] == ((), ())
    one_simplices = dict(
        zip(simplicial_set.sets[1], nerve.simplex_morphisms[1], strict=True)
    )
    assert set(one_simplices.values()) == {("id_A",), ("f",), ("id_B",)}

    # In degree 1, d_0 selects the target and d_1 the source.
    face_d0, face_d1 = simplicial_set.face_maps[0]
    vertices = nerve.simplex_objects[0]
    for simplex_index, (morphism,) in enumerate(nerve.simplex_morphisms[1]):
        source, target = (
            nerve.simplex_objects[1][simplex_index][0],
            nerve.simplex_objects[1][simplex_index][-1],
        )
        assert vertices[face_d1[simplex_index]][0] == source
        assert vertices[face_d0[simplex_index]][0] == target
        assert morphism in {"id_A", "id_B", "f"}

    # Degeneracies insert the identity of the repeated vertex.
    for row in simplicial_set.degeneracy_maps[0]:
        for vertex_index, target_index in enumerate(row):
            object_id = vertices[vertex_index][0]
            assert nerve.simplex_morphisms[1][target_index] == (
                "id_A" if object_id == "A" else "id_B",
            )
            assert nerve.simplex_objects[1][target_index] == (object_id, object_id)

    restored = type(nerve).model_validate_json(nerve.model_dump_json())
    assert restored == nerve


def test_one_object_group_nerve_uses_composition_for_inner_faces() -> None:
    category = FiniteCategory(
        objects=("*",),
        morphisms=(
            MorphismSpec(morphism_id="e", source="*", target="*"),
            MorphismSpec(morphism_id="g", source="*", target="*"),
        ),
        identities=(("*", "e"),),
        composition=(
            ("e", "e", "e"),
            ("e", "g", "g"),
            ("g", "e", "g"),
            ("g", "g", "e"),
        ),
    )
    nerve = nerve_prefix(category, 2)
    simplicial_set = nerve.simplicial_set
    assert tuple(map(len, simplicial_set.sets)) == (1, 2, 4)
    two_simplices = nerve.simplex_morphisms[2]
    degree_one = {path: index for index, path in enumerate(nerve.simplex_morphisms[1])}
    middle_face = simplicial_set.face_maps[1][1]
    for index, (first, second) in enumerate(two_simplices):
        composite = "e" if first == second == "g" else second if first == "e" else first
        assert middle_face[index] == degree_one[(composite,)]


def test_nerve_simplicial_identities_are_exhausted_and_retained() -> None:
    simplicial_set = nerve_prefix(_interval(), 2).simplicial_set
    # The exact count is the finite identity families implemented by the set owner.
    assert simplicial_set.checked_identities == 3 + 1 + 8


def test_nerve_growth_is_rejected_before_materializing_an_oversized_degree() -> None:
    order = 5
    labels = tuple(f"g{i}" for i in range(order))
    category = FiniteCategory(
        objects=("*",),
        morphisms=tuple(
            MorphismSpec(morphism_id=label, source="*", target="*") for label in labels
        ),
        identities=(("*", "g0"),),
        composition=tuple(
            (f"g{i}", f"g{j}", f"g{(i + j) % order}")
            for i in range(order)
            for j in range(order)
        ),
    )
    with pytest.raises(OperationResourceAdmissionError, match="degree 3"):
        nerve_prefix(category, 3)


def _left_identity_violation() -> FiniteCategory:
    # id_B∘g is declared to be f (both A→B), breaking the left identity law
    # id_B∘g = g while every presentation-shape check still passes.
    return FiniteCategory(
        objects=("A", "B"),
        morphisms=(
            MorphismSpec(morphism_id="id_A", source="A", target="A"),
            MorphismSpec(morphism_id="id_B", source="B", target="B"),
            MorphismSpec(morphism_id="f", source="A", target="B"),
            MorphismSpec(morphism_id="g", source="A", target="B"),
        ),
        identities=(("A", "id_A"), ("B", "id_B")),
        composition=(
            ("id_A", "id_A", "id_A"),
            ("f", "id_A", "f"),
            ("g", "id_A", "g"),
            ("id_B", "id_B", "id_B"),
            ("id_B", "f", "f"),
            ("id_B", "g", "f"),
        ),
    )


@pytest.mark.parametrize("max_degree", [0, 1, 2])
def test_law_violating_category_is_rejected_as_a_domain_error(
    max_degree: int,
) -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        nerve_prefix(_left_identity_violation(), max_degree)
    assert error.value.errors()[0]["type"] == "finite_category.left_identity_law"


def _discrete_category(count: int) -> FiniteCategory:
    labels = tuple(f"o{i}" for i in range(count))
    return FiniteCategory(
        objects=labels,
        morphisms=tuple(
            MorphismSpec(morphism_id=f"id_{obj}", source=obj, target=obj)
            for obj in labels
        ),
        identities=tuple((obj, f"id_{obj}") for obj in labels),
        composition=tuple((f"id_{obj}", f"id_{obj}", f"id_{obj}") for obj in labels),
    )


def test_degree_zero_nerve_admits_the_per_degree_simplex_bound() -> None:
    # N_0 of a 33-object discrete category is 33 vertices, above the
    # 32-simplex per-degree bound; admission must reject before expansion.
    oversized = _discrete_category(33)
    with pytest.raises(
        OperationResourceAdmissionError, match="nerve degree 0 has 33 simplices"
    ):
        nerve_prefix(oversized, 0)
    # The one-below boundary stays accepted with its exact hand-counted nerve.
    admitted = nerve_prefix(_discrete_category(32), 0)
    assert tuple(map(len, admitted.simplicial_set.sets)) == (32,)


def test_published_nerve_example_runs_through_the_owner_tool() -> None:
    tool = next(
        t for t in TOOLS if t.operation_id == "category.finite.nerve_prefix.compute"
    )
    result = tool.run(
        tool.request_type.model_validate(
            {
                "category": {
                    "objects": ["A", "B"],
                    "morphisms": [
                        {"morphism_id": "id_A", "source": "A", "target": "A"},
                        {"morphism_id": "id_B", "source": "B", "target": "B"},
                        {"morphism_id": "f", "source": "A", "target": "B"},
                    ],
                    "identities": [["A", "id_A"], ["B", "id_B"]],
                    "composition": [
                        ["id_A", "id_A", "id_A"],
                        ["f", "id_A", "f"],
                        ["id_B", "id_B", "id_B"],
                        ["id_B", "f", "f"],
                    ],
                },
                "max_degree": 2,
            }
        )
    )
    assert result.simplicial_set.sets[0] == ("0:0", "0:1")
    assert ("f",) in result.simplex_morphisms[1]
