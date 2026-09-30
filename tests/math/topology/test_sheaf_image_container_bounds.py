"""Image admission uses each canonical restriction container's own envelope."""

from collections.abc import Iterator

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    FiniteCellularSheaf,
    SheafMorphismResult,
    morphism,
)
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_COVER_MAPS,
    MAX_SHEAF_DERIVED_RESTRICTIONS,
)
from jacobian.math.topology.cellular_sheaves.constants.operations import constant_sheaf
from jacobian.math.topology.cellular_sheaves.morphism_image import (
    SheafMorphismImageResult,
    image_of_morphism,
)


def _identity(sheaf: FiniteCellularSheaf) -> SheafMorphismResult:
    return morphism(
        sheaf,
        sheaf,
        tuple(
            (cell, ((CanonicalRational(num=1, den=1),),))
            for cell in sheaf.canonical_face_order
        ),
    )


def test_five_simplex_identity_image_round_trips_and_is_consumed() -> None:
    vertices = ("a", "b", "c", "d", "e", "f")
    sheaf = constant_sheaf(canonical_complex(vertices, (vertices,)))
    assert len(sheaf.cover_restrictions) == 186
    assert len(sheaf.derived_restrictions) == 416
    result = image_of_morphism(_identity(sheaf))
    assert (
        tuple(stalk.simplex for stalk in result.image.stalks)
        == sheaf.canonical_face_order
    )
    assert all(len(stalk.basis) == 1 for stalk in result.image.stalks)
    for kind in ("cover_restrictions", "derived_restrictions"):
        assert tuple(
            (row.source, row.target, row.entries) for row in getattr(result.image, kind)
        ) == tuple(
            (row.source, row.target, row.entries) for row in getattr(sheaf, kind)
        )
    decoded = SheafMorphismImageResult.model_validate_json(result.model_dump_json())
    assert image_of_morphism(decoded.inclusion).image == result.image


@pytest.mark.parametrize(
    "attribute, limit",
    (
        ("cover_restrictions", MAX_SHEAF_COVER_MAPS),
        ("derived_restrictions", MAX_SHEAF_DERIVED_RESTRICTIONS),
    ),
)
def test_restriction_container_over_its_own_bound_is_refused(
    attribute: str, limit: int
) -> None:
    vertices = ("a", "b", "c")
    sheaf = constant_sheaf(canonical_complex(vertices, (vertices,)))
    clean = _identity(sheaf)
    malformed = sheaf.model_copy(
        update={attribute: (getattr(sheaf, attribute)[0],) * (limit + 1)}
    )
    forged = clean.model_copy(update={"source": malformed})
    with pytest.raises(
        OperationResourceAdmissionError, match=f"too many {attribute} rows"
    ):
        image_of_morphism(forged)


@pytest.mark.parametrize("container", ("cover_restrictions", "derived_restrictions"))
@pytest.mark.parametrize("role", ("source", "target"))
@pytest.mark.parametrize("field", ("row_basis", "column_basis", "entries", "row"))
def test_nested_restriction_containers_are_bounded_before_parent_copy(
    container: str, role: str, field: str
) -> None:
    vertices = ("a", "b", "c")
    sheaf = constant_sheaf(canonical_complex(vertices, (vertices,)))
    restriction = getattr(sheaf, container)[0]
    update: dict[str, object]
    if field == "row":
        update = {"entries": ((CanonicalRational(num=1, den=1),) * 9,)}
    elif field == "entries":
        update = {field: restriction.entries * 9}
    else:
        update = {field: ("e0",) * 9}
    malformed = sheaf.model_copy(
        update={container: (restriction.model_copy(update=update),)}
    )
    forged = _identity(sheaf).model_copy(update={role: malformed})
    with pytest.raises(OperationResourceAdmissionError, match=r"restriction.*envelope"):
        image_of_morphism(forged)


def test_deeply_nested_restriction_scalar_is_rejected_without_recursive_copy() -> None:
    vertices = ("a", "b", "c")
    sheaf = constant_sheaf(canonical_complex(vertices, (vertices,)))
    nested: object = 1
    for _ in range(1500):
        nested = (nested,)
    restriction = sheaf.cover_restrictions[0].model_copy(
        update={"entries": ((nested,),)}
    )
    malformed = sheaf.model_copy(update={"cover_restrictions": (restriction,)})
    forged = _identity(sheaf).model_copy(update={"source": malformed})
    with pytest.raises(OperationDomainValidationError, match="scalar"):
        image_of_morphism(forged)


@pytest.mark.parametrize("rank", (0, 8))
def test_image_preserves_zero_rank_and_full_rank_restriction_matrices(
    rank: int,
) -> None:
    vertices = ("a", "b")
    sheaf = constant_sheaf(
        canonical_complex(vertices, (vertices,)),
        basis=tuple(f"e{i}" for i in range(rank)),
    )
    matrix = tuple(
        tuple(
            CanonicalRational(num=int(row == column), den=1) for column in range(rank)
        )
        for row in range(rank)
    )
    identity = morphism(
        sheaf, sheaf, tuple((face, matrix) for face in sheaf.canonical_face_order)
    )
    result = image_of_morphism(identity)
    assert all(len(stalk.basis) == rank for stalk in result.image.stalks)
    assert all(
        restriction.entries == matrix for restriction in result.image.cover_restrictions
    )
    decoded = SheafMorphismImageResult.model_validate_json(result.model_dump_json())
    assert image_of_morphism(decoded.inclusion).image == result.image


@pytest.mark.parametrize(
    "field", ("stalks", "cover_restrictions", "derived_restrictions")
)
@pytest.mark.parametrize("value", (None, 1, "invalid", {"nested": [1] * 1000}))
def test_malformed_parent_container_kinds_are_refused_before_copy(
    field: str, value: object
) -> None:
    vertices = ("a", "b", "c")
    sheaf = constant_sheaf(canonical_complex(vertices, (vertices,)))
    malformed = sheaf.model_copy(update={field: value})
    forged = _identity(sheaf).model_copy(update={"source": malformed})
    with pytest.raises(OperationDomainValidationError, match="ordered container"):
        image_of_morphism(forged)


@pytest.mark.parametrize(
    "field, value",
    (
        ("prime", {}),
        ("diamonds", []),
        ("comparable_pairs", ()),
        ("coefficient_field", {}),
        ("complex", None),
    ),
)
def test_parent_scalar_and_complex_fields_reject_nested_containers(
    field: str, value: object
) -> None:
    sheaf = constant_sheaf(canonical_complex(("a",), (("a",),)))
    malformed = sheaf.model_copy(update={field: value})
    with pytest.raises(OperationDomainValidationError):
        image_of_morphism(_identity(sheaf).model_copy(update={"source": malformed}))


def test_parent_models_reject_undeclared_nested_fields_before_copy() -> None:
    sheaf = constant_sheaf(canonical_complex(("a",), (("a",),)))
    nested: object = 1
    for _ in range(1500):
        nested = (nested,)
    malformed = sheaf.model_copy(update={"undeclared": nested})
    with pytest.raises(OperationDomainValidationError, match="structurally valid"):
        image_of_morphism(_identity(sheaf).model_copy(update={"source": malformed}))


def test_empty_stalk_images_admit_the_full_simplex_count_envelope() -> None:
    vertices = tuple(f"v{i:02}" for i in range(64))
    sheaf = constant_sheaf(
        canonical_complex(vertices, tuple((vertex,) for vertex in vertices)), basis=()
    )
    identity = morphism(
        sheaf, sheaf, tuple((face, ()) for face in sheaf.canonical_face_order)
    )
    result = image_of_morphism(identity)
    assert len(result.image.stalks) == 64
    assert all(not stalk.basis for stalk in result.image.stalks)
    decoded = SheafMorphismImageResult.model_validate_json(result.model_dump_json())
    assert image_of_morphism(decoded.inclusion).image == result.image


@pytest.mark.parametrize("role", ("source", "target"))
def test_parent_sheaf_subclasses_are_rejected_before_copy(role: str) -> None:
    class ExtendedSheaf(FiniteCellularSheaf):
        payload: object = None

    sheaf = constant_sheaf(canonical_complex(("a",), (("a",),)))
    nested: object = 1
    for _ in range(1500):
        nested = (nested,)
    parent = ExtendedSheaf.model_construct(**sheaf.__dict__, payload=nested)
    forged = _identity(sheaf).model_copy(update={role: parent})
    with pytest.raises(
        OperationDomainValidationError, match="canonical finite cellular sheaves"
    ):
        image_of_morphism(forged)


class _UnvisitedList(list[object]):
    def __len__(self) -> int:
        raise AssertionError("a noncanonical container length was inspected")

    def __iter__(self) -> Iterator[object]:
        raise AssertionError("a noncanonical container was traversed")


@pytest.mark.parametrize(
    "field", ("stalks", "cover_restrictions", "derived_restrictions")
)
def test_parent_container_subclasses_are_refused_without_inspection(field: str) -> None:
    sheaf = constant_sheaf(canonical_complex(("a", "b", "c"), (("a", "b", "c"),)))
    parent = sheaf.model_copy(update={field: _UnvisitedList()})
    with pytest.raises(OperationDomainValidationError, match="ordered container"):
        image_of_morphism(_identity(sheaf).model_copy(update={"source": parent}))


@pytest.mark.parametrize(
    "field", ("row_basis", "column_basis", "entries", "row", "cover_path")
)
def test_restriction_container_subclasses_are_refused_without_inspection(
    field: str,
) -> None:
    sheaf = constant_sheaf(canonical_complex(("a", "b"), (("a", "b"),)))
    update = (
        {"entries": (_UnvisitedList(),)}
        if field == "row"
        else {field: _UnvisitedList()}
    )
    restriction = sheaf.cover_restrictions[0].model_copy(update=update)
    parent = sheaf.model_copy(update={"cover_restrictions": (restriction,)})
    with pytest.raises(OperationDomainValidationError, match="ordered container"):
        image_of_morphism(_identity(sheaf).model_copy(update={"target": parent}))


def test_stalk_basis_subclasses_are_refused_without_inspection() -> None:
    sheaf = constant_sheaf(canonical_complex(("a",), (("a",),)))
    stalk = sheaf.stalks[0].model_copy(update={"basis": _UnvisitedList()})
    parent = sheaf.model_copy(update={"stalks": (stalk,)})
    with pytest.raises(OperationDomainValidationError, match="ordered container"):
        image_of_morphism(_identity(sheaf).model_copy(update={"source": parent}))
