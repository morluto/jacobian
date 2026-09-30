"""Ordering regressions for the cellular-sheaf morphism envelope."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    FiniteCellularSheaf,
    SheafField,
    SheafMorphismResult,
    SheafStalk,
    from_cover_maps,
    morphism,
    morphism_cokernel,
)
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
)
from jacobian.math.topology.cellular_sheaves.morphism_cokernel import (
    SheafMorphismCokernelResult,
    cokernel_of_morphism,
)


def _q(value: str | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _triangle_sheaf(rank: int = 2) -> FiniteCellularSheaf:
    complex_ = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
    faces = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    covers = tuple(
        (face, coface)
        for coface in faces
        for face in faces
        if len(coface) == len(face) + 1 and set(face) < set(coface)
    )
    result = from_cover_maps(
        complex_,
        SheafField.RATIONAL,
        None,
        tuple(
            SheafStalk(simplex=face, basis=tuple(f"x{i}" for i in range(rank)))
            for face in faces
        ),
        tuple(
            CoverRestrictionMatrix(
                source=face,
                target=coface,
                entries=((_q(1),) * rank,) * rank,
            )
            for face, coface in covers
        ),
    )
    assert result.sheaf is not None
    return result.sheaf


def _identity_morphism() -> SheafMorphismResult:
    sheaf = _triangle_sheaf()
    axis = sheaf.canonical_face_order
    components = tuple(
        (
            cell,
            tuple(
                tuple(_q(1) if row == column else _q(0) for column in range(2))
                for row in range(2)
            ),
        )
        for cell in axis
    )
    return morphism(sheaf, sheaf, components)


def _spy_on_parent_reconstruction(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every parent reconstruction the cokernel performs."""
    reconstructions: list[str] = []
    original = morphism_cokernel.__dict__["_readmit_parent_sheaf"]

    def recording(sheaf: FiniteCellularSheaf, *, role: str) -> None:
        reconstructions.append(role)
        original(sheaf, role=role)

    monkeypatch.setattr(morphism_cokernel, "_readmit_parent_sheaf", recording)
    return reconstructions


def test_cokernel_admits_its_envelope_before_reconstructing_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused request must not pay for either parent reconstruction.

    Reconstruction expands the exact functor diagrams, so it belongs inside
    the same work and output envelope as the quotient construction. The
    kernel and image paths already admit the combined bound first; the
    cokernel must too.
    """
    reconstructions = _spy_on_parent_reconstruction(monkeypatch)
    monkeypatch.setattr(morphism_cokernel, "MAX_SHEAF_MORPHISM_OUTPUT_DIGIT_WORK", 0)

    with pytest.raises(OperationResourceAdmissionError) as error:
        cokernel_of_morphism(_identity_morphism())

    assert (
        error.value.errors()[0]["type"]
        == "topology.cellular_sheaf.morphism_cokernel.output_bound"
    )
    assert reconstructions == []


def test_cokernel_admits_reconstruction_work_before_expanding_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The combined work bound is also decided before reconstruction."""
    reconstructions = _spy_on_parent_reconstruction(monkeypatch)
    monkeypatch.setattr(morphism_cokernel, "MAX_SHEAF_MORPHISM_WORK", 0)

    with pytest.raises(OperationResourceAdmissionError) as error:
        cokernel_of_morphism(_identity_morphism())

    assert (
        error.value.errors()[0]["type"]
        == "topology.cellular_sheaf.morphism_cokernel.work_bound"
    )
    assert reconstructions == []


def test_cokernel_still_computes_its_quotient_within_the_envelope() -> None:
    """Moving the admission earlier must not change an admitted result."""
    result = cokernel_of_morphism(_identity_morphism())

    assert result.cokernel.coefficient_field is SheafField.RATIONAL
    assert result.cokernel.complex == _triangle_sheaf().complex
    assert result.projection.components
    assert all(not stalk.basis for stalk in result.cokernel.stalks)
    restored = SheafMorphismCokernelResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert cokernel_of_morphism(restored.projection).cokernel == result.cokernel


@pytest.mark.parametrize("role", ("source", "target"))
@pytest.mark.parametrize("container", ("cover_restrictions", "derived_restrictions"))
def test_cokernel_validates_restriction_axes_before_pricing_the_diagram(
    monkeypatch: pytest.MonkeyPatch, role: str, container: str
) -> None:
    morphism_value = _identity_morphism()
    parent = getattr(morphism_value, role)
    restrictions = getattr(parent, container)
    forged_restriction = restrictions[0].model_copy(update={"target": ("unknown",)})
    forged_parent = parent.model_copy(
        update={container: (forged_restriction, *restrictions[1:])}
    )
    reconstructions = _spy_on_parent_reconstruction(monkeypatch)

    with pytest.raises(OperationDomainValidationError) as error:
        cokernel_of_morphism(morphism_value.model_copy(update={role: forged_parent}))

    assert error.value.errors()[0]["type"].endswith("parent_structure")
    assert reconstructions == []


@pytest.mark.parametrize("role", ("source", "target"))
def test_cokernel_bounds_parent_shape_before_structural_revalidation(
    monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    morphism_value = _identity_morphism()
    parent = getattr(morphism_value, role)
    forged_parent = parent.model_copy(
        update={"derived_restrictions": parent.derived_restrictions * 1000}
    )
    reconstructions = _spy_on_parent_reconstruction(monkeypatch)

    with pytest.raises(OperationResourceAdmissionError) as error:
        cokernel_of_morphism(morphism_value.model_copy(update={role: forged_parent}))

    assert error.value.errors()[0]["type"] == (
        "topology.cellular_sheaf.morphism_cokernel.parent_shape"
    )
    assert reconstructions == []


@pytest.mark.parametrize("axis", ("components", "component", "matrix", "row"))
def test_cokernel_rejects_component_container_subclasses_before_traversal(
    axis: str,
) -> None:
    class UnvisitedTuple(tuple[object, ...]):
        def __len__(self) -> int:
            raise AssertionError("admission must not inspect subclass length")

        def __iter__(self) -> Iterator[object]:
            raise AssertionError("admission must not invoke subclass iterator")

    value = _identity_morphism()
    cell, matrix = value.components[0]
    components: tuple[object, ...]
    if axis == "components":
        components = UnvisitedTuple(value.components)
    elif axis == "component":
        components = (UnvisitedTuple((cell, matrix)), *value.components[1:])
    elif axis == "matrix":
        components = ((cell, UnvisitedTuple(matrix)), *value.components[1:])
    else:
        components = (
            (cell, (UnvisitedTuple(matrix[0]), *matrix[1:])),
            *value.components[1:],
        )
    with pytest.raises(OperationDomainValidationError) as error:
        cokernel_of_morphism(value.model_copy(update={"components": components}))
    assert error.value.errors()[0]["type"].endswith("component_structure")


@pytest.mark.parametrize("role", ("source", "target"))
@pytest.mark.parametrize(
    "container", ("stalks", "cover_restrictions", "derived_restrictions")
)
def test_cokernel_parent_shape_errors_keep_the_cokernel_owner(
    role: str, container: str
) -> None:
    value = _identity_morphism()
    parent = getattr(value, role).model_copy(update={container: None})
    with pytest.raises(OperationDomainValidationError) as error:
        cokernel_of_morphism(value.model_copy(update={role: parent}))
    assert type(error.value) is OperationDomainValidationError
    assert error.value.errors()[0]["type"] == (
        "topology.cellular_sheaf.morphism_cokernel.parent_structure"
    )
