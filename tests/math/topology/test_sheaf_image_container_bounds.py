"""Image admission uses each canonical restriction container's own envelope."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
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
