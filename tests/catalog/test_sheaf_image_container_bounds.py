"""The public image operation admits the canonical five-simplex sheaf."""

import json

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import morphism
from jacobian.math.topology.cellular_sheaves.constants.operations import constant_sheaf
from jacobian.math.topology.cellular_sheaves.morphism_image import (
    SheafMorphismImageResult,
)


def test_public_image_admits_five_simplex_identity_and_reuses_its_result() -> None:
    vertices = ("a", "b", "c", "d", "e", "f")
    sheaf = constant_sheaf(canonical_complex(vertices, (vertices,)))
    identity = morphism(
        sheaf,
        sheaf,
        tuple(
            (face, ((CanonicalRational(num=1, den=1),),))
            for face in sheaf.canonical_face_order
        ),
    )
    catalog = Catalog.open()
    operation = "cellular_sheaf.morphism.image.compute"
    result = invoke_operation(
        operation, {"morphism": identity.model_dump(mode="json")}, catalog
    ).output
    decoded = SheafMorphismImageResult.model_validate_json(json.dumps(result))
    assert len(decoded.image.cover_restrictions) == 186
    assert len(decoded.image.derived_restrictions) == 416
    again = invoke_operation(
        operation, {"morphism": decoded.inclusion.model_dump(mode="json")}, catalog
    ).output
    assert again["image"] == result["image"]
