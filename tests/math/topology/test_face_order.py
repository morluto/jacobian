"""Canonical face ordering retains strict parsing and stable validation codes."""

import json

import pytest
from pydantic import ValidationError

from jacobian.math.topology._models import FacesInDimension


@pytest.mark.parametrize("wire", (False, True), ids=("native", "json"))
@pytest.mark.parametrize(
    ("faces", "code"),
    (
        ((("a",),), "topology.require_canonical_faces_1"),
        ((("b", "a"),), "topology.require_canonical_faces_1"),
        ((("a", "c"), ("a", "b")), "topology.require_canonical_faces_2"),
        ((("a", "b"), ("a", "b")), "topology.require_canonical_faces_2"),
    ),
    ids=("wrong-dimension", "vertex-order", "face-order", "duplicate-face"),
)
def test_noncanonical_faces_keep_their_validation_codes(
    faces: tuple[tuple[str, ...], ...], code: str, wire: bool
) -> None:
    with pytest.raises(ValidationError) as error:
        if wire:
            FacesInDimension.model_validate_json(
                json.dumps({"dimension": 1, "faces": faces}), strict=True
            )
        else:
            FacesInDimension(dimension=1, faces=faces)
    assert [item["type"] for item in error.value.errors()] == [code]


def test_canonical_multiple_faces_round_trip_without_reordering() -> None:
    faces = FacesInDimension(dimension=1, faces=(("a", "b"), ("a", "c"), ("b", "c")))
    assert (
        FacesInDimension.model_validate_json(faces.model_dump_json(), strict=True)
        == faces
    )
