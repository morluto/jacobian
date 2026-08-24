"""Tests for exact rational V-polytope support and exposed faces."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.polytope import (
    PolytopeSupportResult,
    RationalCoordinateSpace,
    RationalCovector,
    RationalExposedFace,
    RationalPolytopeVertex,
    RationalVPolytope,
    polytope_support,
)
from jacobian.math.polytope import _operations as polytope_operations
from jacobian.math.polytope._models import (
    MAX_SUPPORT_COMPONENT_DIGITS,
    PolytopeSupportRequest,
)
from jacobian.math.polytope._operations import compute_polytope_support
from jacobian.math.polytope._tools import POLYTOPE_OPERATIONS


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational(num=str(value), den="1")


def _vertex(vertex_id: str, *coordinates: int) -> RationalPolytopeVertex:
    return RationalPolytopeVertex(
        vertex_id=vertex_id,
        coordinates=tuple(_rational(value) for value in coordinates),
    )


def _square() -> RationalVPolytope:
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=(
            _vertex("bottom_left", 0, 0),
            _vertex("bottom_right", 1, 0),
            _vertex("top_left", 0, 1),
            _vertex("top_right", 1, 1),
        ),
    )


def _covector(*components: int) -> RationalCovector:
    return RationalCovector(
        space=RationalCoordinateSpace(axes=("x", "y")),
        components=tuple(_rational(component) for component in components),
    )


def _large_rational(digits: int) -> CanonicalRational:
    numerator = "9" * digits
    return CanonicalRational(num=numerator, den="8")


def _oversized_simplex_payload() -> dict:
    """A six-dimensional simplex whose first axis carries 151 digits."""
    axes = ["x1", "x2", "x3", "x4", "x5", "x6"]
    zero = {"num": "0", "den": "1"}
    one = {"num": "1", "den": "1"}
    oversized = _large_rational(MAX_SUPPORT_COMPONENT_DIGITS + 1).model_dump()
    vertices = [{"vertex_id": "origin", "coordinates": [zero] * 6}]
    vertices.extend(
        {
            "vertex_id": f"e{axis}",
            "coordinates": [
                one if coordinate == axis else zero for coordinate in range(6)
            ],
        }
        for axis in range(6)
    )
    vertices[1]["coordinates"][0] = oversized
    return {
        "polytope": {"space": {"axes": axes}, "vertices": vertices},
        "covector": {"space": {"axes": axes}, "components": [one] * 6},
    }


def _square_payload() -> dict:
    """A valid serialized square support request payload."""
    zero = {"num": "0", "den": "1"}
    one = {"num": "1", "den": "1"}
    vertices = [
        {"vertex_id": name, "coordinates": [a, b]}
        for name, a, b in (
            ("bottom_left", zero, zero),
            ("bottom_right", one, zero),
            ("top_left", zero, one),
            ("top_right", one, one),
        )
    ]
    return {
        "polytope": {"space": {"axes": ["x", "y"]}, "vertices": vertices},
        "covector": {"space": {"axes": ["x", "y"]}, "components": [one, zero]},
    }


def _forbid_extremality_proof(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the exact extremality proof is ever reached."""

    def unexpected_proof(polytope: object) -> None:
        raise AssertionError("extremality proof ran before the covector preflight")

    monkeypatch.setattr(
        polytope_operations,
        "require_full_dimensional_extreme_vertices",
        unexpected_proof,
    )


@pytest.fixture()
def square_result() -> PolytopeSupportResult:
    return compute_polytope_support(
        PolytopeSupportRequest(polytope=_square(), covector=_covector(0, 1))
    )


def test_support_returns_exact_value_and_complete_exposed_edge() -> None:
    result = compute_polytope_support(
        PolytopeSupportRequest(polytope=_square(), covector=_covector(0, 1))
    )

    assert result.support_value.as_fraction() == 1
    assert tuple(vertex.vertex_id for vertex in result.exposed_face.vertices) == (
        "top_left",
        "top_right",
    )
    assert result.exposed_face.space == result.polytope.space


def test_zero_covector_exposes_the_whole_polytope() -> None:
    polytope = _square()
    result = polytope_support(polytope, _covector(0, 0))

    assert result.support_value.as_fraction() == 0
    assert result.exposed_face.vertices == polytope.vertices


def test_positive_covector_scaling_preserves_face_and_scales_support() -> None:
    polytope = _square()
    unit = polytope_support(polytope, _covector(0, 1))
    scaled = polytope_support(polytope, _covector(0, 3))

    assert scaled.support_value.as_fraction() == 3 * unit.support_value.as_fraction()
    assert scaled.exposed_face == unit.exposed_face


def test_support_is_source_bound_against_forged_value_or_face() -> None:
    result = compute_polytope_support(
        PolytopeSupportRequest(polytope=_square(), covector=_covector(0, 1))
    )

    with pytest.raises(ValidationError, match="support value must equal"):
        PolytopeSupportResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "support_value": {"num": "2", "den": "1"},
            }
        )
    with pytest.raises(ValidationError, match="exposed face must be exactly"):
        PolytopeSupportResult(
            polytope=result.polytope,
            covector=result.covector,
            support_value=result.support_value,
            exposed_face=RationalExposedFace(
                space=result.polytope.space,
                vertices=(result.polytope.vertices[0],),
            ),
        )
    with pytest.raises(ValidationError, match="exposed face must be exactly"):
        PolytopeSupportResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "covector": {
                    "space": {"axes": ["x", "y"]},
                    "components": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
            }
        )


def test_request_rejects_coordinate_axis_mismatch() -> None:
    bad_covector = RationalCovector(
        space=RationalCoordinateSpace(axes=("y", "x")),
        components=(_rational(0), _rational(1)),
    )

    with pytest.raises(ValidationError, match="same coordinate space"):
        PolytopeSupportRequest(polytope=_square(), covector=bad_covector)


def test_v_polytope_rejects_nonextreme_vertex() -> None:
    with pytest.raises(ValidationError, match="exact extreme vertices"):
        RationalVPolytope(
            space=RationalCoordinateSpace(axes=("x", "y")),
            vertices=(
                _vertex("bottom_left", 0, 0),
                _vertex("bottom_right", 2, 0),
                _vertex("middle", 1, 0),
                _vertex("top_left", 0, 2),
                _vertex("top_right", 2, 2),
            ),
        )


def test_v_polytope_rejects_lower_dimensional_hull() -> None:
    with pytest.raises(ValidationError, match="affinely span"):
        RationalVPolytope(
            space=RationalCoordinateSpace(axes=("x", "y")),
            vertices=(
                _vertex("left", 0, 0),
                _vertex("middle", 1, 0),
                _vertex("right", 2, 0),
            ),
        )


def test_support_request_rejects_over_envelope_covector_component() -> None:
    over_bound = _large_rational(MAX_SUPPORT_COMPONENT_DIGITS + 1)
    covector = RationalCovector(
        space=RationalCoordinateSpace(axes=("x", "y")),
        components=(over_bound, _rational(0)),
    )

    with pytest.raises(
        ValidationError,
        match=f"covector component exceeds the {MAX_SUPPORT_COMPONENT_DIGITS}-digit bound",
    ):
        PolytopeSupportRequest(polytope=_square(), covector=covector)


def test_support_request_rejects_over_envelope_vertex_coordinates() -> None:
    over_bound = _large_rational(MAX_SUPPORT_COMPONENT_DIGITS + 1)
    polytope = RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=(
            RationalPolytopeVertex(
                vertex_id="a",
                coordinates=(_rational(0), _rational(0)),
            ),
            RationalPolytopeVertex(
                vertex_id="b",
                coordinates=(over_bound, _rational(0)),
            ),
            RationalPolytopeVertex(
                vertex_id="c",
                coordinates=(_rational(0), _rational(1)),
            ),
        ),
    )

    with pytest.raises(
        ValidationError,
        match=f"polytope vertex coordinate exceeds the {MAX_SUPPORT_COMPONENT_DIGITS}-digit bound",
    ):
        PolytopeSupportRequest(polytope=polytope, covector=_covector(1, 0))


def test_request_preflights_oversized_coordinates_before_hull_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raw payload with canonical-but-over-envelope coordinates is
    rejected by the pre-parsing envelope check; the exact extremality
    proof inside nested V-polytope construction is never reached."""

    def unexpected_proof(polytope: object) -> None:
        raise AssertionError("extremality proof ran before the envelope preflight")

    monkeypatch.setattr(
        polytope_operations,
        "require_full_dimensional_extreme_vertices",
        unexpected_proof,
    )

    with pytest.raises(
        ValidationError,
        match=f"polytope vertex coordinate exceeds the {MAX_SUPPORT_COMPONENT_DIGITS}-digit bound",
    ):
        PolytopeSupportRequest.model_validate(_oversized_simplex_payload())


def test_request_preflights_oversized_covector_payload_before_nested_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The covector half of a raw payload is preflighted the same way."""

    def unexpected_proof(polytope: object) -> None:
        raise AssertionError("extremality proof ran before the envelope preflight")

    monkeypatch.setattr(
        polytope_operations,
        "require_full_dimensional_extreme_vertices",
        unexpected_proof,
    )
    payload = {
        "polytope": {
            "space": {"axes": ["x", "y"]},
            "vertices": [
                {
                    "vertex_id": name,
                    "coordinates": [
                        {"num": str(a), "den": "1"},
                        {"num": str(b), "den": "1"},
                    ],
                }
                for name, a, b in (
                    ("bottom_left", 0, 0),
                    ("bottom_right", 1, 0),
                    ("top_left", 0, 1),
                    ("top_right", 1, 1),
                )
            ],
        },
        "covector": {
            "space": {"axes": ["x", "y"]},
            "components": [
                _large_rational(MAX_SUPPORT_COMPONENT_DIGITS + 1).model_dump(),
                {"num": "0", "den": "1"},
            ],
        },
    }

    with pytest.raises(
        ValidationError,
        match=f"covector component exceeds the {MAX_SUPPORT_COMPONENT_DIGITS}-digit bound",
    ):
        PolytopeSupportRequest.model_validate(payload)


def test_request_rejects_forbidden_extra_field_before_hull_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forbidden outer field is rejected before the valid V-polytope
    pays its exact extremality proof."""

    _forbid_extremality_proof(monkeypatch)
    payload = _square_payload()
    payload["junk"] = 1

    with pytest.raises(
        ValidationError,
        match="unexpected fields for a polytope support request",
    ):
        PolytopeSupportRequest.model_validate(payload)


def test_request_preflights_missing_covector_before_hull_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raw payload omitting the covector is rejected before the valid
    V-polytope pays its exact extremality proof."""

    _forbid_extremality_proof(monkeypatch)
    payload = _square_payload()
    del payload["covector"]

    with pytest.raises(ValidationError, match="covector must be provided"):
        PolytopeSupportRequest.model_validate(payload)


def test_request_preflights_malformed_covector_components_before_hull_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Component shapes that cannot construct a canonical rational are
    rejected before the hull proof instead of after it."""

    _forbid_extremality_proof(monkeypatch)
    payload = _square_payload()
    payload["covector"]["components"] = [
        {"num": "1", "den": "1"},
        {"num": "0"},
    ]

    with pytest.raises(
        ValidationError,
        match="covector component must be a canonical rational",
    ):
        PolytopeSupportRequest.model_validate(payload)

    payload = _square_payload()
    payload["covector"]["components"] = [{"num": "1", "den": "1"}, "0"]

    with pytest.raises(
        ValidationError,
        match="covector component must be a canonical rational",
    ):
        PolytopeSupportRequest.model_validate(payload)


def test_request_preflights_noncanonical_covector_strings_before_hull_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact-key string-shaped components must also parse as canonical
    rationals: a payload retaining a valid near-limit V-polytope is
    rejected before the hull proof instead of paying up to the full
    orientation-test bound for a covector nested validation rejects."""

    _forbid_extremality_proof(monkeypatch)
    for malformed in (
        {"num": "invalid", "den": "1"},
        {"num": "1.5", "den": "1"},
        {"num": "01", "den": "1"},
        {"num": "", "den": "1"},
        {"num": "1", "den": "0"},
        {"num": "1", "den": "-2"},
        {"num": "2", "den": "4"},
        {"num": "0", "den": "2"},
    ):
        payload = _square_payload()
        payload["covector"]["components"] = [
            malformed,
            {"num": "0", "den": "1"},
        ]

        with pytest.raises(
            ValidationError,
            match="covector component must be a canonical rational",
        ):
            PolytopeSupportRequest.model_validate(payload)


def test_request_preflights_covector_dimension_mismatch_before_hull_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A covector whose components do not match its declared axis count
    is rejected before the hull proof."""

    _forbid_extremality_proof(monkeypatch)
    payload = _square_payload()
    payload["covector"]["components"] = [{"num": "1", "den": "1"}]

    with pytest.raises(
        ValidationError,
        match="covector components must use the declared coordinate axis",
    ):
        PolytopeSupportRequest.model_validate(payload)


def test_request_preflights_foreign_covector_space_before_hull_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A covector declaring a different raw space is rejected before the
    hull proof."""

    _forbid_extremality_proof(monkeypatch)
    payload = _square_payload()
    payload["covector"]["space"] = {"axes": ["x", "z"]}

    with pytest.raises(
        ValidationError,
        match="polytope and covector must use the same coordinate space",
    ):
        PolytopeSupportRequest.model_validate(payload)


def test_request_preflights_malformed_covector_space_before_hull_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A covector space violating a published constraint is rejected
    before the hull proof."""

    _forbid_extremality_proof(monkeypatch)
    payload = _square_payload()
    payload["covector"]["space"] = {"axes": []}

    with pytest.raises(
        ValidationError,
        match="covector space axes must be a non-empty sequence",
    ):
        PolytopeSupportRequest.model_validate(payload)

    payload = _square_payload()
    payload["covector"]["space"] = {"axes": ["x", "x"]}

    with pytest.raises(ValidationError, match="coordinate axes must be unique"):
        PolytopeSupportRequest.model_validate(payload)


def test_serialized_request_accepts_components_at_the_envelope_boundary() -> None:
    """The serialized math.run path admits exactly-150-digit components."""
    at_bound = _large_rational(MAX_SUPPORT_COMPONENT_DIGITS)
    polytope = RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=(
            RationalPolytopeVertex(
                vertex_id="a",
                coordinates=(_rational(0), _rational(0)),
            ),
            RationalPolytopeVertex(
                vertex_id="b",
                coordinates=(at_bound, _rational(0)),
            ),
            RationalPolytopeVertex(
                vertex_id="c",
                coordinates=(_rational(0), _rational(1)),
            ),
        ),
    )
    request = PolytopeSupportRequest.model_validate(
        {
            "polytope": polytope.model_dump(mode="json"),
            "covector": _covector(1, 0).model_dump(mode="json"),
        }
    )

    result = compute_polytope_support(request)

    assert result.support_value == at_bound
    assert tuple(vertex.vertex_id for vertex in result.exposed_face.vertices) == ("b",)


def test_native_support_rejects_over_envelope_components() -> None:
    over_bound = _large_rational(MAX_SUPPORT_COMPONENT_DIGITS + 1)
    covector = RationalCovector(
        space=RationalCoordinateSpace(axes=("x", "y")),
        components=(over_bound, _rational(0)),
    )

    with pytest.raises(
        ValueError,
        match=f"covector component exceeds the {MAX_SUPPORT_COMPONENT_DIGITS}-digit bound",
    ):
        polytope_support(_square(), covector)


def test_support_accepts_components_at_the_envelope_boundary() -> None:
    at_bound = _large_rational(MAX_SUPPORT_COMPONENT_DIGITS)
    polytope = RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=(
            RationalPolytopeVertex(
                vertex_id="a",
                coordinates=(_rational(0), _rational(0)),
            ),
            RationalPolytopeVertex(
                vertex_id="b",
                coordinates=(at_bound, _rational(0)),
            ),
            RationalPolytopeVertex(
                vertex_id="c",
                coordinates=(_rational(0), _rational(1)),
            ),
        ),
    )

    result = compute_polytope_support(
        PolytopeSupportRequest(polytope=polytope, covector=_covector(1, 0))
    )

    assert result.support_value == at_bound
    assert tuple(vertex.vertex_id for vertex in result.exposed_face.vertices) == ("b",)


def _over_envelope_source_result_payload() -> dict:
    """A self-consistent serialized result whose retained source the
    support request would reject: the triangle carries one 151-digit
    coordinate, the zero covector makes the support value zero and every
    vertex maximizing."""

    oversized = _large_rational(MAX_SUPPORT_COMPONENT_DIGITS + 1).model_dump()
    zero = {"num": "0", "den": "1"}
    one = {"num": "1", "den": "1"}
    vertices = [
        {"vertex_id": "a", "coordinates": [oversized, zero]},
        {"vertex_id": "b", "coordinates": [zero, zero]},
        {"vertex_id": "c", "coordinates": [zero, one]},
    ]
    return {
        "polytope": {"space": {"axes": ["x", "y"]}, "vertices": vertices},
        "covector": {"space": {"axes": ["x", "y"]}, "components": [zero, zero]},
        "support_value": zero,
        "exposed_face": {"space": {"axes": ["x", "y"]}, "vertices": vertices},
    }


def test_result_rejects_unproducible_over_envelope_source() -> None:
    """A serialized result whose internally consistent retained source
    carries a 151-digit coordinate is rejected: no admitted request could
    have produced it."""

    with pytest.raises(
        ValidationError,
        match=(
            f"polytope vertex coordinate exceeds the "
            f"{MAX_SUPPORT_COMPONENT_DIGITS}-digit bound"
        ),
    ):
        PolytopeSupportResult.model_validate(_over_envelope_source_result_payload())


def test_constructed_result_rejects_unadmitted_source() -> None:
    """The typed construction path reapplies the request admission too."""

    over_bound = _large_rational(MAX_SUPPORT_COMPONENT_DIGITS + 1)
    polytope = RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=(
            RationalPolytopeVertex(
                vertex_id="a",
                coordinates=(_rational(0), _rational(0)),
            ),
            RationalPolytopeVertex(
                vertex_id="b",
                coordinates=(over_bound, _rational(0)),
            ),
            RationalPolytopeVertex(
                vertex_id="c",
                coordinates=(_rational(0), _rational(1)),
            ),
        ),
    )

    with pytest.raises(
        ValidationError,
        match=(
            f"polytope vertex coordinate exceeds the "
            f"{MAX_SUPPORT_COMPONENT_DIGITS}-digit bound"
        ),
    ):
        PolytopeSupportResult(
            polytope=polytope,
            covector=_covector(0, 0),
            support_value=_rational(0),
            exposed_face=RationalExposedFace(
                space=polytope.space,
                vertices=polytope.vertices,
            ),
        )


def test_result_preflights_oversized_covector_before_nested_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Result deserialization measures the retained covector before the
    nested V-polytope's exact extremality proof can execute outside the
    operation's envelope."""

    def unexpected_proof(polytope: object) -> None:
        raise AssertionError("extremality proof ran before the envelope preflight")

    oversized = _large_rational(MAX_SUPPORT_COMPONENT_DIGITS + 1).model_dump()
    polytope_payload = _square().model_dump(mode="json")
    monkeypatch.setattr(
        polytope_operations,
        "require_full_dimensional_extreme_vertices",
        unexpected_proof,
    )
    payload = {
        "polytope": polytope_payload,
        "covector": {
            "space": {"axes": ["x", "y"]},
            "components": [oversized, {"num": "0", "den": "1"}],
        },
        "support_value": oversized,
        "exposed_face": {
            "space": {"axes": ["x", "y"]},
            "vertices": [
                vertex
                for vertex in polytope_payload["vertices"]
                if vertex["vertex_id"] in ("bottom_right", "top_right")
            ],
        },
    }

    with pytest.raises(
        ValidationError,
        match=(
            f"covector component exceeds the {MAX_SUPPORT_COMPONENT_DIGITS}-digit bound"
        ),
    ):
        PolytopeSupportResult.model_validate(payload)


def test_result_preflights_missing_covector_before_nested_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Result deserialization gates the retained covector's presence
    before the nested V-polytope's exact extremality proof executes."""

    result = compute_polytope_support(
        PolytopeSupportRequest(polytope=_square(), covector=_covector(0, 1))
    )
    _forbid_extremality_proof(monkeypatch)
    payload = result.model_dump(mode="json")
    del payload["covector"]

    with pytest.raises(ValidationError, match="covector must be provided"):
        PolytopeSupportResult.model_validate(payload)


def test_result_preflights_missing_support_value_before_nested_parsing(
    square_result: PolytopeSupportResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serialized result omitting ``support_value`` is rejected before
    the retained V-polytope pays its exact extremality proof."""

    _forbid_extremality_proof(monkeypatch)
    payload = square_result.model_dump(mode="json")
    del payload["support_value"]

    with pytest.raises(ValidationError, match="support_value must be provided"):
        PolytopeSupportResult.model_validate(payload)


def test_result_preflights_malformed_support_value_before_nested_parsing(
    square_result: PolytopeSupportResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``support_value`` that cannot construct a canonical rational is
    rejected before the hull proof."""

    _forbid_extremality_proof(monkeypatch)
    payload = square_result.model_dump(mode="json")
    payload["support_value"] = {"num": "0"}

    with pytest.raises(
        ValidationError,
        match="support value must be a canonical rational",
    ):
        PolytopeSupportResult.model_validate(payload)

    payload = square_result.model_dump(mode="json")
    payload["support_value"] = {"num": "invalid", "den": "1"}

    with pytest.raises(
        ValidationError,
        match="support value must be a canonical rational",
    ):
        PolytopeSupportResult.model_validate(payload)

    payload = square_result.model_dump(mode="json")
    payload["support_value"] = {"num": "2", "den": "4"}

    with pytest.raises(
        ValidationError,
        match="support value must be a canonical rational",
    ):
        PolytopeSupportResult.model_validate(payload)


def test_result_preflights_missing_exposed_face_before_nested_parsing(
    square_result: PolytopeSupportResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serialized result omitting ``exposed_face`` is rejected before
    the retained V-polytope pays its exact extremality proof."""

    _forbid_extremality_proof(monkeypatch)
    payload = square_result.model_dump(mode="json")
    del payload["exposed_face"]

    with pytest.raises(ValidationError, match="exposed_face must be provided"):
        PolytopeSupportResult.model_validate(payload)


def test_result_preflights_malformed_exposed_face_before_nested_parsing(
    square_result: PolytopeSupportResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exposed face violating a published structural constraint is
    rejected before the hull proof."""

    _forbid_extremality_proof(monkeypatch)
    payload = square_result.model_dump(mode="json")
    payload["exposed_face"] = {"space": {"axes": ["x", "y"]}}

    with pytest.raises(
        ValidationError,
        match="exposed face must be an object with space and vertices",
    ):
        PolytopeSupportResult.model_validate(payload)

    payload = square_result.model_dump(mode="json")
    payload["exposed_face"]["vertices"][0]["vertex_id"] = 7

    with pytest.raises(
        ValidationError,
        match="exposed face vertex must be an object with a short vertex_id",
    ):
        PolytopeSupportResult.model_validate(payload)

    payload = square_result.model_dump(mode="json")
    payload["exposed_face"]["vertices"][0]["coordinates"][0] = {
        "num": "invalid",
        "den": "1",
    }

    with pytest.raises(
        ValidationError,
        match="exposed face vertex coordinate must be a canonical rational",
    ):
        PolytopeSupportResult.model_validate(payload)


def test_result_preflights_broken_face_invariants_before_nested_parsing(
    square_result: PolytopeSupportResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exposed face whose entries violate the defining ordering or
    distinctness invariants is rejected before the hull proof."""

    _forbid_extremality_proof(monkeypatch)
    payload = square_result.model_dump(mode="json")
    payload["exposed_face"]["vertices"].reverse()

    with pytest.raises(
        ValidationError,
        match="exposed-face vertex IDs must be unique and strictly ordered",
    ):
        PolytopeSupportResult.model_validate(payload)

    payload = square_result.model_dump(mode="json")
    duplicated = dict(payload["exposed_face"]["vertices"][0])
    duplicated["vertex_id"] = "zz_dup"
    payload["exposed_face"]["vertices"].append(duplicated)

    with pytest.raises(
        ValidationError,
        match="exposed-face vertices must have distinct coordinates",
    ):
        PolytopeSupportResult.model_validate(payload)


def test_result_preflights_foreign_exposed_face_space_before_nested_parsing(
    square_result: PolytopeSupportResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exposed face declaring a different, individually valid raw space
    is rejected before the retained source pays its exact extremality
    proof, exactly as the covector space is compared in the same gate."""

    _forbid_extremality_proof(monkeypatch)
    payload = square_result.model_dump(mode="json")
    payload["exposed_face"]["space"] = {"axes": ["u", "v"]}

    with pytest.raises(
        ValidationError,
        match="exposed face must use the same coordinate space as the polytope",
    ):
        PolytopeSupportResult.model_validate(payload)

    payload = square_result.model_dump(mode="json")
    payload["exposed_face"]["space"] = {"axes": ["y", "x"]}

    with pytest.raises(
        ValidationError,
        match="exposed face must use the same coordinate space as the polytope",
    ):
        PolytopeSupportResult.model_validate(payload)


def test_constructed_result_rejects_foreign_exposed_face_space_at_bind(
    square_result: PolytopeSupportResult,
) -> None:
    """An already-built face is bound to the source coordinate space by
    the after-validator instead of the raw-payload preflight."""

    with pytest.raises(ValidationError, match="exposed face must be exactly"):
        PolytopeSupportResult(
            polytope=square_result.polytope,
            covector=square_result.covector,
            support_value=square_result.support_value,
            exposed_face=RationalExposedFace(
                space=RationalCoordinateSpace(axes=("u", "v")),
                vertices=square_result.exposed_face.vertices,
            ),
        )


def test_result_rejects_forbidden_extra_field_before_nested_parsing(
    square_result: PolytopeSupportResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forbidden outer field is rejected before the retained
    V-polytope pays its exact extremality proof."""

    _forbid_extremality_proof(monkeypatch)
    payload = square_result.model_dump(mode="json")
    payload["junk"] = 1

    with pytest.raises(
        ValidationError,
        match="unexpected fields for a support result",
    ):
        PolytopeSupportResult.model_validate(payload)


def test_serialized_result_still_round_trips_through_conclusion_gate(
    square_result: PolytopeSupportResult,
) -> None:
    """The conclusion gate admits the exact published result shape."""

    assert (
        PolytopeSupportResult.model_validate(square_result.model_dump(mode="json"))
        == square_result
    )


def test_serialized_result_round_trips_at_the_envelope_boundary() -> None:
    """Exactly-at-envelope results deserialize unchanged."""

    at_bound = _large_rational(MAX_SUPPORT_COMPONENT_DIGITS)
    polytope = RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=(
            RationalPolytopeVertex(
                vertex_id="a",
                coordinates=(_rational(0), _rational(0)),
            ),
            RationalPolytopeVertex(
                vertex_id="b",
                coordinates=(at_bound, _rational(0)),
            ),
            RationalPolytopeVertex(
                vertex_id="c",
                coordinates=(_rational(0), _rational(1)),
            ),
        ),
    )
    result = compute_polytope_support(
        PolytopeSupportRequest(polytope=polytope, covector=_covector(1, 0))
    )

    assert (
        PolytopeSupportResult.model_validate(result.model_dump(mode="json")) == result
    )


def test_large_coordinate_canonical_values_construct_and_round_trip() -> None:
    beyond_envelope = MAX_SUPPORT_COMPONENT_DIGITS + 10
    big = _large_rational(beyond_envelope)
    bigger = _large_rational(beyond_envelope + 1)
    polytope = RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=(
            RationalPolytopeVertex(
                vertex_id="origin",
                coordinates=(_rational(0), _rational(0)),
            ),
            RationalPolytopeVertex(
                vertex_id="right",
                coordinates=(big, _rational(0)),
            ),
            RationalPolytopeVertex(
                vertex_id="top",
                coordinates=(_rational(0), bigger),
            ),
        ),
    )
    covector = RationalCovector(
        space=RationalCoordinateSpace(axes=("x", "y")),
        components=(big, bigger),
    )

    assert (
        RationalVPolytope.model_validate(polytope.model_dump(mode="json")) == polytope
    )
    assert RationalCovector.model_validate(covector.model_dump(mode="json")) == covector


def test_extremality_subfacet_bound_is_enforced_before_filtering() -> None:
    axes = ("a", "b", "c", "d", "e")
    vertices = tuple(
        RationalPolytopeVertex(
            vertex_id=f"v{index:02d}",
            coordinates=tuple(_rational(index ** (power + 1)) for power in range(5)),
        )
        for index in range(64)
    )

    with pytest.raises(ValidationError, match="subfacet bound"):
        RationalVPolytope(space=RationalCoordinateSpace(axes=axes), vertices=vertices)


def test_extremality_orientation_work_is_enforced_before_filtering() -> None:
    axes = ("x", "y", "z")
    vertices = tuple(
        RationalPolytopeVertex(
            vertex_id=f"v{index:02d}",
            coordinates=(_rational(index), _rational(index**2), _rational(index**3)),
        )
        for index in range(64)
    )

    with pytest.raises(ValidationError, match="orientation-test bound"):
        RationalVPolytope(space=RationalCoordinateSpace(axes=axes), vertices=vertices)


def test_extremality_budget_bounds_are_enforced_before_exact_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vertex family over either published work bound is rejected before
    any canonical coordinate is converted or row-reduced: the budgets
    depend only on the vertex count and dimension."""

    def unexpected_conversion(polytope: object) -> None:
        raise AssertionError("exact conversion ran before the budget gate")

    monkeypatch.setattr(
        polytope_operations, "_support_sympy_points", unexpected_conversion
    )

    six_axes = tuple(f"x{axis}" for axis in range(6))
    vertices_64_by_6 = tuple(
        RationalPolytopeVertex(
            vertex_id=f"v{index:02d}",
            coordinates=tuple(_rational(index ** (power + 1)) for power in range(6)),
        )
        for index in range(64)
    )

    with pytest.raises(ValidationError, match="subfacet bound"):
        RationalVPolytope(
            space=RationalCoordinateSpace(axes=six_axes),
            vertices=vertices_64_by_6,
        )

    four_axes = ("w", "x", "y", "z")
    vertices_40_by_4 = tuple(
        RationalPolytopeVertex(
            vertex_id=f"v{index:02d}",
            coordinates=tuple(_rational(index ** (power + 1)) for power in range(4)),
        )
        for index in range(40)
    )

    with pytest.raises(ValidationError, match="orientation-test bound"):
        RationalVPolytope(
            space=RationalCoordinateSpace(axes=four_axes),
            vertices=vertices_40_by_4,
        )


def test_support_schema_publishes_component_and_hull_work_bounds() -> None:
    schema = PolytopeSupportRequest.model_json_schema()
    vertex_description = schema["$defs"]["RationalVPolytope"]["properties"]["vertices"][
        "description"
    ]

    assert str(MAX_SUPPORT_COMPONENT_DIGITS) in schema["description"]
    for field in ("polytope", "covector"):
        description = schema["properties"][field]["description"]
        assert str(MAX_SUPPORT_COMPONENT_DIGITS) in description
    assert "C(n,d)" in vertex_description
    assert "orientation tests" in vertex_description


def test_support_schema_publishes_common_space_requirement() -> None:
    schema = PolytopeSupportRequest.model_json_schema()

    descriptions = [schema["description"]]
    descriptions.extend(
        schema["properties"][field]["description"] for field in ("polytope", "covector")
    )
    for description in descriptions:
        assert "space" in description
        assert "identical" in description


def test_support_example_teaches_common_space_precondition() -> None:
    support_tool = next(
        tool
        for tool in POLYTOPE_OPERATIONS
        if tool.operation_id == "polytope.rational.support.compute"
    )
    (only_example,) = support_tool.examples

    assert "identical to the polytope's" in only_example.description
