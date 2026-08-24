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
    MAX_EXTREMALITY_HEIGHT_WORK,
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


def _forbid_support_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if bind's exact support replay is ever reached."""

    def unexpected_replay(polytope: object, covector: object) -> object:
        raise AssertionError("support replay ran before the conclusion preflight")

    monkeypatch.setattr(polytope_operations, "support_data", unexpected_replay)


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


def test_request_preflights_built_foreign_covector_space_before_hull_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serialized near-limit polytope paired with an already-built
    covector from a different space is rejected by the same raw gate
    before nested parsing reconstructs (and proves) the source."""

    _forbid_extremality_proof(monkeypatch)
    payload = _square_payload()
    payload["covector"] = RationalCovector(
        space=RationalCoordinateSpace(axes=("u", "v")),
        components=(_rational(0), _rational(1)),
    )

    with pytest.raises(
        ValidationError,
        match="polytope and covector must use the same coordinate space",
    ):
        PolytopeSupportRequest.model_validate(payload)


def test_request_gate_covers_built_polytope_space_pairing() -> None:
    """The covector gate reads a constructed polytope's declared axes
    too, so an all-built foreign pairing is rejected by the same cheap
    check instead of the after-validator."""

    payload = _square_payload()
    payload["polytope"] = RationalVPolytope(
        space=RationalCoordinateSpace(axes=("u", "v")),
        vertices=(
            _vertex("bottom_left", 0, 0),
            _vertex("bottom_right", 1, 0),
            _vertex("top_left", 0, 1),
            _vertex("top_right", 1, 1),
        ),
    )
    payload["covector"] = _covector(1, 0)

    with pytest.raises(
        ValidationError,
        match="polytope and covector must use the same coordinate space",
    ):
        PolytopeSupportRequest.model_validate(payload)


def test_built_matching_covector_composes_with_serialized_polytope() -> None:
    """A constructed covector sharing the serialized source's declared
    axes still composes unchanged: the mixed-payload gate mirrors the
    common-space rule without narrowing the published domain."""

    payload = _square_payload()
    payload["covector"] = _covector(1, 0)
    request = PolytopeSupportRequest.model_validate(payload)
    result = compute_polytope_support(request)

    assert result.support_value.as_fraction() == 1
    assert tuple(vertex.vertex_id for vertex in result.exposed_face.vertices) == (
        "bottom_right",
        "top_right",
    )


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


def test_result_preflights_built_foreign_exposed_face_space_before_nested_parsing(
    square_result: PolytopeSupportResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serialized near-limit source paired with an already-built face
    from a different space is rejected by the same raw gate before nested
    parsing reconstructs (and proves) the retained polytope."""

    _forbid_extremality_proof(monkeypatch)
    payload = square_result.model_dump(mode="json")
    payload["exposed_face"] = RationalExposedFace(
        space=RationalCoordinateSpace(axes=("u", "v")),
        vertices=(
            RationalPolytopeVertex(
                vertex_id="u_left",
                coordinates=(_rational(0), _rational(1)),
            ),
            RationalPolytopeVertex(
                vertex_id="u_right",
                coordinates=(_rational(1), _rational(1)),
            ),
        ),
    )

    with pytest.raises(
        ValidationError,
        match="exposed face must use the same coordinate space as the polytope",
    ):
        PolytopeSupportResult.model_validate(payload)


def test_result_preflights_forged_support_value_before_nested_parsing(
    square_result: PolytopeSupportResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A well-formed but incorrect support value binds against the raw
    source before nested validation reconstructs (and proves) it."""

    _forbid_extremality_proof(monkeypatch)
    payload = square_result.model_dump(mode="json")
    payload["support_value"] = {"num": "2", "den": "1"}

    with pytest.raises(
        ValidationError,
        match="support value must equal the exact maximum on every vertex",
    ):
        PolytopeSupportResult.model_validate(payload)


def test_result_preflights_forged_exposed_face_before_nested_parsing(
    square_result: PolytopeSupportResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-space face with wrong membership binds against the raw
    source before nested validation reconstructs (and proves) it."""

    _forbid_extremality_proof(monkeypatch)
    payload = square_result.model_dump(mode="json")
    payload["exposed_face"]["vertices"] = payload["exposed_face"]["vertices"][:1]

    with pytest.raises(
        ValidationError,
        match="exposed face must be exactly the complete maximizing vertex family",
    ):
        PolytopeSupportResult.model_validate(payload)


def test_result_preflight_binding_defers_unparsed_structural_faults(
    square_result: PolytopeSupportResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serialized source whose vertex rows are structurally malformed in
    ways the binding assembly does not model still reaches ordinary nested
    validation, which owns those published errors."""

    payload = square_result.model_dump(mode="json")
    del payload["polytope"]["vertices"][1]["vertex_id"]

    with pytest.raises(ValidationError, match="vertex_id"):
        PolytopeSupportResult.model_validate(payload)


def test_constructed_result_rejects_foreign_exposed_face_space_at_construction(
    square_result: PolytopeSupportResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-built face in another coordinate space is rejected at
    construction with the typed space mismatch, before any binding work
    — including bind's exact support replay — runs."""

    _forbid_support_replay(monkeypatch)

    with pytest.raises(ValidationError, match="same coordinate space"):
        PolytopeSupportResult(
            polytope=square_result.polytope,
            covector=square_result.covector,
            support_value=square_result.support_value,
            exposed_face=RationalExposedFace(
                space=RationalCoordinateSpace(axes=("u", "v")),
                vertices=square_result.exposed_face.vertices,
            ),
        )


def test_built_matching_exposed_face_composes_with_serialized_result(
    square_result: PolytopeSupportResult,
) -> None:
    """A constructed face sharing the serialized source's declared axes
    still composes unchanged: the mixed-payload gate mirrors the
    common-space rule without narrowing the published domain."""

    payload = square_result.model_dump(mode="json")
    payload["exposed_face"] = square_result.exposed_face

    assert PolytopeSupportResult.model_validate(payload) == square_result


def test_constructed_result_still_rejects_forged_face_at_bind(
    square_result: PolytopeSupportResult,
) -> None:
    """A built face in the correct space but with forged membership is
    still bound to the source by the after-validator."""

    with pytest.raises(ValidationError, match="exposed face must be exactly"):
        PolytopeSupportResult(
            polytope=square_result.polytope,
            covector=square_result.covector,
            support_value=square_result.support_value,
            exposed_face=RationalExposedFace(
                space=square_result.polytope.space,
                vertices=(square_result.polytope.vertices[0],),
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


def test_extremality_height_work_bound_is_enforced_before_exact_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vertex family with admitted counts but canonical-limit heights is
    rejected before any conversion: exact determinant work grows as
    orientation_tests * height^2, so counts alone cannot bound the proof."""

    def unexpected_conversion(polytope: object) -> None:
        raise AssertionError("exact conversion ran before the height-work gate")

    monkeypatch.setattr(
        polytope_operations, "_support_sympy_points", unexpected_conversion
    )

    four_axes = ("w", "x", "y", "z")
    digits = 32_768  # the canonical rational limit: counts pass, work does not
    vertices_12_by_4 = tuple(
        RationalPolytopeVertex(
            vertex_id=f"v{index:02d}",
            coordinates=tuple(
                CanonicalRational(num="9" * (digits - 2) + f"{index:02d}", den="1")
                for _ in range(4)
            ),
        )
        for index in range(12)
    )

    with pytest.raises(ValidationError, match="height-work bound"):
        RationalVPolytope(
            space=RationalCoordinateSpace(axes=four_axes),
            vertices=vertices_12_by_4,
        )


def test_extremality_height_work_grades_admission_by_coordinate_height() -> None:
    """At a height the count gates alone would admit freely, only families
    whose orientation-test count fits the coupled budget still validate:
    the same near-threshold height admits a 3-vertex triangle and rejects a
    6-vertex dim-3 family just past ``MAX_EXTREMALITY_HEIGHT_WORK``."""

    threshold_digits = 18_257  # 3 * 18_257^2 <= MAX < 60 * 18_258^2
    big = CanonicalRational(num="9" * threshold_digits, den="1")
    zero = CanonicalRational(num="0", den="1")
    triangle = RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=(
            RationalPolytopeVertex(vertex_id="origin", coordinates=(zero, zero)),
            RationalPolytopeVertex(vertex_id="right", coordinates=(big, zero)),
            RationalPolytopeVertex(vertex_id="top", coordinates=(zero, big)),
        ),
    )
    assert len(triangle.vertices[1].coordinates[0].num) == threshold_digits

    def unexpected_conversion(polytope: object) -> None:
        raise AssertionError("exact conversion ran before the height-work gate")

    over_digits = threshold_digits + 1
    vertices_over = tuple(
        RationalPolytopeVertex(
            vertex_id=f"v{index:02d}",
            coordinates=tuple(
                CanonicalRational(num="9" * (over_digits - 2) + f"{index:02d}", den="1")
                for _ in range(3)
            ),
        )
        for index in range(6)
    )
    with pytest.raises(ValidationError, match="height-work bound") as exc_info:
        RationalVPolytope(
            space=RationalCoordinateSpace(axes=("x", "y", "z")),
            vertices=vertices_over,
        )
    assert str(MAX_EXTREMALITY_HEIGHT_WORK) in str(exc_info.value)


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
    assert str(MAX_EXTREMALITY_HEIGHT_WORK) in vertex_description


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


def test_surrogate_axis_label_rejected_before_hull_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lone surrogate axis label cannot cross strict JSON serialization,
    so the request must fail on label validation before the exact
    extremality proof runs."""
    _forbid_extremality_proof(monkeypatch)
    payload = _square_payload()
    payload["polytope"]["space"]["axes"] = ["x\ud800", "y"]
    payload["covector"]["space"]["axes"] = ["x\ud800", "y"]

    with pytest.raises(ValidationError, match=r"[Uu]nicode"):
        PolytopeSupportRequest.model_validate(payload)


def test_surrogate_vertex_id_rejected_before_hull_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_extremality_proof(monkeypatch)
    payload = _square_payload()
    payload["polytope"]["vertices"][1]["vertex_id"] = "br\ud800"

    with pytest.raises(ValidationError, match=r"[Uu]nicode"):
        PolytopeSupportRequest.model_validate(payload)


def test_result_preflights_surrogate_face_vertex_id_before_nested_parsing(
    square_result: PolytopeSupportResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exposed-face vertex ID outside the Unicode scalar label grammar
    fails the raw conclusion gate before nested parsing can reach the
    retained source's exact extremality proof."""
    _forbid_extremality_proof(monkeypatch)
    payload = square_result.model_dump(mode="json")
    payload["exposed_face"]["vertices"][0]["vertex_id"] = "top_left\ud800"

    with pytest.raises(ValidationError, match=r"[Uu]nicode"):
        PolytopeSupportResult.model_validate(payload)


def test_seven_axis_polytope_rejects_support_pairing_before_hull_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canonical space may carry seven axes for facet-profile sources,
    but support pairs a V-polytope with a covector of at most six
    components, so the space mismatch is reported by the raw covector gate
    before any exact proof runs."""
    _forbid_extremality_proof(monkeypatch)
    payload = _square_payload()
    payload["polytope"]["space"]["axes"] = [f"x{axis}" for axis in range(7)]

    with pytest.raises(ValidationError, match="same coordinate space"):
        PolytopeSupportRequest.model_validate(payload)


def test_seven_axis_polytope_cannot_pair_with_any_covector() -> None:
    axes = RationalCoordinateSpace(axes=tuple(f"x{axis}" for axis in range(7)))

    with pytest.raises(ValidationError, match=r"at most 6 items|at most 6 entries"):
        RationalCovector(
            space=axes,
            components=tuple(_rational(component) for component in range(7)),
        )

    polytope = RationalVPolytope(
        space=axes,
        vertices=(
            RationalPolytopeVertex(
                vertex_id=f"v{index:02d}",
                coordinates=tuple(
                    _rational(1 if coordinate == index else 0)
                    for coordinate in range(7)
                ),
            )
            for index in range(8)
        ),
    )
    with pytest.raises(ValueError, match="same coordinate space"):
        polytope_operations.polytope_support(polytope, _covector(0, 1))


def test_accepted_result_encodes_strict_json(square_result) -> None:
    """Every accepted canonical result crosses the supported serialization
    boundary unchanged."""
    from jacobian.canonical import CanonicalLimits, encode_strict_json

    encode_strict_json(square_result.model_dump(mode="json"), limits=CanonicalLimits())


def _oversized_face_payload() -> dict[str, object]:
    """A serialized face at the declared container maxima whose aggregate
    coordinates alone encode past the strict JSON transport limit.

    Sixty-four vertices in six dimensions, five near-canonical-limit
    components per row (rows stay distinct through their first coordinate),
    so the payload is individually well-formed but cannot cross the
    supported serialization boundary as a whole.
    """
    big = {"num": "9" * 32_768, "den": "8"}
    return {
        "space": {"axes": [f"x{axis}" for axis in range(6)]},
        "vertices": [
            {
                "vertex_id": f"{index:02d}",
                "coordinates": [
                    {"num": str(index), "den": "1"},
                    *[{"num": big["num"], "den": big["den"]}] * 5,
                ],
            }
            for index in range(64)
        ],
    }


def test_oversized_aggregate_face_rejected_before_nested_parsing() -> None:
    """A face whose coordinates alone exceed the transport envelope can
    never compose across the supported serialization boundary, so the
    aggregate bound must reject it as a typed validation error before any
    coordinate is parsed."""
    from jacobian.canonical import CanonicalLimits, encode_strict_json

    payload = _oversized_face_payload()
    encoded = len(
        encode_strict_json(payload, limits=CanonicalLimits(max_output_bytes=2**30))
    )
    assert encoded > CanonicalLimits().max_output_bytes

    with pytest.raises(
        ValidationError,
        match="exposed face exceeds the canonical JSON output bound",
    ):
        RationalExposedFace.model_validate(payload)


def test_built_component_shapes_are_measured_by_the_aggregate_bound() -> None:
    """The aggregate gate measures authored reduced components the same way
    whether they arrive as raw payloads or as built canonical values."""
    from jacobian.math.polytope._models import _estimate_face_wire_bytes

    big = CanonicalRational(num="9" * 32_768, den="8")
    space = RationalCoordinateSpace(axes=("x", "y"))
    vertices = (
        RationalPolytopeVertex(vertex_id="a", coordinates=(big, big)),
        RationalPolytopeVertex(vertex_id="b", coordinates=(big, big)),
    )

    built_estimate = _estimate_face_wire_bytes({"space": space, "vertices": vertices})
    raw_estimate = _estimate_face_wire_bytes(
        {
            "space": {"axes": ["x", "y"]},
            "vertices": [
                {
                    "vertex_id": "a",
                    "coordinates": [{"num": big.num, "den": big.den}] * 2,
                },
                {
                    "vertex_id": "b",
                    "coordinates": [{"num": big.num, "den": big.den}] * 2,
                },
            ],
        }
    )

    assert built_estimate == raw_estimate
    assert built_estimate > 2 * 2 * (len(big.num) + len(big.den))


def test_single_vertex_face_with_maximal_components_still_composes() -> None:
    """The aggregate bound admits faces under the transport limit: maximal
    per-component heights alone do not trigger it, and the accepted value
    encodes strictly."""
    from jacobian.canonical import CanonicalLimits, encode_strict_json

    big = _large_rational(32_768)
    face = RationalExposedFace(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=(RationalPolytopeVertex(vertex_id="v00", coordinates=(big, big)),),
    )

    encoded = len(
        encode_strict_json(face.model_dump(mode="json"), limits=CanonicalLimits())
    )
    assert encoded < CanonicalLimits().max_output_bytes
