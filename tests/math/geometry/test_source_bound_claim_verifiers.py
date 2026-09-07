"""Serialized source binding and claim-verifier regressions for exact geometry."""

from __future__ import annotations

import json

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.exact import (
    distance_graph,
    distance_profile,
    euclidean_orbit_profile,
    pinned_line_distance_profile,
    verify_distance_graph,
    verify_distance_profile,
    verify_euclidean_orbit_profile,
    verify_pinned_line_distance_profile,
)
from jacobian.math.geometry.exact._models import (
    DistanceGraphResult,
    DistanceProfileResult,
    LabelledRationalPoint,
    PinnedLineConfiguration,
    PointConfiguration,
)
from jacobian.math.geometry.exact.pinned_distance import (
    compute_pinned_distance_support_profile,
    verify_pinned_distance_support_profile,
)
from jacobian.math.geometry.exact.spanned_line_profile import (
    compute_spanned_line_profile,
    verify_spanned_line_profile,
)
from jacobian.math.geometry.exact.triangle_area_profile import (
    compute_triangle_area_profile,
    verify_triangle_area_profile,
)


def _q(value: int) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value, 1)


def _configuration() -> PointConfiguration:
    return PointConfiguration(
        points=(
            LabelledRationalPoint(label="a", coordinates=(_q(0), _q(0))),
            LabelledRationalPoint(label="b", coordinates=(_q(1), _q(0))),
            LabelledRationalPoint(label="c", coordinates=(_q(0), _q(1))),
        )
    )


def _forged_json(result: object, path: tuple[object, ...], value: object) -> object:
    payload = json.loads(result.model_dump_json())  # type: ignore[attr-defined]
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return payload


def test_distance_profiles_retain_sources_and_reject_serialized_forgery() -> None:
    source = _configuration()
    profile = distance_profile(source)
    decoded = DistanceProfileResult.model_validate_json(profile.model_dump_json())
    assert verify_distance_profile(decoded)

    forged = DistanceProfileResult.model_validate_json(
        json.dumps(_forged_json(profile, ("entries", 0, "pair_count"), 99))
    )
    assert not verify_distance_profile(forged)


def test_distance_graph_retain_target_and_source() -> None:
    source = _configuration()
    result = distance_graph(source, _q(1))
    decoded = DistanceGraphResult.model_validate_json(result.model_dump_json())
    assert verify_distance_graph(decoded)

    forged_target = DistanceGraphResult.model_validate_json(
        json.dumps(_forged_json(result, ("target_squared_distance", "num"), "-1")),
        strict=True,
    )
    assert not verify_distance_graph(forged_target)

    forged_axis = DistanceGraphResult.model_validate_json(
        json.dumps(_forged_json(result, ("graph", "vertex_count"), 4)),
        strict=True,
    )
    assert not verify_distance_graph(forged_axis)


def test_orbit_and_pinned_line_claims_reject_serialized_forgery() -> None:
    source = _configuration()
    orbit = euclidean_orbit_profile(source)
    decoded_orbit = type(orbit).model_validate_json(orbit.model_dump_json())
    assert verify_euclidean_orbit_profile(decoded_orbit)
    forged_orbit = type(orbit).model_validate_json(
        json.dumps(_forged_json(orbit, ("isometry_form", "entries", 0, 0, "num"), "7")),
        strict=True,
    )
    assert not verify_euclidean_orbit_profile(forged_orbit)

    pinned_source = PinnedLineConfiguration.model_validate(source.model_dump())
    pinned = pinned_line_distance_profile(pinned_source, (_q(0), _q(1)))
    decoded_pinned = type(pinned).model_validate_json(pinned.model_dump_json())
    assert verify_pinned_line_distance_profile(decoded_pinned)
    forged_pinned = type(pinned).model_validate_json(
        json.dumps(_forged_json(pinned, ("anchor", 0, "num"), "1"))
    )
    assert not verify_pinned_line_distance_profile(forged_pinned)


def test_point_profile_claims_reject_serialized_forgery() -> None:
    source = _configuration()

    support = compute_pinned_distance_support_profile(source)
    decoded_support = type(support).model_validate_json(support.model_dump_json())
    assert verify_pinned_distance_support_profile(decoded_support)
    forged_support = type(support).model_validate_json(
        json.dumps(
            _forged_json(
                support, ("entries", 0, "distance_classes", 0, "target_labels"), []
            )
        )
    )
    assert not verify_pinned_distance_support_profile(forged_support)

    lines = compute_spanned_line_profile(source)
    decoded_lines = type(lines).model_validate_json(lines.model_dump_json())
    assert verify_spanned_line_profile(decoded_lines)
    forged_lines = type(lines).model_validate_json(
        json.dumps(_forged_json(lines, ("line_count",), 0))
    )
    assert not verify_spanned_line_profile(forged_lines)

    areas = compute_triangle_area_profile(source)
    decoded_areas = type(areas).model_validate_json(areas.model_dump_json())
    assert verify_triangle_area_profile(decoded_areas)
    forged_areas = type(areas).model_validate_json(
        json.dumps(_forged_json(areas, ("entries", 0, "area", "num"), "7"))
    )
    assert not verify_triangle_area_profile(forged_areas)
