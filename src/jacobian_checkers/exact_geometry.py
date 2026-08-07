"""Independent replay of selected exact rational planar-geometry results.

The checker intentionally depends only on the Python standard library. It does
not import SymPy, Jacobian domain operations, or their computational helpers.
"""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from typing import Any

from jacobian_checkers.bound_artifacts import valid_unscoped_unencoded_bindings

_ARTIFACT_URI = re.compile(r"^artifact://sha256/[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_ARTIFACT_KEYS = {
    "artifact_uri",
    "object_digest",
    "payload_digest",
    "schema_uri",
    "semantics_uri",
    "parents",
    "payload",
}
_OPERATIONS = {
    "geometry.points.compute.squared_distance",
    "geometry.points.compute.convex_hull",
    "geometry.segment.compute.midpoint",
    "geometry.segments.intersection.compute",
    "geometry.polygon.simple.decide",
    "geometry.polygon.point.classify",
    "geometry.triangle.compute.orientation",
    "geometry.triangle.compute.centroid",
}


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _valid_artifact(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != _ARTIFACT_KEYS:
        return False
    parents = value.get("parents")
    return (
        all(
            isinstance(value.get(key), str)
            and (
                _ARTIFACT_URI.fullmatch(value[key]) is not None
                if key in {"artifact_uri", "schema_uri", "semantics_uri"}
                else _DIGEST.fullmatch(value[key]) is not None
            )
            for key in (
                "artifact_uri",
                "object_digest",
                "payload_digest",
                "schema_uri",
                "semantics_uri",
            )
        )
        and isinstance(parents, list)
        and len(parents) == len(set(parents))
        and all(
            isinstance(parent, str) and _ARTIFACT_URI.fullmatch(parent) is not None
            for parent in parents
        )
    )


def _integer(value: object) -> int:
    if not isinstance(value, str) or _INTEGER.fullmatch(value) is None:
        raise ValueError("rational component is not a canonical integer")
    result = int(value)
    if str(result) != value:
        raise ValueError("rational component is not canonical")
    return result


def _rational(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational value has an invalid shape")
    numerator = _integer(value["num"])
    denominator = _integer(value["den"])
    if denominator <= 0:
        raise ValueError("rational denominator must be positive")
    result = Fraction(numerator, denominator)
    if (result.numerator, result.denominator) != (numerator, denominator):
        raise ValueError("rational value is not reduced")
    return result


def _point(value: object) -> tuple[Fraction, Fraction]:
    if not isinstance(value, dict) or set(value) != {"x", "y"}:
        raise ValueError("point has an invalid shape")
    return _rational(value["x"]), _rational(value["y"])


def _pair(payload: object) -> tuple[tuple[Fraction, Fraction], ...]:
    if not isinstance(payload, dict) or set(payload) != {"first", "second"}:
        raise ValueError("point pair has an invalid shape")
    return _point(payload["first"]), _point(payload["second"])


def _triple(payload: object) -> tuple[tuple[Fraction, Fraction], ...]:
    if not isinstance(payload, dict) or set(payload) != {"first", "second", "third"}:
        raise ValueError("point triple has an invalid shape")
    return (
        _point(payload["first"]),
        _point(payload["second"]),
        _point(payload["third"]),
    )


def _points(payload: object) -> list[tuple[Fraction, Fraction]]:
    if not isinstance(payload, dict) or set(payload) != {"points"}:
        raise ValueError("point set has an invalid shape")
    raw = payload["points"]
    if not isinstance(raw, list) or not 1 <= len(raw) <= 128:
        raise ValueError("point set has an invalid size")
    points = [_point(item) for item in raw]
    if len(points) != len(set(points)):
        raise ValueError("point set repeats coordinates")
    return points


def _segment(
    value: object,
) -> tuple[
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
]:
    if not isinstance(value, dict) or set(value) != {"start", "end"}:
        raise ValueError("closed segment has an invalid shape")
    return _point(value["start"]), _point(value["end"])


def _segment_pair(
    payload: object,
) -> tuple[
    tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
]:
    if not isinstance(payload, dict) or set(payload) != {"first", "second"}:
        raise ValueError("segment pair has an invalid shape")
    return _segment(payload["first"]), _segment(payload["second"])


def _cross(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
) -> Fraction:
    return left[0] * right[1] - left[1] * right[0]


def _subtract(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    return left[0] - right[0], left[1] - right[1]


def _add_scaled(
    point: tuple[Fraction, Fraction],
    direction: tuple[Fraction, Fraction],
    parameter: Fraction,
) -> tuple[Fraction, Fraction]:
    return (
        point[0] + parameter * direction[0],
        point[1] + parameter * direction[1],
    )


def _on_segment(
    point: tuple[Fraction, Fraction],
    start: tuple[Fraction, Fraction],
    end: tuple[Fraction, Fraction],
) -> bool:
    return _cross(_subtract(point, start), _subtract(end, start)) == 0 and all(
        min(left, right) <= value <= max(left, right)
        for value, left, right in zip(point, start, end, strict=True)
    )


def _segment_intersection(
    first: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    second: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
) -> dict[str, object]:
    first_start, first_end = first
    second_start, second_end = second
    first_degenerate = first_start == first_end
    second_degenerate = second_start == second_end
    if first_degenerate or second_degenerate:
        if first_degenerate and _on_segment(first_start, second_start, second_end):
            point = first_start
        elif second_degenerate and _on_segment(second_start, first_start, first_end):
            point = second_start
        else:
            return {
                "status": "DISJOINT",
                "point": None,
                "contact_kind": None,
                "overlap": None,
            }
        return {
            "status": "POINT",
            "point": point,
            "contact_kind": "DEGENERATE_TOUCH",
            "overlap": None,
        }
    first_direction = _subtract(first_end, first_start)
    second_direction = _subtract(second_end, second_start)
    denominator = _cross(first_direction, second_direction)
    offset = _subtract(second_start, first_start)
    if denominator != 0:
        first_parameter = _cross(offset, second_direction) / denominator
        second_parameter = _cross(offset, first_direction) / denominator
        if not (0 <= first_parameter <= 1 and 0 <= second_parameter <= 1):
            return {
                "status": "DISJOINT",
                "point": None,
                "contact_kind": None,
                "overlap": None,
            }
        return {
            "status": "POINT",
            "point": _add_scaled(
                first_start,
                first_direction,
                first_parameter,
            ),
            "contact_kind": (
                "PROPER"
                if 0 < first_parameter < 1 and 0 < second_parameter < 1
                else "ENDPOINT_TOUCH"
            ),
            "overlap": None,
        }
    if _cross(offset, first_direction) != 0:
        return {
            "status": "DISJOINT",
            "point": None,
            "contact_kind": None,
            "overlap": None,
        }
    common = sorted(
        {
            point
            for point in (first_start, first_end, second_start, second_end)
            if _on_segment(point, first_start, first_end)
            and _on_segment(point, second_start, second_end)
        }
    )
    if not common:
        return {
            "status": "DISJOINT",
            "point": None,
            "contact_kind": None,
            "overlap": None,
        }
    if len(common) == 1:
        return {
            "status": "POINT",
            "point": common[0],
            "contact_kind": "ENDPOINT_TOUCH",
            "overlap": None,
        }
    return {
        "status": "OVERLAP",
        "point": None,
        "contact_kind": None,
        "overlap": (common[0], common[-1]),
    }


def _convex_hull(
    points: list[tuple[Fraction, Fraction]],
) -> list[tuple[Fraction, Fraction]]:
    ordered = sorted(points)
    if len(ordered) <= 1:
        return ordered

    def turn(
        first: tuple[Fraction, Fraction],
        second: tuple[Fraction, Fraction],
        third: tuple[Fraction, Fraction],
    ) -> Fraction:
        return _cross(_subtract(second, first), _subtract(third, first))

    lower: list[tuple[Fraction, Fraction]] = []
    for point in ordered:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[Fraction, Fraction]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _polygon_decision(
    points: list[tuple[Fraction, Fraction]],
) -> dict[str, object]:
    if len(points) < 3:
        raise ValueError("polygon ring is too small")
    checked = 0
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            checked += 1
            intersection = _segment_intersection(
                (points[first], points[(first + 1) % len(points)]),
                (points[second], points[(second + 1) % len(points)]),
            )
            adjacent = (first - second) % len(points) in {1, len(points) - 1}
            shared = (
                points[0] if (first, second) == (0, len(points) - 1) else points[second]
            )
            valid = (
                intersection["status"] == "POINT"
                and intersection["point"] == shared
                and intersection["contact_kind"] == "ENDPOINT_TOUCH"
                if adjacent
                else intersection["status"] == "DISJOINT"
            )
            if not valid:
                return {
                    "vertex_count": len(points),
                    "is_simple": False,
                    "checked_edge_pairs": checked,
                    "witness": {
                        "first_edge_index": first,
                        "second_edge_index": second,
                        "intersection": intersection,
                    },
                }
    return {
        "vertex_count": len(points),
        "is_simple": True,
        "checked_edge_pairs": checked,
        "witness": None,
    }


def _point_classification(
    polygon: list[tuple[Fraction, Fraction]],
    point: tuple[Fraction, Fraction],
) -> dict[str, object]:
    if not _polygon_decision(polygon)["is_simple"]:
        raise ValueError("classification polygon is not simple")
    for index, start in enumerate(polygon):
        if _on_segment(point, start, polygon[(index + 1) % len(polygon)]):
            return {
                "polygon_vertex_count": len(polygon),
                "classification": "BOUNDARY",
                "boundary_edge_index": index,
            }
    inside = False
    x, y = point
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if (start[1] > y) != (end[1] > y):
            crossing_x = start[0] + (y - start[1]) * (end[0] - start[0]) / (
                end[1] - start[1]
            )
            if crossing_x > x:
                inside = not inside
    return {
        "polygon_vertex_count": len(polygon),
        "classification": "INSIDE" if inside else "OUTSIDE",
        "boundary_edge_index": None,
    }


def _expected(operation: str, claim: object) -> dict[str, object]:
    if operation == "geometry.points.compute.squared_distance":
        first, second = _pair(claim)
        value = (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
        return {"value": value}
    if operation == "geometry.points.compute.convex_hull":
        return {"points": _convex_hull(_points(claim))}
    if operation == "geometry.segment.compute.midpoint":
        first, second = _pair(claim)
        return {
            "point": (
                (first[0] + second[0]) / 2,
                (first[1] + second[1]) / 2,
            )
        }
    if operation == "geometry.segments.intersection.compute":
        return _segment_intersection(*_segment_pair(claim))
    if operation == "geometry.polygon.simple.decide":
        return _polygon_decision(_points(claim))
    if operation == "geometry.polygon.point.classify":
        if not isinstance(claim, dict) or set(claim) != {"polygon", "point"}:
            raise ValueError("polygon classification source has an invalid shape")
        return _point_classification(
            _points(claim["polygon"]),
            _point(claim["point"]),
        )
    first, second, third = _triple(claim)
    if operation == "geometry.triangle.compute.orientation":
        determinant = (second[0] - first[0]) * (third[1] - first[1]) - (
            second[1] - first[1]
        ) * (third[0] - first[0])
        return {"orientation": (determinant > 0) - (determinant < 0)}
    if operation == "geometry.triangle.compute.centroid":
        return {
            "point": (
                (first[0] + second[0] + third[0]) / 3,
                (first[1] + second[1] + third[1]) / 3,
            )
        }
    raise ValueError("unsupported exact geometry operation")


def _candidate(payload: object, operation: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("candidate has an invalid shape")
    if operation == "geometry.points.compute.squared_distance":
        if set(payload) != {"value"}:
            raise ValueError("squared-distance candidate has an invalid shape")
        return {"value": _rational(payload["value"])}
    if operation == "geometry.points.compute.convex_hull":
        return {"points": _points(payload)}
    if operation in {
        "geometry.segment.compute.midpoint",
        "geometry.triangle.compute.centroid",
    }:
        if set(payload) != {"point"}:
            raise ValueError("point candidate has an invalid shape")
        return {"point": _point(payload["point"])}
    if operation == "geometry.segments.intersection.compute":
        if set(payload) != {"status", "point", "contact_kind", "overlap"}:
            raise ValueError("segment-intersection candidate has an invalid shape")
        status = payload["status"]
        if status not in {"DISJOINT", "POINT", "OVERLAP"}:
            raise ValueError("segment-intersection status is invalid")
        return {
            "status": status,
            "point": None if payload["point"] is None else _point(payload["point"]),
            "contact_kind": payload["contact_kind"],
            "overlap": (
                None if payload["overlap"] is None else _segment(payload["overlap"])
            ),
        }
    if operation == "geometry.polygon.simple.decide":
        if set(payload) != {
            "vertex_count",
            "is_simple",
            "checked_edge_pairs",
            "witness",
        }:
            raise ValueError("simple-polygon candidate has an invalid shape")
        if (
            type(payload["vertex_count"]) is not int
            or type(payload["is_simple"]) is not bool
            or type(payload["checked_edge_pairs"]) is not int
        ):
            raise ValueError("simple-polygon decision fields are invalid")
        witness = payload["witness"]
        parsed_witness: dict[str, object] | None
        if witness is None:
            parsed_witness = None
        else:
            if not isinstance(witness, dict) or set(witness) != {
                "first_edge_index",
                "second_edge_index",
                "intersection",
            }:
                raise ValueError("polygon witness has an invalid shape")
            if (
                type(witness["first_edge_index"]) is not int
                or type(witness["second_edge_index"]) is not int
            ):
                raise ValueError("polygon witness indices are invalid")
            parsed_witness = {
                "first_edge_index": witness["first_edge_index"],
                "second_edge_index": witness["second_edge_index"],
                "intersection": _candidate(
                    witness["intersection"],
                    "geometry.segments.intersection.compute",
                ),
            }
        return {
            "vertex_count": payload["vertex_count"],
            "is_simple": payload["is_simple"],
            "checked_edge_pairs": payload["checked_edge_pairs"],
            "witness": parsed_witness,
        }
    if operation == "geometry.polygon.point.classify":
        if set(payload) != {
            "polygon_vertex_count",
            "classification",
            "boundary_edge_index",
        }:
            raise ValueError("point-classification candidate has an invalid shape")
        if type(payload["polygon_vertex_count"]) is not int:
            raise ValueError("polygon vertex count is invalid")
        if payload["classification"] not in {"INSIDE", "BOUNDARY", "OUTSIDE"}:
            raise ValueError("point classification is invalid")
        boundary = payload["boundary_edge_index"]
        if boundary is not None and type(boundary) is not int:
            raise ValueError("boundary edge index is invalid")
        return {
            "polygon_vertex_count": payload["polygon_vertex_count"],
            "classification": payload["classification"],
            "boundary_edge_index": boundary,
        }
    if set(payload) != {"orientation"} or type(payload["orientation"]) is not int:
        raise ValueError("orientation candidate has an invalid shape")
    return {"orientation": payload["orientation"]}


def check_exact_geometry(request: dict[str, Any]) -> dict[str, Any]:
    """Accept a bound result exactly when direct rational replay agrees."""

    try:
        if not isinstance(request, dict) or set(request) != {
            "request_version",
            "claim",
            "candidate",
            "semantics",
            "scope",
            "witness",
            "expected_bindings",
        }:
            return _reject("malformed checker request")
        if request["request_version"] != "1" or request["scope"] is not None:
            return _reject("unsupported checker request")
        claim = request["claim"]
        candidate = request["candidate"]
        semantics = request["semantics"]
        witness = request["witness"]
        if not all(
            _valid_artifact(item) for item in (claim, candidate, semantics, witness)
        ):
            return _reject("checker artifact metadata is malformed")
        bindings = request["expected_bindings"]
        if not valid_unscoped_unencoded_bindings(bindings):
            return _reject("expected evidence bindings are malformed")
        if (
            bindings["claim_digest"] != claim["object_digest"]
            or bindings["candidate_digest"] != candidate["object_digest"]
            or bindings["semantics_digest"] != semantics["object_digest"]
            or semantics["artifact_uri"] != claim["semantics_uri"]
        ):
            return _reject("expected evidence bindings do not match artifacts")
        if (
            claim["semantics_uri"] != candidate["semantics_uri"]
            or claim["semantics_uri"] != witness["semantics_uri"]
            or candidate["parents"] != [claim["artifact_uri"]]
        ):
            return _reject("candidate is not exactly bound to the geometry input")
        for artifact in (claim, candidate, semantics, witness):
            if artifact["payload_digest"] != _sha256(
                _canonical_json(artifact["payload"])
            ):
                return _reject("artifact payload digest does not match")
        envelope = witness["payload"]
        if (
            not isinstance(envelope, dict)
            or set(envelope)
            != {
                "evidence_schema_version",
                "witness_format",
                "format_version",
                "role",
                "bindings",
                "payload",
            }
            or envelope["evidence_schema_version"] != "1"
            or envelope["witness_format"] != "geometry.exact_rational_result"
            or envelope["format_version"] != "1"
            or envelope["role"] != "SUPPORTS_CLAIM"
            or envelope["bindings"] != bindings
            or len(witness["parents"]) != 2
            or set(witness["parents"])
            != {claim["artifact_uri"], candidate["artifact_uri"]}
        ):
            return _reject("geometry witness envelope is malformed or rebound")
        payload = envelope["payload"]
        if (
            not isinstance(payload, dict)
            or set(payload) != {"operation_id", "input_uri", "result_uri"}
            or payload["operation_id"] not in _OPERATIONS
            or payload["input_uri"] != claim["artifact_uri"]
            or payload["result_uri"] != candidate["artifact_uri"]
        ):
            return _reject("geometry witness payload is malformed or rebound")
        operation = payload["operation_id"]
        if _candidate(candidate["payload"], operation) != _expected(
            operation, claim["payload"]
        ):
            return _reject("exact rational replay disagrees with the candidate")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "DIRECT_WITNESS",
            "coverage": "NOT_APPLICABLE",
            "detail": "direct standard-library rational replay accepted the result",
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return _reject(str(exc))
