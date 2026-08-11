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

from jacobian_checkers.bound_artifacts import (
    bound_request,
    valid_unscoped_unencoded_bindings,
)

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
    "geometry.polygon.triangulation.minimum_weight.compute",
    "geometry.points.compute.squared_distance",
    "geometry.points.compute.convex_hull",
    "geometry.segment.compute.midpoint",
    "geometry.segments.intersection.compute",
    "geometry.polygon.simple.decide",
    "geometry.polygon.point.classify",
    "geometry.triangle.compute.orientation",
    "geometry.triangle.compute.centroid",
}


def _triangulation_source(
    claim: object,
) -> tuple[list[tuple[Fraction, Fraction]], object]:
    if not isinstance(claim, dict) or set(claim) != {
        "polygon",
        "diagonal_weights",
        "objective",
    }:
        raise ValueError("triangulation source has an invalid shape")
    if claim["objective"] != "NON_HULL_DIAGONAL_WEIGHT_SUM":
        raise ValueError("triangulation objective is unsupported")
    points = _points(claim["polygon"])
    count = len(points)
    if not 4 <= count <= 32:
        raise ValueError("triangulation vertex count is unsupported")
    turns = [
        _cross(
            _subtract(points[(index + 1) % count], points[index]),
            _subtract(points[(index + 2) % count], points[index]),
        )
        for index in range(count)
    ]
    if any(turn <= 0 for turn in turns):
        raise ValueError("triangulation polygon is not strict CCW convex")
    return points, claim["diagonal_weights"]


def _triangulation_weights(
    raw_weights: object,
    count: int,
) -> dict[tuple[int, int], Fraction]:
    if not isinstance(raw_weights, list):
        raise ValueError("triangulation weights are malformed")
    weights: dict[tuple[int, int], Fraction] = {}
    for item in raw_weights:
        if not isinstance(item, dict) or set(item) != {"first", "second", "weight"}:
            raise ValueError("triangulation weight entry is malformed")
        first, second = item["first"], item["second"]
        if (
            type(first) is not int
            or type(second) is not int
            or not 0 <= first < second < count
        ):
            raise ValueError("triangulation weight endpoints are malformed")
        pair = (first, second)
        if pair in weights:
            raise ValueError("triangulation weight pair is duplicated")
        weight = _rational(item["weight"])
        if weight < 0:
            raise ValueError("triangulation weight is negative")
        weights[pair] = weight
    expected_pairs = {
        (first, second)
        for first in range(count)
        for second in range(first + 1, count)
        if second != first + 1 and (first, second) != (0, count - 1)
    }
    if set(weights) != expected_pairs or list(weights) != sorted(weights):
        raise ValueError("triangulation weights are incomplete or noncanonical")
    return weights


def _triangulation_dynamic_program(
    count: int,
    weights: dict[tuple[int, int], Fraction],
) -> tuple[
    dict[tuple[int, int], Fraction], dict[tuple[int, int], int], list[dict[str, object]]
]:

    def edge_weight(first: int, second: int) -> Fraction:
        pair = (first, second) if first < second else (second, first)
        return (
            Fraction()
            if second == first + 1 or pair == (0, count - 1)
            else weights[pair]
        )

    optimum = {(index, index + 1): Fraction() for index in range(count - 1)}
    splits: dict[tuple[int, int], int] = {}
    ledger: list[dict[str, object]] = []
    for span in range(2, count):
        for start in range(count - span):
            end = start + span
            value, pivot = min(
                (
                    optimum[start, candidate]
                    + optimum[candidate, end]
                    + edge_weight(start, candidate)
                    + edge_weight(candidate, end),
                    candidate,
                )
                for candidate in range(start + 1, end)
            )
            optimum[start, end] = value
            splits[start, end] = pivot
            ledger.append(
                {"start": start, "end": end, "split": pivot, "optimum": value}
            )
    return optimum, splits, ledger


def _triangulation_reconstruct(
    count: int,
    splits: dict[tuple[int, int], int],
) -> tuple[list[tuple[int, int, int]], set[tuple[int, int]]]:
    triangles: list[tuple[int, int, int]] = []
    diagonals: set[tuple[int, int]] = set()

    def reconstruct(start: int, end: int) -> None:
        if end == start + 1:
            return
        pivot = splits[start, end]
        triangles.append((start, pivot, end))
        for pair in ((start, pivot), (pivot, end)):
            ordered = pair if pair[0] < pair[1] else (pair[1], pair[0])
            if pair[1] != pair[0] + 1 and ordered != (0, count - 1):
                diagonals.add(ordered)
        reconstruct(start, pivot)
        reconstruct(pivot, end)

    reconstruct(0, count - 1)
    return triangles, diagonals


def _minimum_weight_triangulation(claim: object) -> dict[str, object]:
    points, raw_weights = _triangulation_source(claim)
    count = len(points)
    weights = _triangulation_weights(raw_weights, count)
    optimum, _splits, ledger = _triangulation_dynamic_program(count, weights)
    triangles, diagonals = _triangulation_reconstruct(count, _splits)
    return {
        "vertex_count": count,
        "diagonals": [
            {"first": first, "second": second, "weight": weights[first, second]}
            for first, second in sorted(diagonals)
        ],
        "triangles": [{"vertices": item} for item in sorted(triangles)],
        "split_table": ledger,
        "optimum": optimum[0, count - 1],
        "objective": "NON_HULL_DIAGONAL_WEIGHT_SUM",
        "tie_break": "LOWEST_SPLIT_INDEX",
        "exactness": "EXACT_RATIONAL",
        "verification": "UNVERIFIED",
    }


def _triangulation_candidate(payload: dict[str, object]) -> dict[str, object]:
    if set(payload) != {
        "vertex_count",
        "diagonals",
        "triangles",
        "split_table",
        "optimum",
        "objective",
        "tie_break",
        "exactness",
        "verification",
    }:
        raise ValueError("triangulation candidate has an invalid shape")
    result = dict(payload)
    result["optimum"] = _rational(payload["optimum"])
    diagonals = payload["diagonals"]
    if not isinstance(diagonals, list):
        raise ValueError("triangulation diagonals are malformed")
    result["diagonals"] = [
        {
            "first": item["first"],
            "second": item["second"],
            "weight": _rational(item["weight"]),
        }
        for item in diagonals
        if isinstance(item, dict) and set(item) == {"first", "second", "weight"}
    ]
    triangles = payload["triangles"]
    if not isinstance(triangles, list):
        raise ValueError("triangulation triangles are malformed")
    result["triangles"] = [
        {"vertices": tuple(item["vertices"])}
        for item in triangles
        if isinstance(item, dict)
        and set(item) == {"vertices"}
        and isinstance(item["vertices"], list)
    ]
    ledger = payload["split_table"]
    if not isinstance(ledger, list):
        raise ValueError("triangulation split table is malformed")
    result["split_table"] = [
        {
            "start": item["start"],
            "end": item["end"],
            "split": item["split"],
            "optimum": _rational(item["optimum"]),
        }
        for item in ledger
        if isinstance(item, dict) and set(item) == {"start", "end", "split", "optimum"}
    ]
    return result


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


def _polygon_classification_expected(claim: object) -> dict[str, object]:
    if not isinstance(claim, dict) or set(claim) != {"polygon", "point"}:
        raise ValueError("polygon classification source has an invalid shape")
    return _point_classification(
        _points(claim["polygon"]),
        _point(claim["point"]),
    )


def _triangle_expected(operation: str, claim: object) -> dict[str, object]:
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


def _expected(operation: str, claim: object) -> dict[str, object]:
    if operation == "geometry.polygon.triangulation.minimum_weight.compute":
        return _minimum_weight_triangulation(claim)
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
        return _polygon_classification_expected(claim)
    return _triangle_expected(operation, claim)


def _squared_distance_candidate(payload: dict[str, object]) -> dict[str, object]:
    if set(payload) != {"value"}:
        raise ValueError("squared-distance candidate has an invalid shape")
    return {"value": _rational(payload["value"])}


def _point_candidate(payload: dict[str, object]) -> dict[str, object]:
    if set(payload) != {"point"}:
        raise ValueError("point candidate has an invalid shape")
    return {"point": _point(payload["point"])}


def _segment_intersection_candidate(
    payload: dict[str, object],
) -> dict[str, object]:
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


def _polygon_witness(witness: object) -> dict[str, object]:
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
    return {
        "first_edge_index": witness["first_edge_index"],
        "second_edge_index": witness["second_edge_index"],
        "intersection": _candidate(
            witness["intersection"],
            "geometry.segments.intersection.compute",
        ),
    }


def _simple_polygon_candidate(payload: dict[str, object]) -> dict[str, object]:
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
    parsed_witness: dict[str, object] | None = (
        None if witness is None else _polygon_witness(witness)
    )
    return {
        "vertex_count": payload["vertex_count"],
        "is_simple": payload["is_simple"],
        "checked_edge_pairs": payload["checked_edge_pairs"],
        "witness": parsed_witness,
    }


def _point_classification_candidate(
    payload: dict[str, object],
) -> dict[str, object]:
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


def _orientation_candidate(payload: dict[str, object]) -> dict[str, object]:
    if set(payload) != {"orientation"} or type(payload["orientation"]) is not int:
        raise ValueError("orientation candidate has an invalid shape")
    return {"orientation": payload["orientation"]}


def _candidate(payload: object, operation: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("candidate has an invalid shape")
    if operation == "geometry.points.compute.squared_distance":
        return _squared_distance_candidate(payload)
    if operation == "geometry.points.compute.convex_hull":
        return {"points": _points(payload)}
    if operation in {
        "geometry.segment.compute.midpoint",
        "geometry.triangle.compute.centroid",
    }:
        return _point_candidate(payload)
    if operation == "geometry.segments.intersection.compute":
        return _segment_intersection_candidate(payload)
    if operation == "geometry.polygon.simple.decide":
        return _simple_polygon_candidate(payload)
    if operation == "geometry.polygon.point.classify":
        return _point_classification_candidate(payload)
    if operation == "geometry.polygon.triangulation.minimum_weight.compute":
        return _triangulation_candidate(payload)
    return _orientation_candidate(payload)


def _request_shape_detail(request: object) -> str | None:
    if not isinstance(request, dict) or set(request) != {
        "request_version",
        "claim",
        "candidate",
        "semantics",
        "scope",
        "witness",
        "expected_bindings",
    }:
        return "malformed checker request"
    if request["request_version"] != "1" or request["scope"] is not None:
        return "unsupported checker request"
    return None


def _artifact_metadata_detail(
    claim: dict[str, Any],
    candidate: dict[str, Any],
    semantics: dict[str, Any],
    witness: dict[str, Any],
    bindings: object,
) -> str | None:
    if not all(
        _valid_artifact(item) for item in (claim, candidate, semantics, witness)
    ):
        return "checker artifact metadata is malformed"
    if not valid_unscoped_unencoded_bindings(bindings):
        return "expected evidence bindings are malformed"
    return None


def _binding_match_detail(
    bindings: dict[str, Any],
    claim: dict[str, Any],
    candidate: dict[str, Any],
    semantics: dict[str, Any],
    witness: dict[str, Any],
) -> str | None:
    if (
        bindings["claim_digest"] != claim["object_digest"]
        or bindings["candidate_digest"] != candidate["object_digest"]
        or bindings["semantics_digest"] != semantics["object_digest"]
        or semantics["artifact_uri"] != claim["semantics_uri"]
    ):
        return "expected evidence bindings do not match artifacts"
    if (
        claim["semantics_uri"] != candidate["semantics_uri"]
        or claim["semantics_uri"] != witness["semantics_uri"]
        or candidate["parents"] != [claim["artifact_uri"]]
    ):
        return "candidate is not exactly bound to the geometry input"
    return None


def _artifact_digest_detail(
    artifacts: tuple[dict[str, Any], ...],
) -> str | None:
    for artifact in artifacts:
        if artifact["payload_digest"] != _sha256(_canonical_json(artifact["payload"])):
            return "artifact payload digest does not match"
    return None


def _witness_envelope_fields(
    envelope: object,
    bindings: dict[str, Any],
) -> bool:
    return (
        isinstance(envelope, dict)
        and set(envelope)
        == {
            "evidence_schema_version",
            "witness_format",
            "format_version",
            "role",
            "bindings",
            "payload",
        }
        and envelope["evidence_schema_version"] == "1"
        and envelope["witness_format"] == "geometry.exact_rational_result"
        and envelope["format_version"] == "1"
        and envelope["role"] == "SUPPORTS_CLAIM"
        and envelope["bindings"] == bindings
    )


def _witness_envelope_detail(
    envelope: object,
    bindings: dict[str, Any],
    witness: dict[str, Any],
    claim: dict[str, Any],
    candidate: dict[str, Any],
) -> str | None:
    if not _witness_envelope_fields(envelope, bindings):
        return "geometry witness envelope is malformed or rebound"
    if len(witness["parents"]) != 2 or set(witness["parents"]) != {
        claim["artifact_uri"],
        candidate["artifact_uri"],
    }:
        return "geometry witness envelope is malformed or rebound"
    return None


def _witness_payload_detail(
    payload: object,
    claim: dict[str, Any],
    candidate: dict[str, Any],
) -> str | None:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"operation_id", "input_uri", "result_uri"}
        or payload["operation_id"] not in _OPERATIONS
        or payload["input_uri"] != claim["artifact_uri"]
        or payload["result_uri"] != candidate["artifact_uri"]
    ):
        return "geometry witness payload is malformed or rebound"
    return None


def check_exact_geometry(request: dict[str, Any]) -> dict[str, Any]:
    """Accept a bound result exactly when direct rational replay agrees."""

    try:
        if request.get("request_version") == "2":
            operation = request.get("operation_id")
        else:
            witness = request.get("witness")
            envelope = witness.get("payload") if isinstance(witness, dict) else None
            witness_payload = (
                envelope.get("payload") if isinstance(envelope, dict) else None
            )
            operation = (
                witness_payload.get("operation_id")
                if isinstance(witness_payload, dict)
                else None
            )
        if operation not in _OPERATIONS:
            return _reject("unsupported exact geometry operation")
        claim, candidate = bound_request(
            request,
            operation_id=operation,
            witness_format="geometry.exact_rational_result",
        )
        if _candidate(candidate, operation) != _expected(operation, claim):
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
