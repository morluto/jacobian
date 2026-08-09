"""Independent exact replay for finite simple-graph invariant certificates."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_MAX_GRAPH_ORDER = 256
_MAX_NEIGHBORHOOD_ORDER = 24


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "CHECKED_CERTIFICATE",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _parse_graph(
    value: object,
) -> tuple[tuple[str, ...], dict[str, set[str]]]:
    if not isinstance(value, dict) or set(value) != {
        "graph_schema_version",
        "vertices",
        "edges",
    }:
        raise ValueError("malformed graph payload")
    vertices = value["vertices"]
    edges = value["edges"]
    if (
        value["graph_schema_version"] != "1"
        or not isinstance(vertices, list)
        or not 0 <= len(vertices) <= _MAX_GRAPH_ORDER
        or any(not isinstance(vertex, str) or not vertex for vertex in vertices)
        or len(set(vertices)) != len(vertices)
        or not isinstance(edges, list)
    ):
        raise ValueError("malformed graph vertices or edges")
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in vertices}
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or not all(isinstance(endpoint, str) for endpoint in edge)
            or edge[0] not in adjacency
            or edge[1] not in adjacency
            or edge[0] >= edge[1]
        ):
            raise ValueError("invalid simple undirected edge")
        normalized = (edge[0], edge[1])
        if normalized in seen:
            raise ValueError("duplicate graph edge")
        seen.add(normalized)
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])
    return tuple(sorted(vertices)), adjacency


def _parse_rational(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("malformed rational")
    numerator = value["num"]
    denominator = value["den"]
    if (
        not isinstance(numerator, str)
        or not isinstance(denominator, str)
        or _INTEGER.fullmatch(numerator) is None
        or _INTEGER.fullmatch(denominator) is None
    ):
        raise ValueError("noncanonical rational")
    result = Fraction(int(numerator), int(denominator))
    if str(result.numerator) != numerator or str(result.denominator) != denominator:
        raise ValueError("noncanonical rational")
    return result


def _maximum_independent_set_size(
    neighborhood: tuple[str, ...],
    adjacency: dict[str, set[str]],
) -> int:
    order = len(neighborhood)
    if order > _MAX_NEIGHBORHOOD_ORDER:
        raise ValueError("neighborhood exceeds exact checker limit")
    full = (1 << order) - 1
    complement_adjacency: list[int] = []
    for index, vertex in enumerate(neighborhood):
        adjacent_mask = 0
        for other_index, other in enumerate(neighborhood):
            if other in adjacency[vertex]:
                adjacent_mask |= 1 << other_index
        complement_adjacency.append(full & ~(1 << index) & ~adjacent_mask)

    best = 0

    def expand(candidates: int, size: int) -> None:
        nonlocal best
        if not candidates:
            best = max(best, size)
            return
        while candidates:
            if size + candidates.bit_count() <= best:
                return
            selected = candidates & -candidates
            candidates ^= selected
            index = selected.bit_length() - 1
            expand(candidates & complement_adjacency[index], size + 1)

    expand(full, 0)
    return best


def check_neighborhood_independence(request: dict[str, Any]) -> dict[str, Any]:
    """Replay every local independence optimum with stdlib branch-and-bound."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim_artifact = request["claim"]
        candidate_artifact = request["candidate"]
        scope_artifact = request["scope"]
        certificate = request["certificate"]["payload"]
        if (
            not isinstance(claim_artifact, dict)
            or not isinstance(candidate_artifact, dict)
            or not isinstance(scope_artifact, dict)
            or not isinstance(certificate, dict)
        ):
            return _reject("graph invariant replay artifacts are malformed")
        claim = claim_artifact.get("payload")
        candidate = candidate_artifact.get("payload")
        source_graph = scope_artifact.get("payload")
        if (
            not isinstance(claim, dict)
            or claim.get("claim_schema_version") != "1"
            or claim.get("predicate") != "EXACT_NEIGHBORHOOD_INDEPENDENCE_PROFILE"
        ):
            return _reject("unexpected graph invariant claim")
        if (
            certificate.get("evidence_schema_version") != "1"
            or certificate.get("certificate_type") != "graph.neighborhood_independence"
            or certificate.get("format_version") != "1"
            or certificate.get("bindings") != request.get("expected_bindings")
        ):
            return _reject("unexpected graph invariant certificate or bindings")
        certificate_payload = certificate.get("payload")
        if (
            not isinstance(certificate_payload, dict)
            or set(certificate_payload)
            != {"method", "source_graph_uri", "invariant_uri"}
            or certificate_payload.get("method") != "EXACT_STDLIB_BRANCH_AND_BOUND"
        ):
            return _reject("graph invariant replay payload is malformed")
        source_graph_uri = scope_artifact.get("artifact_uri")
        invariant_uri = candidate_artifact.get("artifact_uri")
        if (
            claim.get("source_graph_uri") != source_graph_uri
            or certificate_payload.get("source_graph_uri") != source_graph_uri
            or certificate_payload.get("invariant_uri") != invariant_uri
        ):
            return _reject("graph invariant artifact identities do not match")
        vertices, adjacency = _parse_graph(source_graph)
        _replay_candidate(
            candidate,
            vertices=vertices,
            adjacency=adjacency,
            source_graph_uri=source_graph_uri,
        )
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_INTEGER",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "NOT_APPLICABLE",
            "detail": (
                "every neighborhood independence optimum replayed exactly with "
                "independent stdlib branch-and-bound"
            ),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _reject("malformed graph neighborhood-independence request")


def _validate_candidate_metadata(
    value: dict[str, Any],
    *,
    source_graph_uri: object,
    vertices: tuple[str, ...],
) -> list[Any]:
    records = value["records"]
    if (
        value["invariant_schema_version"] != "1"
        or value["graph_uri"] != source_graph_uri
        or value["maximum_neighborhood_order"] != _MAX_NEIGHBORHOOD_ORDER
        or value["backend"] != "networkx"
        or not isinstance(value["backend_version"], str)
        or not value["backend_version"]
        or not isinstance(records, list)
        or len(records) != len(vertices)
    ):
        raise ValueError("graph invariant metadata does not match the source")
    return records


def _validate_neighborhood_record(
    record: object,
    adjacency: dict[str, set[str]],
) -> tuple[str, int]:
    if not isinstance(record, dict) or set(record) != {
        "vertex",
        "neighborhood",
        "independent_set",
        "independence_number",
    }:
        raise ValueError("malformed neighborhood record")
    vertex = record["vertex"]
    neighborhood = record["neighborhood"]
    witness = record["independent_set"]
    optimum = record["independence_number"]
    if (
        not isinstance(vertex, str)
        or vertex not in adjacency
        or not isinstance(neighborhood, list)
        or neighborhood != sorted(adjacency[vertex])
        or len(neighborhood) > _MAX_NEIGHBORHOOD_ORDER
        or not isinstance(witness, list)
        or witness != sorted(set(witness))
        or not set(witness) <= set(neighborhood)
        or not isinstance(optimum, int)
        or isinstance(optimum, bool)
        or optimum != len(witness)
    ):
        raise ValueError("invalid neighborhood optimum record")
    if any(
        right in adjacency[left]
        for index, left in enumerate(witness)
        for right in witness[index + 1 :]
    ):
        raise ValueError("declared local witness is not independent")
    if optimum != _maximum_independent_set_size(tuple(neighborhood), adjacency):
        raise ValueError("declared local independence number is not optimal")
    return vertex, optimum


def _replay_candidate(
    value: object,
    *,
    vertices: tuple[str, ...],
    adjacency: dict[str, set[str]],
    source_graph_uri: object,
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "invariant_schema_version",
        "graph_uri",
        "records",
        "total",
        "average",
        "maximum_neighborhood_order",
        "backend",
        "backend_version",
    }:
        raise ValueError("malformed graph invariant candidate")
    records = _validate_candidate_metadata(
        value,
        source_graph_uri=source_graph_uri,
        vertices=vertices,
    )
    total = 0
    recorded_vertices: list[str] = []
    for record in records:
        vertex, optimum = _validate_neighborhood_record(record, adjacency)
        total += optimum
        recorded_vertices.append(vertex)
    if tuple(recorded_vertices) != vertices:
        raise ValueError("profile does not cover every graph vertex canonically")
    if (
        not isinstance(value["total"], int)
        or isinstance(value["total"], bool)
        or value["total"] != total
    ):
        raise ValueError("profile total is incorrect")
    expected_average = Fraction(total, len(vertices)) if vertices else Fraction(0)
    if _parse_rational(value["average"]) != expected_average:
        raise ValueError("profile average is incorrect")
