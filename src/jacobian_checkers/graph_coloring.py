"""Independent standard-library replay for graph-to-CNF coloring encodings."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_ARTIFACT_URI = re.compile(r"^artifact://sha256/[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_VERTICES = 32
_MAX_COLORS = 32


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "CHECKED_CERTIFICATE",
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


def _literal_key(literal: int) -> tuple[int, bool]:
    return abs(literal), literal > 0


def _parse_graph(value: object) -> tuple[list[str], list[tuple[str, str]]]:
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
        or len(vertices) > _MAX_VERTICES
        or any(not isinstance(vertex, str) or not vertex for vertex in vertices)
        or vertices != sorted(vertices)
        or len(vertices) != len(set(vertices))
        or not isinstance(edges, list)
    ):
        raise ValueError("graph vertices are not canonical")
    vertex_set = set(vertices)
    parsed_edges: list[tuple[str, str]] = []
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or not all(isinstance(endpoint, str) for endpoint in edge)
            or edge[0] >= edge[1]
            or edge[0] not in vertex_set
            or edge[1] not in vertex_set
        ):
            raise ValueError("graph edge is not canonical")
        parsed_edges.append((edge[0], edge[1]))
    if parsed_edges != sorted(parsed_edges) or len(parsed_edges) != len(
        set(parsed_edges)
    ):
        raise ValueError("graph edges are not canonical")
    return vertices, parsed_edges


def _expected_cnf(
    vertices: list[str],
    edges: list[tuple[str, str]],
    colors: int,
) -> dict[str, Any]:
    variable_names = [
        f"v{vertex:02d}_c{color:02d}"
        for vertex in range(len(vertices))
        for color in range(colors)
    ]

    def variable(vertex: int, color: int) -> int:
        return vertex * colors + color + 1

    raw_clauses: list[tuple[int, ...]] = []
    for vertex in range(len(vertices)):
        raw_clauses.append(tuple(variable(vertex, color) for color in range(colors)))
        for color_left in range(colors):
            for color_right in range(color_left + 1, colors):
                raw_clauses.append(
                    (-variable(vertex, color_left), -variable(vertex, color_right))
                )
    vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
    for edge_left, edge_right in edges:
        for color in range(colors):
            raw_clauses.append(
                (
                    -variable(vertex_index[edge_left], color),
                    -variable(vertex_index[edge_right], color),
                )
            )

    clauses = sorted(
        {tuple(sorted(clause, key=_literal_key)) for clause in raw_clauses},
        key=lambda clause: tuple(_literal_key(literal) for literal in clause),
    )
    variables = [
        {"id": index, "name": name}
        for index, name in enumerate(variable_names, start=1)
    ]
    variable_map = {
        "variable_map_format": "jacobian.sat.variable-map/v1",
        "variables": variables,
    }
    variable_map_digest = _sha256(_canonical_json(variable_map))
    clause_payload = [{"literals": list(clause)} for clause in clauses]
    dimacs_rows = [f"p cnf {len(variables)} {len(clause_payload)}\n"]
    for clause in clauses:
        prefix = " ".join(str(literal) for literal in clause)
        dimacs_rows.append(f"{prefix} 0\n" if prefix else "0\n")
    dimacs_digest = _sha256("".join(dimacs_rows).encode("ascii"))
    return {
        "cnf_schema_version": "1",
        "variables": variables,
        "clauses": clause_payload,
        "projection_format": "DIMACS-CNF",
        "projection_version": "jacobian.dimacs.cnf/v1",
        "variable_map_digest": variable_map_digest,
        "dimacs_digest": dimacs_digest,
    }


def _artifact_uri(value: object) -> bool:
    return isinstance(value, str) and _ARTIFACT_URI.fullmatch(value) is not None


def _request_shape_detail(request: object) -> str | None:
    if not isinstance(request, dict) or set(request) != {
        "request_version",
        "claim",
        "candidate",
        "scope",
        "certificate",
        "expected_bindings",
    }:
        return "malformed checker request"
    if request["request_version"] != "1":
        return "unsupported checker request"
    return None


def _artifact_and_payload_detail(
    claim_artifact: object,
    candidate_artifact: object,
    scope_artifact: object,
    certificate_artifact: object,
    expected_bindings: object,
) -> str | None:
    if (
        not isinstance(claim_artifact, dict)
        or not isinstance(candidate_artifact, dict)
        or not isinstance(scope_artifact, dict)
        or not isinstance(certificate_artifact, dict)
    ):
        return "graph-coloring artifacts are malformed"
    if not isinstance(expected_bindings, dict):
        return "graph-coloring evidence bindings are malformed"
    claim = claim_artifact["payload"]
    candidate = candidate_artifact["payload"]
    scope = scope_artifact["payload"]
    certificate = certificate_artifact["payload"]
    if not all(
        isinstance(item, dict) for item in (claim, candidate, scope, certificate)
    ):
        return "graph-coloring artifact payloads are malformed"
    return None


def _claim_and_colors_detail(claim: dict[str, Any]) -> str | None:
    if (
        claim.get("claim_schema_version") != "1"
        or claim.get("predicate") != "GRAPH_K_COLORABILITY_ENCODING"
        or set(claim) != {"claim_schema_version", "predicate", "graph", "colors"}
    ):
        return "unexpected graph-coloring claim"
    colors = claim["colors"]
    if type(colors) is not int or not 1 <= colors <= _MAX_COLORS:
        return "invalid graph-coloring color count"
    return None


def _candidate_detail(candidate: dict[str, Any]) -> str | None:
    if (
        set(candidate) != {"candidate_schema_version", "cnf_uri", "scope_uri"}
        or candidate.get("candidate_schema_version") != "1"
        or not _artifact_uri(candidate.get("cnf_uri"))
        or not _artifact_uri(candidate.get("scope_uri"))
    ):
        return "malformed graph-coloring candidate"
    return None


def _scope_detail(
    scope: dict[str, Any],
    claim_graph: object,
    colors: int,
) -> str | None:
    if (
        set(scope)
        != {
            "scope_schema_version",
            "graph",
            "colors",
            "cnf_uri",
            "cnf_object_digest",
            "cnf",
        }
        or scope.get("scope_schema_version") != "1"
        or scope.get("graph") != claim_graph
        or scope.get("colors") != colors
        or not _artifact_uri(scope.get("cnf_uri"))
        or not _DIGEST.fullmatch(str(scope.get("cnf_object_digest")))
    ):
        return "malformed graph-coloring scope"
    return None


def _scope_pointer_detail(
    scope: dict[str, Any],
    candidate: dict[str, Any],
    scope_artifact: dict[str, Any],
) -> str | None:
    if (
        scope["cnf_uri"] != candidate["cnf_uri"]
        or scope_artifact["artifact_uri"] != candidate["scope_uri"]
    ):
        return "graph-coloring candidate pointers are inconsistent"
    return None


def _candidate_and_scope_detail(
    candidate: dict[str, Any],
    scope: dict[str, Any],
    claim_graph: object,
    colors: int,
    scope_artifact: dict[str, Any],
) -> str | None:
    detail = _candidate_detail(candidate)
    if detail is not None:
        return detail
    detail = _scope_detail(scope, claim_graph, colors)
    if detail is not None:
        return detail
    return _scope_pointer_detail(scope, candidate, scope_artifact)


def _certificate_detail(
    certificate: dict[str, Any],
    expected_bindings: dict[str, Any],
    claim_artifact: dict[str, Any],
    candidate_artifact: dict[str, Any],
    scope_artifact: dict[str, Any],
) -> str | None:
    replay = certificate.get("payload")
    if (
        set(certificate)
        != {
            "evidence_schema_version",
            "certificate_type",
            "format_version",
            "bindings",
            "payload_digest",
            "payload",
        }
        or certificate.get("evidence_schema_version") != "1"
        or certificate.get("certificate_type") != "graph.coloring.encoding"
        or certificate.get("format_version") != "1"
        or certificate.get("bindings") != expected_bindings
        or not isinstance(replay, dict)
        or replay
        != {
            "method": "INDEPENDENT_GRAPH_COLORING_CNF_REPLAY",
            "claim_uri": claim_artifact["artifact_uri"],
            "candidate_uri": candidate_artifact["artifact_uri"],
            "scope_uri": scope_artifact["artifact_uri"],
        }
        or certificate.get("payload_digest") != _sha256(_canonical_json(replay))
    ):
        return "graph-coloring certificate is not exactly bound"
    if not _DIGEST.fullmatch(str(expected_bindings.get("claim_digest"))):
        return "graph-coloring evidence bindings are malformed"
    return None


def check_encoding(request: dict[str, Any]) -> dict[str, Any]:
    """Replay the graph-owned k-colorability CNF semantics independently."""

    try:
        detail = _request_shape_detail(request)
        if detail is not None:
            return _reject(detail)
        claim_artifact = request["claim"]
        candidate_artifact = request["candidate"]
        scope_artifact = request["scope"]
        certificate_artifact = request["certificate"]
        expected_bindings = request["expected_bindings"]
        detail = _artifact_and_payload_detail(
            claim_artifact,
            candidate_artifact,
            scope_artifact,
            certificate_artifact,
            expected_bindings,
        )
        if detail is not None:
            return _reject(detail)
        claim = claim_artifact["payload"]
        candidate = candidate_artifact["payload"]
        scope = scope_artifact["payload"]
        certificate = certificate_artifact["payload"]
        detail = _claim_and_colors_detail(claim)
        if detail is not None:
            return _reject(detail)
        vertices, edges = _parse_graph(claim["graph"])
        colors = claim["colors"]
        detail = _candidate_and_scope_detail(
            candidate, scope, claim["graph"], colors, scope_artifact
        )
        if detail is not None:
            return _reject(detail)
        expected_cnf = _expected_cnf(vertices, edges, colors)
        if scope["cnf"] != expected_cnf:
            return _reject("CNF does not encode the claimed graph colorability")
        detail = _certificate_detail(
            certificate,
            expected_bindings,
            claim_artifact,
            candidate_artifact,
            scope_artifact,
        )
        if detail is not None:
            return _reject(detail)
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_INTEGER",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "NOT_APPLICABLE",
            "detail": (
                "independent replay reconstructed every exactly-one and edge-separation "
                f"clause for {len(vertices)} vertices and {colors} colors"
            ),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed graph-coloring checker request")
