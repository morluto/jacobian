"""Independent replay of simple-graph degree-sequence certificates."""

from __future__ import annotations

from typing import Any

_MAX_ORDER = 512


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "CHECKED_CERTIFICATE",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _decision(conclusion: str, detail: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": conclusion,
        "arithmetic": "EXACT_INTEGER",
        "method": "CHECKED_CERTIFICATE",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _sequence(value: object) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= _MAX_ORDER
        or any(
            not isinstance(degree, int) or isinstance(degree, bool) or degree < 0
            for degree in value
        )
    ):
        raise ValueError("invalid degree sequence")
    return tuple(value)


def _graph_degree_sequence(value: object) -> tuple[int, ...]:
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
        or not 1 <= len(vertices) <= _MAX_ORDER
        or any(not isinstance(vertex, str) or not vertex for vertex in vertices)
        or len(set(vertices)) != len(vertices)
        or not isinstance(edges, list)
    ):
        raise ValueError("malformed graph vertices or edges")
    degrees = dict.fromkeys(vertices, 0)
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or not all(isinstance(endpoint, str) for endpoint in edge)
            or edge[0] not in degrees
            or edge[1] not in degrees
            or edge[0] >= edge[1]
        ):
            raise ValueError("invalid simple undirected edge")
        normalized = (edge[0], edge[1])
        if normalized in seen:
            raise ValueError("duplicate graph edge")
        seen.add(normalized)
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    return tuple(sorted(degrees.values(), reverse=True))


def _expected_erdos_gallai(
    sequence: tuple[int, ...],
    k: int,
) -> tuple[int, int]:
    ordered = sorted(sequence, reverse=True)
    if not 1 <= k <= len(ordered):
        raise ValueError("Erdos-Gallai index is outside the sequence")
    lhs = sum(ordered[:k])
    rhs = k * (k - 1) + sum(min(degree, k) for degree in ordered[k:])
    return lhs, rhs


def _check_odd_sum_obstruction(
    obstruction: dict[str, Any],
    sequence: tuple[int, ...],
) -> str:
    if set(obstruction) != {"kind", "degree_sum"}:
        raise ValueError("malformed odd-sum obstruction")
    degree_sum = obstruction["degree_sum"]
    if degree_sum != sum(sequence) or degree_sum % 2 != 1:
        raise ValueError("odd-sum obstruction does not replay")
    return "odd degree sum proves the sequence is non-graphical"


def _check_max_degree_obstruction(
    obstruction: dict[str, Any],
    sequence: tuple[int, ...],
) -> str:
    if set(obstruction) != {"kind", "index", "degree", "order"}:
        raise ValueError("malformed maximum-degree obstruction")
    index = obstruction["index"]
    degree = obstruction["degree"]
    order = obstruction["order"]
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < len(sequence)
        or degree != sequence[index]
        or order != len(sequence)
        or degree < order
    ):
        raise ValueError("maximum-degree obstruction does not replay")
    return "a degree exceeds the simple-graph maximum"


def _check_erdos_gallai_obstruction(
    obstruction: dict[str, Any],
    sequence: tuple[int, ...],
) -> str:
    if set(obstruction) != {"kind", "k", "lhs", "rhs"}:
        raise ValueError("malformed Erdos-Gallai obstruction")
    k = obstruction["k"]
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError("invalid Erdos-Gallai index")
    lhs, rhs = _expected_erdos_gallai(sequence, k)
    if obstruction["lhs"] != lhs or obstruction["rhs"] != rhs or lhs <= rhs:
        raise ValueError("Erdos-Gallai obstruction does not replay")
    return f"Erdos-Gallai inequality fails at k={k}"


def _check_obstruction(
    sequence: tuple[int, ...],
    obstruction: object,
    method: object,
) -> str:
    if not isinstance(obstruction, dict):
        raise ValueError("missing non-graphical obstruction")
    kind = obstruction.get("kind")
    if kind == "ODD_SUM" and method == "ODD_SUM_OBSTRUCTION":
        return _check_odd_sum_obstruction(obstruction, sequence)
    if kind == "MAX_DEGREE" and method == "MAX_DEGREE_OBSTRUCTION":
        return _check_max_degree_obstruction(obstruction, sequence)
    if kind == "ERDOS_GALLAI" and method == "ERDOS_GALLAI_OBSTRUCTION":
        return _check_erdos_gallai_obstruction(obstruction, sequence)
    raise ValueError("obstruction kind differs from the replay method")


def _artifact_shape_detail(
    claim_artifact: object,
    candidate_artifact: object,
    certificate: object,
) -> str | None:
    if (
        not isinstance(claim_artifact, dict)
        or not isinstance(candidate_artifact, dict)
        or not isinstance(certificate, dict)
    ):
        return "degree-sequence replay artifacts are malformed"
    return None


def _claim_candidate_detail(claim: object, candidate: object) -> str | None:
    if (
        not isinstance(claim, dict)
        or claim.get("claim_schema_version") != "1"
        or claim.get("predicate") != "SIMPLE_GRAPH_DEGREE_SEQUENCE"
        or not isinstance(candidate, dict)
        or candidate.get("result_schema_version") != "1"
    ):
        return "unexpected degree-sequence claim or candidate"
    return None


def _certificate_detail(
    certificate: dict[str, Any],
    expected_bindings: object,
) -> str | None:
    if (
        certificate.get("evidence_schema_version") != "1"
        or certificate.get("certificate_type") != "graph.degree_sequence"
        or certificate.get("format_version") != "1"
        or certificate.get("bindings") != expected_bindings
    ):
        return "unexpected degree-sequence certificate or bindings"
    return None


def _artifact_and_claim_detail(
    claim_artifact: object,
    candidate_artifact: object,
    certificate: dict[str, Any],
    expected_bindings: object,
) -> str | None:
    detail = _artifact_shape_detail(claim_artifact, candidate_artifact, certificate)
    if detail is not None:
        return detail
    if not isinstance(claim_artifact, dict) or not isinstance(candidate_artifact, dict):
        return "degree-sequence replay artifacts are malformed"
    claim = claim_artifact.get("payload")
    candidate = candidate_artifact.get("payload")
    detail = _claim_candidate_detail(claim, candidate)
    if detail is not None:
        return detail
    return _certificate_detail(certificate, expected_bindings)


def _consistency_detail(
    candidate: dict[str, Any],
    payload: dict[str, Any],
    sequence: tuple[int, ...],
) -> str | None:
    if (
        _sequence(candidate.get("degree_sequence")) != sequence
        or _sequence(payload.get("degree_sequence")) != sequence
        or candidate.get("conclusion") != payload.get("conclusion")
        or candidate.get("graph_uri") != payload.get("graph_uri")
        or candidate.get("obstruction") != payload.get("obstruction")
    ):
        return "claim, candidate, and certificate differ"
    return None


def _graphical_detail(
    candidate: dict[str, Any],
    payload: dict[str, Any],
    sequence: tuple[int, ...],
) -> str | None:
    if (
        payload.get("method") != "EXACT_DEGREE_REPLAY"
        or candidate.get("obstruction") is not None
        or payload.get("obstruction") is not None
        or candidate.get("graph_uri") is None
    ):
        return "graphical certificate is malformed"
    realized = _graph_degree_sequence(candidate.get("graph"))
    if realized != tuple(sorted(sequence, reverse=True)):
        return "graph degrees differ from the claimed sequence"
    return None


def _non_graphical_detail(
    candidate: dict[str, Any],
    payload: dict[str, Any],
) -> str | None:
    if (
        candidate.get("graph_uri") is not None
        or candidate.get("graph") is not None
        or payload.get("graph_uri") is not None
    ):
        return "non-graphical certificate carries a graph"
    return None


def _conclusion_result(
    candidate: dict[str, Any],
    payload: dict[str, Any],
    sequence: tuple[int, ...],
) -> dict[str, Any] | None:
    conclusion = candidate.get("conclusion")
    if conclusion == "GRAPHICAL":
        detail = _graphical_detail(candidate, payload, sequence)
        if detail is not None:
            return _reject(detail)
        return _decision("TRUE", "simple graph degrees replayed exactly")
    if conclusion == "NON_GRAPHICAL":
        detail = _non_graphical_detail(candidate, payload)
        if detail is not None:
            return _reject(detail)
        obstruction_detail = _check_obstruction(
            sequence, candidate.get("obstruction"), payload.get("method")
        )
        return _decision("FALSE", obstruction_detail)
    return None


def check_degree_sequence(request: dict[str, Any]) -> dict[str, Any]:
    """Replay a graph realization or exact non-graphical obstruction."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim_artifact = request["claim"]
        candidate_artifact = request["candidate"]
        certificate = request["certificate"]["payload"]
        detail = _artifact_and_claim_detail(
            claim_artifact,
            candidate_artifact,
            certificate,
            request.get("expected_bindings"),
        )
        if detail is not None:
            return _reject(detail)
        claim = claim_artifact.get("payload")
        candidate = candidate_artifact.get("payload")
        payload = certificate.get("payload")
        if not isinstance(payload, dict):
            return _reject("degree-sequence replay payload is malformed")
        sequence = _sequence(claim.get("degree_sequence"))
        detail = _consistency_detail(candidate, payload, sequence)
        if detail is not None:
            return _reject(detail)
        result = _conclusion_result(candidate, payload, sequence)
        if result is not None:
            return result
        return _reject("unknown degree-sequence conclusion")
    except (KeyError, TypeError, ValueError):
        return _reject("degree-sequence replay failed")
