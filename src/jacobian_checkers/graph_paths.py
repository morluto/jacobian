"""Independent direct checkers for graph/path evidence.

This package intentionally imports no Jacobian search or plugin implementation.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def check_omitted_path(request: dict[str, Any]) -> dict[str, Any]:
    try:
        claim = request["claim"]
        candidate = request["candidate"]
        witness = request["witness"]["payload"]
        expected_bindings = request["expected_bindings"]
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        rejection = _validate_witness_header(
            witness,
            expected_bindings,
            witness_format="graph.omitted_path",
            role="DEFEATS_CANDIDATE",
            role_message="witness role does not defeat the candidate",
        )
        if rejection is not None:
            return rejection
        claim_payload = _claim_view(claim["payload"])
        rejection = _validate_path_claim(claim_payload)
        if rejection is not None:
            return rejection

        graph = candidate["payload"]
        parsed = _parse_omitted_path_graph(graph)
        if isinstance(parsed, dict):
            return parsed
        vertex_set, arc_set, source, terminals, intended_paths = parsed

        normalized = _normalize_intended_paths(
            intended_paths,
            vertex_set=vertex_set,
            arc_set=arc_set,
            source=source,
            terminals=terminals,
        )
        if isinstance(normalized, dict):
            return normalized
        normalized_intended = normalized

        path = witness.get("payload", {}).get("path")
        if not _is_valid_path(
            path,
            vertex_set=vertex_set,
            arc_set=arc_set,
            source=source,
            terminals=set(terminals),
        ):
            return _reject("witness path is not a legal simple source-terminal path")
        if tuple(path) in normalized_intended:
            return _reject("witness path is already in the intended family")
        return {
            "accepted": True,
            "conclusion": "FALSE",
            "arithmetic": "EXACT_INTEGER",
            "method": "DIRECT_WITNESS",
            "coverage": "NOT_APPLICABLE",
            "detail": "legal graph path is omitted from the intended family",
        }
    except (KeyError, TypeError, ValueError):
        return _reject("malformed checker request")


def check_path_enumeration(request: dict[str, Any]) -> dict[str, Any]:
    try:
        claim = request["claim"]
        candidate = request["candidate"]
        scope = request["scope"]
        certificate = request["certificate"]["payload"]
        expected_bindings = request["expected_bindings"]
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        rejection = _validate_enumeration_certificate(certificate, expected_bindings)
        if rejection is not None:
            return rejection
        claim_payload = _claim_view(claim["payload"])
        rejection = _validate_path_claim(claim_payload)
        if rejection is not None:
            return rejection
        scope_result = _validate_enumeration_scope(scope)
        if isinstance(scope_result, dict):
            return scope_result
        max_length = scope_result

        graph = candidate["payload"]
        parsed = _parse_graph(graph)
        if parsed is None:
            return _reject("candidate graph is malformed")
        vertex_set, arc_set, source, terminals, intended_paths = parsed
        if max_length < len(vertex_set):
            return _reject("scope can omit a legal simple path")

        computed_paths = _enumerate_simple_paths(
            vertex_set=vertex_set,
            arc_set=arc_set,
            source=source,
            terminals=terminals,
            max_length=max_length,
        )
        supplied_paths = certificate.get("payload", {}).get("actual_paths")
        supplied_result = _validate_supplied_paths(
            supplied_paths,
            vertex_set=vertex_set,
            arc_set=arc_set,
            source=source,
            terminals=terminals,
        )
        if isinstance(supplied_result, dict):
            return supplied_result
        supplied_tuples = supplied_result
        if set(supplied_tuples) != computed_paths:
            return _reject("certificate path table is incomplete or contains extras")
        conclusion = "TRUE" if computed_paths == intended_paths else "FALSE"
        return {
            "accepted": True,
            "conclusion": conclusion,
            "arithmetic": "EXACT_INTEGER",
            "method": "EXHAUSTIVE_FINITE",
            "coverage": "EXHAUSTIVE",
            "detail": "all simple source-terminal paths replayed",
        }
    except (KeyError, TypeError, ValueError):
        return _reject("malformed checker request")


def check_odd_cycle(request: dict[str, Any]) -> dict[str, Any]:
    """Check an odd-cycle counter-witness without graph-library imports."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim = _claim_view(request["claim"]["payload"])
        candidate = request["candidate"]["payload"]
        witness = request["witness"]["payload"]
        if claim.get("predicate") != "is_bipartite":
            return _reject("unsupported claim predicate")
        rejection = _validate_witness_header(
            witness,
            request["expected_bindings"],
            witness_format="graph.odd_cycle",
            role="DEFEATS_CANDIDATE",
            role_message="witness role does not defeat the candidate",
        )
        if rejection is not None:
            return rejection
        vertices = candidate.get("vertices")
        arcs = candidate.get("arcs")
        if (
            not isinstance(vertices, list)
            or not all(isinstance(vertex, str) for vertex in vertices)
            or len(vertices) != len(set(vertices))
            or not isinstance(arcs, list)
        ):
            return _reject("candidate graph is malformed")
        vertex_set = set(vertices)
        edge_set: set[frozenset[str]] = set()
        for edge in arcs:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or edge[0] not in vertex_set
                or edge[1] not in vertex_set
                or edge[0] == edge[1]
            ):
                return _reject("candidate edge is malformed")
            edge_set.add(frozenset((edge[0], edge[1])))
        cycle = witness.get("payload", {}).get("cycle")
        if (
            not isinstance(cycle, list)
            or len(cycle) < 3
            or len(cycle) % 2 != 1
            or len(cycle) != len(set(cycle))
            or any(vertex not in vertex_set for vertex in cycle)
        ):
            return _reject("witness is not a simple odd cycle")
        pairs = zip(cycle, (*cycle[1:], cycle[0]), strict=True)
        if any(frozenset(pair) not in edge_set for pair in pairs):
            return _reject("witness cycle uses a non-edge")
        return {
            "accepted": True,
            "conclusion": "FALSE",
            "arithmetic": "EXACT_INTEGER",
            "method": "DIRECT_WITNESS",
            "coverage": "NOT_APPLICABLE",
            "detail": "odd cycle refutes bipartiteness",
        }
    except (KeyError, TypeError, ValueError):
        return _reject("malformed checker request")


def check_two_coloring(request: dict[str, Any]) -> dict[str, Any]:
    """Check a complete two-color assignment as a positive witness."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim = _claim_view(request["claim"]["payload"])
        candidate = request["candidate"]["payload"]
        witness = request["witness"]["payload"]
        if claim.get("predicate") != "is_bipartite":
            return _reject("unsupported claim predicate")
        rejection = _validate_witness_header(
            witness,
            request["expected_bindings"],
            witness_format="graph.2coloring",
            role="SUPPORTS_CLAIM",
            role_message="witness role does not support the claim",
        )
        if rejection is not None:
            return rejection
        vertices = candidate.get("vertices")
        arcs = candidate.get("arcs")
        coloring = witness.get("payload", {}).get("coloring")
        if (
            not isinstance(vertices, list)
            or not all(isinstance(vertex, str) for vertex in vertices)
            or len(vertices) != len(set(vertices))
            or not isinstance(arcs, list)
            or not isinstance(coloring, dict)
            or set(coloring) != set(vertices)
            or any(
                type(color) is not int or color not in {0, 1}
                for color in coloring.values()
            )
        ):
            return _reject("candidate or coloring is malformed")
        for edge in arcs:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or edge[0] not in coloring
                or edge[1] not in coloring
                or coloring[edge[0]] == coloring[edge[1]]
            ):
                return _reject("an edge is not properly two-colored")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_INTEGER",
            "method": "DIRECT_WITNESS",
            "coverage": "NOT_APPLICABLE",
            "detail": "all vertices and edges satisfy the two-coloring",
        }
    except (KeyError, TypeError, ValueError):
        return _reject("malformed checker request")


def check_counterexample_preservation(
    request: dict[str, Any],
) -> dict[str, Any]:
    """Verify a reduced graph still falsifies the supported claim."""

    try:
        if request.get("request_version") != "1":
            return _reject_preservation("unsupported request version")
        claim = _claim_view(request["claim"]["payload"])
        reduced = request["reduced"]["payload"]
        evidence = request["preservation"]["payload"]
        rejection = _validate_preservation_evidence(
            evidence, request["expected_bindings"]
        )
        if rejection is not None:
            return rejection
        predicate = claim.get("predicate")
        if predicate == "intended_paths_complete":
            parsed = _parse_graph(reduced)
            if parsed is None or claim.get("simple") is not True:
                return _reject_preservation("reduced path candidate is malformed")
            vertices, arcs, source, terminals, intended = parsed
            actual = _enumerate_simple_paths(
                vertex_set=vertices,
                arc_set=arcs,
                source=source,
                terminals=terminals,
                max_length=len(vertices),
            )
            preserves = actual != intended
        elif predicate == "is_bipartite":
            parsed_graph = _parse_plain_graph(reduced)
            if parsed_graph is None:
                return _reject_preservation("reduced graph candidate is malformed")
            preserves = not _is_bipartite(*parsed_graph)
        else:
            return _reject_preservation("unsupported claim predicate")
        if not preserves:
            return _reject_preservation("reduced graph no longer falsifies the claim")
        return {
            "accepted": True,
            "conclusion": "FALSE",
            "arithmetic": "EXACT_INTEGER",
            "method": "EXHAUSTIVE_FINITE",
            "coverage": "EXHAUSTIVE",
            "detail": "reduced graph counterexample replayed exactly",
        }
    except (KeyError, TypeError, ValueError):
        return _reject_preservation("malformed checker request")


def _reject_preservation(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": detail,
    }


def _claim_view(payload: dict[str, Any]) -> dict[str, Any]:
    predicate = payload.get("predicate")
    if not isinstance(predicate, dict):
        return payload
    parameters = predicate.get("parameters", {})
    bounds = payload.get("bounds", {})
    return {
        "predicate": predicate.get("name"),
        **(parameters if isinstance(parameters, dict) else {}),
        **(bounds if isinstance(bounds, dict) else {}),
    }


def _parse_plain_graph(
    graph: dict[str, Any],
) -> tuple[set[str], set[tuple[str, str]]] | None:
    vertices = graph.get("vertices")
    arcs = graph.get("arcs")
    if (
        not isinstance(vertices, list)
        or not all(isinstance(vertex, str) for vertex in vertices)
        or len(vertices) != len(set(vertices))
        or not isinstance(arcs, list)
    ):
        return None
    vertex_set = set(vertices)
    edges: set[tuple[str, str]] = set()
    for edge in arcs:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or edge[0] not in vertex_set
            or edge[1] not in vertex_set
            or edge[0] == edge[1]
        ):
            return None
        edges.add((edge[0], edge[1]))
    return vertex_set, edges


def _is_bipartite(
    vertices: set[str],
    edges: set[tuple[str, str]],
) -> bool:
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in vertices}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    colors: dict[str, int] = {}
    for start in sorted(vertices):
        if start in colors:
            continue
        colors[start] = 0
        queue = [start]
        while queue:
            vertex = queue.pop()
            for neighbor in adjacency[vertex]:
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[vertex]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[vertex]:
                    return False
    return True


def _parse_graph(
    graph: dict[str, Any],
) -> (
    tuple[
        set[str],
        set[tuple[str, str]],
        str,
        set[str],
        set[tuple[str, ...]],
    ]
    | None
):
    vertices = graph.get("vertices")
    arcs = graph.get("arcs")
    source = graph.get("source")
    terminals = graph.get("terminals")
    intended = graph.get("intended_paths")
    if (
        not isinstance(vertices, list)
        or not all(isinstance(vertex, str) for vertex in vertices)
        or len(vertices) != len(set(vertices))
    ):
        return None
    vertex_set = set(vertices)
    if not isinstance(source, str) or source not in vertex_set:
        return None
    if (
        not isinstance(terminals, list)
        or not terminals
        or not all(terminal in vertex_set for terminal in terminals)
    ):
        return None
    if not isinstance(arcs, list):
        return None
    arc_pairs = _parse_graph_arcs(arcs, vertex_set)
    if arc_pairs is None:
        return None
    if len(arc_pairs) != len(set(arc_pairs)) or not isinstance(intended, list):
        return None
    arc_set = set(arc_pairs)
    terminal_set: set[str] = set(terminals)
    if source in terminal_set:
        return None
    intended_set = _parse_intended_paths(
        intended,
        vertex_set=vertex_set,
        arc_set=arc_set,
        source=source,
        terminal_set=terminal_set,
    )
    if intended_set is None:
        return None
    return vertex_set, arc_set, source, terminal_set, intended_set


def _enumerate_simple_paths(
    *,
    vertex_set: set[str],
    arc_set: set[tuple[str, str]],
    source: str,
    terminals: set[str],
    max_length: int,
) -> set[tuple[str, ...]]:
    adjacency: dict[str, list[str]] = {vertex: [] for vertex in vertex_set}
    for left, right in arc_set:
        adjacency[left].append(right)
    for neighbors in adjacency.values():
        neighbors.sort()
    paths: set[tuple[str, ...]] = set()

    def visit(vertex: str, path: list[str]) -> None:
        if vertex in terminals:
            paths.add(tuple(path))
        if len(path) >= max_length:
            return
        for neighbor in adjacency[vertex]:
            if neighbor not in path:
                visit(neighbor, [*path, neighbor])

    visit(source, [source])
    return paths


def _is_valid_path(
    path: Any,
    *,
    vertex_set: set[str],
    arc_set: set[tuple[str, str]],
    source: str,
    terminals: set[str],
) -> bool:
    if (
        not isinstance(path, list)
        or len(path) < 2
        or not all(isinstance(vertex, str) for vertex in path)
        or path[0] != source
        or path[-1] not in terminals
        or len(path) != len(set(path))
        or any(vertex not in vertex_set for vertex in path)
    ):
        return False
    return all((left, right) in arc_set for left, right in pairwise(path))


def _validate_witness_header(
    witness: dict[str, Any],
    expected_bindings: object,
    *,
    witness_format: str,
    role: str,
    role_message: str,
) -> dict[str, Any] | None:
    if witness.get("witness_format") != witness_format:
        return _reject("unexpected witness format")
    if witness.get("format_version") != "1":
        return _reject("unsupported witness format version")
    if witness.get("role") != role:
        return _reject(role_message)
    if witness.get("bindings") != expected_bindings:
        return _reject("witness bindings do not match the request")
    return None


def _validate_path_claim(claim_payload: dict[str, Any]) -> dict[str, Any] | None:
    if claim_payload.get("predicate") != "intended_paths_complete":
        return _reject("unsupported claim predicate")
    if claim_payload.get("simple") is not True:
        return _reject("checker supports simple-path semantics only")
    return None


def _parse_omitted_path_graph(
    graph: dict[str, Any],
) -> tuple[set[str], set[tuple[str, str]], str, list[str], Any] | dict[str, Any]:
    vertices = graph.get("vertices")
    arcs = graph.get("arcs")
    source = graph.get("source")
    terminals = graph.get("terminals")
    intended_paths = graph.get("intended_paths")
    if (
        not isinstance(vertices, list)
        or not all(isinstance(vertex, str) for vertex in vertices)
        or len(vertices) != len(set(vertices))
    ):
        return _reject("vertices must be a unique string list")
    vertex_set = set(vertices)
    if not isinstance(source, str) or source not in vertex_set:
        return _reject("source is not a graph vertex")
    if (
        not isinstance(terminals, list)
        or not terminals
        or not all(terminal in vertex_set for terminal in terminals)
    ):
        return _reject("terminals are invalid")
    if not isinstance(arcs, list):
        return _reject("arcs must be a list")
    arc_pairs: list[tuple[str, str]] = []
    for arc in arcs:
        if (
            not isinstance(arc, list)
            or len(arc) != 2
            or arc[0] not in vertex_set
            or arc[1] not in vertex_set
        ):
            return _reject("arc is malformed or out of domain")
        arc_pairs.append((arc[0], arc[1]))
    if len(arc_pairs) != len(set(arc_pairs)):
        return _reject("duplicate arcs are not allowed")
    arc_set = set(arc_pairs)
    terminal_list = [terminal for terminal in terminals if isinstance(terminal, str)]
    return vertex_set, arc_set, source, terminal_list, intended_paths


def _normalize_intended_paths(
    intended_paths: object,
    *,
    vertex_set: set[str],
    arc_set: set[tuple[str, str]],
    source: str,
    terminals: list[str],
) -> set[tuple[str, ...]] | dict[str, Any]:
    if not isinstance(intended_paths, list):
        return _reject("intended_paths must be a list")
    normalized_intended: set[tuple[str, ...]] = set()
    for intended in intended_paths:
        if not _is_valid_path(
            intended,
            vertex_set=vertex_set,
            arc_set=arc_set,
            source=source,
            terminals=set(terminals),
        ):
            return _reject("an intended path is invalid")
        normalized_intended.add(tuple(intended))
    return normalized_intended


def _validate_enumeration_certificate(
    certificate: dict[str, Any],
    expected_bindings: object,
) -> dict[str, Any] | None:
    if certificate.get("certificate_type") != "graph.path_enumeration":
        return _reject("unexpected certificate format")
    if certificate.get("format_version") != "1":
        return _reject("unsupported certificate format version")
    if certificate.get("bindings") != expected_bindings:
        return _reject("certificate bindings do not match the request")
    return None


def _validate_enumeration_scope(scope: object) -> int | dict[str, Any]:
    if not isinstance(scope, dict):
        return _reject("path enumeration requires a bound scope")
    scope_payload = scope["payload"]
    if scope_payload.get("simple") is not True:
        return _reject("scope must request simple paths")
    max_length = scope_payload.get("max_length")
    if not isinstance(max_length, int) or isinstance(max_length, bool):
        return _reject("scope max_length must be an integer")
    return max_length


def _validate_supplied_paths(
    supplied_paths: object,
    *,
    vertex_set: set[str],
    arc_set: set[tuple[str, str]],
    source: str,
    terminals: set[str],
) -> list[tuple[str, ...]] | dict[str, Any]:
    if not isinstance(supplied_paths, list):
        return _reject("certificate path table is missing")
    supplied_tuples: list[tuple[str, ...]] = []
    for path in supplied_paths:
        if not _is_valid_path(
            path,
            vertex_set=vertex_set,
            arc_set=arc_set,
            source=source,
            terminals=terminals,
        ):
            return _reject("certificate contains an invalid path")
        supplied_tuples.append(tuple(path))
    if len(supplied_tuples) != len(set(supplied_tuples)):
        return _reject("certificate path table contains duplicates")
    return supplied_tuples


def _validate_preservation_evidence(
    evidence: dict[str, Any],
    expected_bindings: object,
) -> dict[str, Any] | None:
    if evidence.get("preservation_format") != ("graph.counterexample_preservation"):
        return _reject_preservation("unexpected preservation format")
    if evidence.get("format_version") != "1":
        return _reject_preservation("unsupported preservation version")
    if evidence.get("bindings") != expected_bindings:
        return _reject_preservation("preservation bindings do not match the request")
    return None


def _parse_graph_arcs(
    arcs: object,
    vertex_set: set[str],
) -> list[tuple[str, str]] | None:
    if not isinstance(arcs, list):
        return None
    arc_pairs: list[tuple[str, str]] = []
    for arc in arcs:
        if (
            not isinstance(arc, list)
            or len(arc) != 2
            or arc[0] not in vertex_set
            or arc[1] not in vertex_set
        ):
            return None
        arc_pairs.append((arc[0], arc[1]))
    return arc_pairs


def _parse_intended_paths(
    intended: object,
    *,
    vertex_set: set[str],
    arc_set: set[tuple[str, str]],
    source: str,
    terminal_set: set[str],
) -> set[tuple[str, ...]] | None:
    if not isinstance(intended, list):
        return None
    intended_set: set[tuple[str, ...]] = set()
    for path in intended:
        if not _is_valid_path(
            path,
            vertex_set=vertex_set,
            arc_set=arc_set,
            source=source,
            terminals=terminal_set,
        ):
            return None
        intended_set.add(tuple(path))
    return intended_set
