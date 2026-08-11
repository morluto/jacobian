"""Search-side directed graph/path reference plugin.

Implements the maintained graph-path reference scenarios:
- PATH-CLOSURE-001: intended source-terminal path family is incomplete.
- GRAPH-BIP-001: a triangle plus isolated vertices is not bipartite.

All outputs are unverified search results; checkers replay evidence separately.
"""

from __future__ import annotations

import time
from copy import deepcopy
from itertools import pairwise, permutations
from typing import Any

import networkx as nx
from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.contracts.plugin_graphs import (
    GraphCanonicalizeRequest,
    GraphEnumerationRequest,
    GraphPathCandidate,
    GraphPathCapabilityRequest,
    GraphPathClaim,
    GraphPathReductionRequest,
)


def _path_coverage(claim: GraphPathClaim, candidate: GraphPathCandidate) -> str:
    return (
        "EXHAUSTIVE"
        if _max_path_length(claim, candidate) >= len(candidate.vertices)
        else "BOUNDED"
    )


# ---------------------------------------------------------------------------
# Graph construction and search
# ---------------------------------------------------------------------------


def _as_digraph(candidate: GraphPathCandidate) -> nx.DiGraph[str]:
    g: nx.DiGraph[str] = nx.DiGraph()
    g.add_nodes_from(candidate.vertices)
    g.add_edges_from(candidate.arcs)
    return g


def _max_path_length(claim: GraphPathClaim, candidate: GraphPathCandidate) -> int:
    value = claim.max_path_length
    if value is None:
        return len(candidate.vertices)
    return value


def _enumerate_simple_paths(
    candidate: GraphPathCandidate, max_length: int
) -> set[tuple[str, ...]]:
    """Enumerate all simple source-terminal paths with at most max_length vertices."""
    g = _as_digraph(candidate)
    if candidate.source is None or not candidate.terminals:
        return set()
    cutoff = max(1, max_length - 1)
    paths: set[tuple[str, ...]] = set()
    for target in candidate.terminals:
        for path in nx.all_simple_paths(g, candidate.source, target, cutoff=cutoff):
            paths.add(tuple(path))
    return paths


def _find_omitted_path(
    candidate: GraphPathCandidate, max_length: int
) -> list[str] | None:
    actual = _enumerate_simple_paths(candidate, max_length)
    intended = set(candidate.intended_paths or ())
    for path in sorted(actual):
        if path not in intended:
            return list(path)
    return None


def _find_odd_cycle(candidate: GraphPathCandidate) -> list[str] | None:
    """Return an odd cycle in the underlying graph, or None if none exists."""
    g = _as_digraph(candidate)
    ug = g.to_undirected()
    if nx.is_bipartite(ug):
        return None
    for cycle in nx.cycle_basis(ug):
        if len(cycle) % 2 == 1:
            return cycle
    return None


def _two_coloring(candidate: GraphPathCandidate) -> dict[str, int] | None:
    ug = _as_digraph(candidate).to_undirected()
    if not nx.is_bipartite(ug):
        return None
    coloring: dict[str, int] = {}
    for component in nx.connected_components(ug):
        subgraph = ug.subgraph(component)
        c = nx.bipartite.color(subgraph)
        for v, col in c.items():
            coloring[v] = col
    return coloring


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def _now_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _ok(start: float) -> dict[str, Any]:
    return {
        "execution": {"status": "COMPLETED", "runtime_ms": _now_ms(start)},
        "input": {"status": "ACCEPTED", "errors": [], "warnings": []},
        "verified": False,
    }


def _rejected(errors: list[str], start: float) -> dict[str, Any]:
    return {
        "execution": {"status": "COMPLETED", "runtime_ms": _now_ms(start)},
        "input": {"status": "REJECTED", "errors": errors, "warnings": []},
        "verified": False,
    }


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def _evaluate_typed(
    claim: GraphPathClaim,
    candidate: GraphPathCandidate,
) -> dict[str, Any]:
    max_len = _max_path_length(claim, candidate)
    if claim.predicate == "intended_paths_complete":
        actual = _enumerate_simple_paths(candidate, max_len)
        intended = set(candidate.intended_paths or ())
        omitted = [list(path) for path in sorted(actual) if path not in intended]
        result: dict[str, Any] = {
            "objective": {
                "name": "path_family_complete",
                "value": not omitted,
                "num_actual": len(actual),
                "num_intended": len(intended),
            },
            "proposed_witness": None,
            "coverage": _path_coverage(claim, candidate),
            "arithmetic": "EXACT_INTEGER",
            "detail": (
                "intended family complete"
                if not omitted
                else f"{len(omitted)} omitted path(s)"
            ),
        }
        if omitted:
            result["proposed_witness"] = {
                "witness_format": "graph.omitted_path",
                "format_version": "1",
                "role": "DEFEATS_CANDIDATE",
                "payload": {"path": omitted[0]},
            }
        return result

    if claim.predicate == "is_bipartite":
        odd_cycle = _find_odd_cycle(candidate)
        if odd_cycle:
            return {
                "objective": {"name": "is_bipartite", "value": False},
                "proposed_witness": {
                    "witness_format": "graph.odd_cycle",
                    "format_version": "1",
                    "role": "DEFEATS_CANDIDATE",
                    "payload": {"cycle": odd_cycle},
                },
                "coverage": "EXHAUSTIVE",
                "arithmetic": "EXACT_INTEGER",
                "detail": f"odd cycle of length {len(odd_cycle)}",
            }
        coloring = _two_coloring(candidate)
        return {
            "objective": {"name": "is_bipartite", "value": True},
            "proposed_witness": {
                "witness_format": "graph.2coloring",
                "format_version": "1",
                "role": "SUPPORTS_CLAIM",
                "payload": {"coloring": coloring},
            },
            "coverage": "EXHAUSTIVE",
            "arithmetic": "EXACT_INTEGER",
            "detail": "graph is bipartite",
        }

    raise ValueError("unsupported graph claim predicate")


def _graph_capability_request(request: dict[str, Any]) -> GraphPathCapabilityRequest:
    try:
        return GraphPathCapabilityRequest.model_validate(request)
    except ValidationError as exc:
        raise ValueError(
            "graph capability request does not match its contract"
        ) from exc


def _graph_reduction_request(request: dict[str, Any]) -> GraphPathReductionRequest:
    try:
        return GraphPathReductionRequest.model_validate(request)
    except ValidationError as exc:
        raise ValueError("graph reduction request does not match its contract") from exc


def evaluate_capability(request: dict[str, Any]) -> dict[str, Any]:
    """Return one graph evaluation in the generic evaluator contract."""

    try:
        selected = GraphPathCapabilityRequest.model_validate(request)
    except ValidationError as exc:
        raise ValueError(
            "graph evaluation request does not match its contract"
        ) from exc
    result = _evaluate_typed(selected.claim, selected.candidate)
    objective = result["objective"]
    coverage = result["coverage"]
    proposed = result.get("proposed_witness")
    return {
        "response_version": "1",
        "conclusion": "TRUE" if objective["value"] else "FALSE",
        "arithmetic": result["arithmetic"],
        "method": (
            "EXHAUSTIVE_FINITE" if coverage == "EXHAUSTIVE" else "BOUNDED_SEARCH"
        ),
        "coverage": coverage,
        "objectives": {objective["name"]: objective["value"]},
        "features": {
            key: str(value)
            for key, value in objective.items()
            if key not in {"name", "value"}
        },
        "failure_classifications": (
            ["omitted_semantic_object"]
            if proposed is not None
            and proposed["witness_format"] == "graph.omitted_path"
            else []
        ),
        "detail": result["detail"],
    }


def _find_path_witness_typed(
    claim: GraphPathClaim,
    candidate: GraphPathCandidate,
    role: str,
) -> dict[str, Any]:
    max_len = _max_path_length(claim, candidate)

    if role == "DEFEATS_CANDIDATE":
        omitted = _find_omitted_path(candidate, max_len)
        if omitted:
            return {
                "status": "FOUND",
                "witness": {"path": omitted},
                "witness_format": "graph.omitted_path",
                "format_version": "1",
                "role": "DEFEATS_CANDIDATE",
                "coverage": "NOT_APPLICABLE",
                "arithmetic": "EXACT_INTEGER",
                "detail": "legal source-terminal path omitted from intended family",
            }
        return {
            "status": "SEARCH_EXHAUSTED",
            "witness": None,
            "coverage": _path_coverage(claim, candidate),
            "arithmetic": "EXACT_INTEGER",
            "detail": "no omitted path found within bounded scope",
        }

    if role == "SUPPORTS_CLAIM":
        actual = _enumerate_simple_paths(candidate, max_len)
        intended = set(candidate.intended_paths or ())
        return {
            "status": "SEARCH_EXHAUSTED"
            if actual == intended
            else "NOT_FOUND_WITHIN_SCOPE",
            "witness": None,
            "coverage": _path_coverage(claim, candidate),
            "arithmetic": "EXACT_INTEGER",
            "detail": (
                "intended family matches actual family"
                if actual == intended
                else "intended family does not match actual family"
            ),
        }

    raise ValueError("unsupported witness role")


def _find_bipartite_witness_typed(
    candidate: GraphPathCandidate,
    role: str,
) -> dict[str, Any]:
    if role == "DEFEATS_CANDIDATE":
        odd_cycle = _find_odd_cycle(candidate)
        if odd_cycle:
            return {
                "status": "FOUND",
                "witness": {"cycle": odd_cycle},
                "witness_format": "graph.odd_cycle",
                "format_version": "1",
                "role": "DEFEATS_CANDIDATE",
                "coverage": "EXHAUSTIVE",
                "arithmetic": "EXACT_INTEGER",
                "detail": f"odd cycle of length {len(odd_cycle)}",
            }
        return {
            "status": "SEARCH_EXHAUSTED",
            "witness": None,
            "coverage": "EXHAUSTIVE",
            "arithmetic": "EXACT_INTEGER",
            "detail": "no odd cycle found",
        }

    if role == "SUPPORTS_CLAIM":
        coloring = _two_coloring(candidate)
        return {
            "status": "FOUND" if coloring else "SEARCH_EXHAUSTED",
            "witness": {"coloring": coloring} if coloring else None,
            **(
                {
                    "witness_format": "graph.2coloring",
                    "format_version": "1",
                    "role": "SUPPORTS_CLAIM",
                }
                if coloring
                else {}
            ),
            "coverage": "EXHAUSTIVE",
            "arithmetic": "EXACT_INTEGER",
            "detail": "2-coloring witness" if coloring else "graph is not bipartite",
        }

    raise ValueError("unsupported witness role")


def _find_witness_typed(
    claim: GraphPathClaim,
    candidate: GraphPathCandidate,
    role: str,
) -> dict[str, Any]:
    if claim.predicate == "intended_paths_complete":
        return _find_path_witness_typed(claim, candidate, role)
    if claim.predicate == "is_bipartite":
        return _find_bipartite_witness_typed(candidate, role)
    raise ValueError("unsupported graph claim predicate")


def find_witness_capability(request: dict[str, Any]) -> dict[str, Any]:
    """Return graph witness search in the generic oracle contract."""

    selected = _graph_capability_request(request)
    return _find_witness_typed(
        selected.claim,
        selected.candidate,
        selected.witness_role,
    )


def materialize(request: dict[str, Any]) -> dict[str, Any]:
    """Materialize a complete bounded family for a graph claim."""
    start = time.monotonic()
    try:
        selected = GraphPathCapabilityRequest.model_validate(request)
    except ValidationError:
        return _rejected(
            ["graph materialization request does not match its contract"], start
        )
    predicate = selected.claim.predicate
    if predicate != "intended_paths_complete":
        return _rejected(["materialize supports intended_paths_complete only"], start)

    max_len = _max_path_length(selected.claim, selected.candidate)
    paths = sorted(_enumerate_simple_paths(selected.candidate, max_len))

    response = _ok(start)
    response["family"] = [list(p) for p in paths]
    response["coverage"] = _path_coverage(selected.claim, selected.candidate)
    response["arithmetic"] = "EXACT_INTEGER"
    response["detail"] = "all simple source-terminal paths within bound"
    return response


def _reductions_typed(
    selected: GraphPathReductionRequest,
    *,
    start: float,
) -> dict[str, Any]:
    predicate = selected.claim.predicate
    max_len = _max_path_length(selected.claim, selected.target)

    protected_vertices: set[str] = set()
    protected_arcs: set[tuple[str, str]] = set()

    if predicate == "intended_paths_complete":
        omitted = _find_omitted_path(selected.target, max_len)
        if omitted is None:
            response = _ok(start)
            response["reductions"] = []
            response["detail"] = "no counter-witness to preserve"
            return response
        protected_vertices = set(omitted)
        protected_arcs = set(pairwise(omitted))
    elif predicate == "is_bipartite":
        cycle = _find_odd_cycle(selected.target)
        if cycle is None:
            response = _ok(start)
            response["reductions"] = []
            response["detail"] = "no odd cycle to preserve"
            return response
        protected_vertices = set(cycle)
        cycle_edges = set(pairwise(cycle)) | {(cycle[-1], cycle[0])}
        protected_arcs = cycle_edges | {
            (target_vertex, source_vertex)
            for source_vertex, target_vertex in cycle_edges
        }
    else:
        return _rejected(["unsupported predicate for reductions"], start)

    vertices = selected.target.vertices
    arcs = selected.target.arcs
    current_vertices = len(vertices)
    current_edges = len(arcs)

    proposed: list[dict[str, Any]] = []
    for v in vertices:
        if v not in protected_vertices:
            incident = sum(1 for a in arcs if a[0] == v or a[1] == v)
            proposed.append(
                {
                    "reduction_kind": "delete_vertex",
                    "vertex": v,
                    "objectives": {
                        "vertices": current_vertices - 1,
                        "edges": current_edges - incident,
                    },
                }
            )

    for arc in arcs:
        pair = (arc[0], arc[1])
        if pair not in protected_arcs:
            proposed.append(
                {
                    "reduction_kind": "delete_edge",
                    "edge": [arc[0], arc[1]],
                    "objectives": {
                        "vertices": current_vertices,
                        "edges": current_edges - 1,
                    },
                }
            )

    proposed.sort(key=lambda r: (r["objectives"]["vertices"], r["objectives"]["edges"]))

    response = _ok(start)
    response["reductions"] = proposed
    response["coverage"] = "BOUNDED"
    response["arithmetic"] = "EXACT_INTEGER"
    response["detail"] = f"{len(proposed)} reduction(s) preserve a counter-witness"
    return response


def reductions_capability(request: dict[str, Any]) -> dict[str, Any]:
    """Return complete reduced payloads for the generic shrinker."""

    selected = _graph_reduction_request(request)
    target = selected.target.model_dump(mode="json", exclude_none=True)
    response = _reductions_typed(selected, start=time.monotonic())
    if response["input"]["status"] != "ACCEPTED":
        raise ValueError("; ".join(response["input"]["errors"]))
    requested = set(selected.reducers)
    objective_names = tuple(selected.objectives)
    proposals: list[dict[str, Any]] = []
    for operation in response["reductions"]:
        reducer = operation["reduction_kind"]
        if requested and reducer not in requested:
            continue
        payload = deepcopy(target)
        if reducer == "delete_vertex":
            vertex = operation["vertex"]
            payload["vertices"] = [
                item for item in payload["vertices"] if item != vertex
            ]
            payload["arcs"] = [edge for edge in payload["arcs"] if vertex not in edge]
            if "terminals" in payload:
                payload["terminals"] = [
                    item for item in payload["terminals"] if item != vertex
                ]
            if "intended_paths" in payload:
                payload["intended_paths"] = [
                    path for path in payload["intended_paths"] if vertex not in path
                ]
        elif reducer == "delete_edge":
            edge = operation["edge"]
            payload["arcs"] = [item for item in payload["arcs"] if item != edge]
            if "intended_paths" in payload:
                pair = tuple(edge)
                payload["intended_paths"] = [
                    path
                    for path in payload["intended_paths"]
                    if pair not in set(pairwise(path))
                ]
        else:
            continue
        proposals.append(
            {
                "reducer": reducer,
                "payload": payload,
                "objectives": {
                    name: operation["objectives"][name]
                    for name in objective_names
                    if name in operation["objectives"]
                },
            }
        )
    current = {
        "vertices": len(target.get("vertices", [])),
        "edges": len(target.get("arcs", [])),
    }
    return {
        "response_version": "1",
        "current_objectives": {
            name: current[name] for name in objective_names if name in current
        },
        "reductions": proposals,
        "detail": response["detail"],
    }


def canonicalize_capability(request: dict[str, Any]) -> dict[str, Any]:
    """Return the exact lexicographic form of a small labeled graph.

    This bundled reference implementation deliberately favors transparency
    over scale. Production graph plugins can bind nauty/Traces while retaining
    the same capability contract.
    """

    try:
        selected = GraphCanonicalizeRequest.model_validate(request)
    except ValidationError as exc:
        raise ValueError(
            "graph canonicalization request does not match its contract"
        ) from exc
    structure = selected.structure.model_dump(mode="json", exclude_none=True)
    vertices = structure["vertices"]
    if len(vertices) > 9:
        raise ValueError("reference canonicalizer is limited to nine vertices")

    best_bytes: bytes | None = None
    best_payload: dict[str, Any] | None = None
    best_mappings: list[dict[str, str]] = []
    canonical_names = tuple(f"v{index}" for index in range(len(vertices)))
    for ordering in permutations(vertices):
        mapping = dict(zip(ordering, canonical_names, strict=True))
        payload = _relabel_graph_payload(structure, mapping)
        encoded = canonicalize_json(payload)
        if best_bytes is None or encoded < best_bytes:
            best_bytes = encoded
            best_payload = payload
            best_mappings = [mapping]
        elif encoded == best_bytes:
            best_mappings.append(mapping)

    if best_payload is None:
        raise RuntimeError("no valid graph path payload was found")
    chosen_mapping = best_mappings[0]
    orbit_sets: list[set[str]] = [{vertex} for vertex in vertices]
    for canonical_name in canonical_names:
        members = {
            vertex
            for mapping in best_mappings
            for vertex, mapped in mapping.items()
            if mapped == canonical_name
        }
        if not members:
            continue
        merged = set().union(
            *(orbit for orbit in orbit_sets if orbit.intersection(members))
        )
        orbit_sets = [orbit for orbit in orbit_sets if not orbit.intersection(merged)]
        orbit_sets.append(merged)

    return {
        "response_version": "1",
        "canonical_payload": best_payload,
        "mapping": chosen_mapping,
        "automorphism_group_order": str(len(best_mappings)),
        "orbits": [
            sorted(orbit)
            for orbit in sorted(orbit_sets, key=lambda value: sorted(value))
        ],
    }


def _relabel_graph_payload(
    structure: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    payload = deepcopy(structure)
    payload["vertices"] = sorted(mapping.values())
    payload["arcs"] = sorted(
        [[mapping[left], mapping[right]] for left, right in structure.get("arcs", [])]
    )
    if isinstance(structure.get("source"), str):
        payload["source"] = mapping[structure["source"]]
    if isinstance(structure.get("terminals"), list):
        payload["terminals"] = sorted(
            mapping[terminal] for terminal in structure["terminals"]
        )
    if isinstance(structure.get("intended_paths"), list):
        payload["intended_paths"] = sorted(
            [
                [mapping[vertex] for vertex in path]
                for path in structure["intended_paths"]
            ]
        )
    return payload


def enumerate_candidates_capability(request: dict[str, Any]) -> dict[str, Any]:
    """Page through labeled DAGs whose arcs respect the vertex index order."""

    try:
        selected = GraphEnumerationRequest.model_validate(request)
    except ValidationError as exc:
        raise ValueError(
            "graph enumeration request does not match its contract"
        ) from exc
    vertex_count = selected.bounds["vertices"]
    page_size = selected.page_size
    offset = selected.cursor.offset if selected.cursor is not None else 0

    vertices = [f"v{index}" for index in range(vertex_count)]
    possible_arcs = [
        [vertices[left], vertices[right]]
        for left in range(vertex_count)
        for right in range(left + 1, vertex_count)
    ]
    total = 1 << len(possible_arcs)
    stop = min(offset + page_size, total)
    candidates = [
        {
            "vertices": vertices,
            "arcs": [arc for bit, arc in enumerate(possible_arcs) if mask & (1 << bit)],
        }
        for mask in range(offset, stop)
    ]
    complete = stop >= total
    return {
        "response_version": "1",
        "candidates": candidates,
        "next_cursor": None if complete else {"offset": stop},
        "complete": complete,
        "scope": {
            "vertices": vertex_count,
            "simple": True,
            "directed": True,
            "acyclic": True,
            "labeled": True,
            "arc_rule": "v_i_to_v_j_only_when_i_less_than_j",
            "candidate_count": total,
        },
    }
