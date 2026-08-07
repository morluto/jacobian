"""Independent exact replay for finite enumerated partitions."""

from __future__ import annotations

from typing import Any


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": detail,
    }


def _certificate_detail(
    certificate: dict[str, Any],
    expected_bindings: object,
    claim_uri: object,
) -> str | None:
    if certificate.get("certificate_type") != "finite.partition":
        return "unexpected certificate type"
    if certificate.get("format_version") != "1":
        return "unsupported certificate format"
    if certificate.get("bindings") != expected_bindings:
        return "certificate bindings do not match request"
    certificate_payload = certificate.get("payload")
    if (
        not isinstance(certificate_payload, dict)
        or certificate_payload.get("relation_id") != "case.relation.partitions"
        or certificate_payload.get("obligation_uri") != claim_uri
    ):
        return "certificate relationship metadata is not bound"
    return None


def _scope_and_cases_detail(universe: object, cases: object) -> str | None:
    if (
        not isinstance(universe, list)
        or not all(isinstance(item, str) for item in universe)
        or len(universe) != len(set(universe))
        or not isinstance(cases, list)
    ):
        return "scope or partition is malformed"
    return None


def _case_fields_detail(
    case_id: object,
    members: object,
    seen_case_ids: set[str],
) -> str | None:
    if (
        not isinstance(case_id, str)
        or not case_id
        or case_id in seen_case_ids
        or not isinstance(members, list)
        or not all(isinstance(member, str) for member in members)
        or len(members) != len(set(members))
    ):
        return "case identifiers or members are malformed"
    return None


def _validate_case_memberships(
    members: list[str],
    universe_set: set[str],
    memberships: dict[str, str],
    case_id: str,
    require_disjoint: bool,
) -> str | None:
    for member in members:
        if member not in universe_set:
            return "partition contains an element outside the scope"
        if require_disjoint and member in memberships:
            return "partition cases overlap"
        memberships[member] = case_id
    return None


def _validate_cases(
    cases: list[Any],
    universe_set: set[str],
    require_disjoint: bool,
) -> tuple[dict[str, str], str | None]:
    seen_case_ids: set[str] = set()
    memberships: dict[str, str] = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"case_id", "members"}:
            return memberships, "case is malformed"
        case_id = case["case_id"]
        members = case["members"]
        detail = _case_fields_detail(case_id, members, seen_case_ids)
        if detail is not None:
            return memberships, detail
        seen_case_ids.add(case_id)
        detail = _validate_case_memberships(
            members, universe_set, memberships, case_id, require_disjoint
        )
        if detail is not None:
            return memberships, detail
    return memberships, None


def check_partition(request: dict[str, Any]) -> dict[str, Any]:
    """Recompute finite coverage and disjointness from bound artifacts."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim = request["claim"]["payload"]
        partition = request["candidate"]["payload"]
        scope = request["scope"]["payload"]
        certificate = request["certificate"]["payload"]
        if claim.get("predicate") != "finite_partition":
            return _reject("unsupported claim predicate")
        claim_uri = request["claim"].get("artifact_uri")
        detail = _certificate_detail(
            certificate, request["expected_bindings"], claim_uri
        )
        if detail is not None:
            return _reject(detail)
        universe = scope.get("elements")
        cases = partition.get("cases")
        detail = _scope_and_cases_detail(universe, cases)
        if detail is not None:
            return _reject(detail)
        universe_set = set(universe)
        memberships, detail = _validate_cases(
            cases, universe_set, claim.get("require_disjoint", True)
        )
        if detail is not None:
            return _reject(detail)
        if set(memberships) != universe_set:
            return _reject("partition does not cover the exact finite scope")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_INTEGER",
            "method": "EXHAUSTIVE_FINITE",
            "coverage": "EXHAUSTIVE",
            "detail": (
                f"replayed {len(cases)} cases over {len(universe)} exact elements"
            ),
            "relation_id": "case.relation.partitions",
            "relationship_source_artifact_uris": ([request["scope"]["artifact_uri"]]),
            "relationship_target_artifact_uris": (
                [request["candidate"]["artifact_uri"]]
            ),
            "obligation_uri": claim_uri,
        }
    except (KeyError, TypeError, ValueError):
        return _reject("malformed checker request")
