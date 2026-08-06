"""Deliberately simple untrusted conjecture-plugin entrypoints."""

from __future__ import annotations

import copy
from typing import Any


def transform_fixture_hypothesis(request: dict[str, Any]) -> dict[str, Any]:
    source = request["source"]
    operation = request["operation"]
    claim = (
        copy.deepcopy(request["constraints"]["claim_template"])
        if operation == "PARAMETER_GENERALIZE"
        else copy.deepcopy(source["payload"])
    )
    parameters = claim["predicate"].setdefault("parameters", {})
    parameters["hypothesis_operation"] = operation.lower()
    proposal: dict[str, Any] = {
        "claim": claim,
        "edit": {
            "kind": operation.lower(),
            "description": "record the fixture hypothesis operation",
            "path": "/predicate/parameters/hypothesis_operation",
            "before": None,
            "after": operation.lower(),
        },
        "metrics": {"fixture_rank": "1"},
        "detail": "fixture hypothesis",
    }
    if operation == "PARAMETER_GENERALIZE":
        proposal["parameter_region"] = {
            "kind": request["constraints"].get("region_kind", "SUFFICIENT"),
            "conditions": {"n": {"minimum": "1"}},
            "evidence": "SAMPLED",
            "sample_uris": [request["evidence"][-1]["artifact_uri"]],
        }
    proposals = (
        [proposal, copy.deepcopy(proposal)] if operation == "GENERATE" else [proposal]
    )
    return {
        "response_version": "1",
        "proposals": proposals,
        "state": {"operation": operation},
        "complete": True,
        "detail": "fixture hypothesis transformation",
    }


def transform_with_unsupported_region_promotion(
    request: dict[str, Any],
) -> dict[str, Any]:
    source = request["source"]
    return {
        "response_version": "1",
        "proposals": [
            {
                "claim": source["payload"],
                "edit": {
                    "kind": "parameter",
                    "description": "unsupported promotion attempt",
                },
                "parameter_region": {
                    "kind": "SUFFICIENT",
                    "conditions": {"n": {"minimum": "1"}},
                    "evidence": "VERIFIED_SUFFICIENT",
                    "subject_uri": request["evidence"][0]["artifact_uri"],
                    "verification_record_uri": (request["evidence"][0]["artifact_uri"]),
                },
            }
        ],
        "state": {},
        "complete": True,
    }


def transform_with_unbound_region_sample(
    request: dict[str, Any],
) -> dict[str, Any]:
    response = transform_fixture_hypothesis(request)
    for proposal in response["proposals"]:
        proposal["parameter_region"] = {
            "kind": "SUFFICIENT",
            "conditions": {"n": {"minimum": "1"}},
            "evidence": "SAMPLED",
            "sample_uris": [request["constraints"]["sample_uri"]],
        }
    return response
