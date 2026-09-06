"""Flow and cut witnesses stay bound to their source networks."""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.flows import (
    verify_edge_disjoint_paths,
    verify_max_flow,
    verify_min_cost_flow,
    verify_min_cut,
)
from jacobian.math.graphs.flows._models import (
    EdgeDisjointPathsRequest,
    EdgeDisjointPathsResult,
    MaxFlowRequest,
    MaxFlowResult,
    MinCostFlowRequest,
    MinCostFlowResult,
    MinCutRequest,
)
from jacobian.math.graphs.flows._tools import (
    compute_edge_disjoint_paths,
    compute_max_flow,
    compute_min_cost_flow,
    compute_min_cut,
)


def test_flow_cut_and_paths_round_trip() -> None:
    graph = {
        "vertex_count": 3,
        "edges": [
            {"source": 0, "target": 1, "capacity": {"num": "1", "den": "1"}},
            {"source": 1, "target": 2, "capacity": {"num": "1", "den": "1"}},
        ],
    }
    flow = compute_max_flow(
        MaxFlowRequest.model_validate({"graph": graph, "source": 0, "sink": 2})
    )
    cut = compute_min_cut(
        MinCutRequest.model_validate({"graph": graph, "source": 0, "sink": 2})
    )
    paths = compute_edge_disjoint_paths(
        EdgeDisjointPathsRequest.model_validate(
            {
                "graph": {"vertex_count": 3, "edges": [[0, 1], [1, 2]]},
                "source": 0,
                "sink": 2,
            }
        )
    )
    result: Any
    verifier: Callable[[Any], bool]
    for result, verifier in (
        (flow, verify_max_flow),
        (cut, verify_min_cut),
        (paths, verify_edge_disjoint_paths),
    ):
        assert verifier(type(result).model_validate_json(result.model_dump_json()))
    payload = flow.model_dump(mode="json")
    payload["flow_edges"] = []
    payload["flow_value"] = {"num": "0", "den": "1"}
    assert not verify_max_flow(type(flow).model_validate(payload))
    payload = flow.model_dump(mode="json")
    payload["flow_edges"][0]["target"] = 2
    with pytest.raises(ValidationError):
        type(flow).model_validate(payload)
    payload = cut.model_dump(mode="json")
    payload["cut_value"] = {"num": "2", "den": "1"}
    assert not verify_min_cut(type(cut).model_validate(payload))
    payload = paths.model_dump(mode="json")
    payload["paths"] = [[0, 2]]
    assert not verify_edge_disjoint_paths(type(paths).model_validate(payload))


def _costed_graph() -> dict:
    return {
        "vertex_count": 3,
        "edges": [
            {
                "source": 0,
                "target": 1,
                "capacity": {"num": "5", "den": "1"},
                "cost": {"num": "1", "den": "1"},
            },
            {
                "source": 1,
                "target": 2,
                "capacity": {"num": "5", "den": "1"},
                "cost": {"num": "2", "den": "1"},
            },
            {
                "source": 0,
                "target": 2,
                "capacity": {"num": "5", "den": "1"},
                "cost": {"num": "4", "den": "1"},
            },
        ],
    }


def test_min_cost_flow_claim_checks_feasibility_and_cost() -> None:
    result = compute_min_cost_flow(
        MinCostFlowRequest.model_validate(
            {"graph": _costed_graph(), "demands": [-2, 0, 2]}
        )
    )
    assert result.feasible
    assert verify_min_cost_flow(
        MinCostFlowResult.model_validate_json(result.model_dump_json())
    )
    assert result.total_cost.as_fraction() == 6

    forged = result.model_dump(mode="json")
    forged["total_cost"] = {"num": "7", "den": "1"}
    assert not verify_min_cost_flow(MinCostFlowResult.model_validate(forged))

    over_capacity = result.model_dump(mode="json")
    over_capacity["flow_edges"][0]["flow"] = {"num": "99", "den": "1"}
    assert not verify_min_cost_flow(MinCostFlowResult.model_validate(over_capacity))

    unbound = result.model_dump(mode="json")
    unbound["flow_edges"].append(
        {"source": 2, "target": 0, "flow": {"num": "1", "den": "1"}}
    )
    assert not verify_min_cost_flow(MinCostFlowResult.model_validate(unbound))


def test_infeasible_min_cost_outcome_is_a_producer_outcome() -> None:
    result = compute_min_cost_flow(
        MinCostFlowRequest.model_validate(
            {
                "graph": {
                    "vertex_count": 2,
                    "edges": [
                        {
                            "source": 0,
                            "target": 1,
                            "capacity": {"num": "1", "den": "1"},
                            "cost": {"num": "1", "den": "1"},
                        }
                    ],
                },
                "demands": [-2, 2],
            }
        )
    )
    assert result.feasible is False
    assert not verify_min_cost_flow(result)


def test_verifiers_reject_malformed_claims_without_raising() -> None:
    flow = compute_max_flow(
        MaxFlowRequest.model_validate(
            {
                "graph": {
                    "vertex_count": 2,
                    "edges": [
                        {
                            "source": 0,
                            "target": 1,
                            "capacity": {"num": "1", "den": "1"},
                        }
                    ],
                },
                "source": 0,
                "sink": 1,
            }
        )
    )
    payload = flow.model_dump(mode="json")
    payload["sink"] = 7
    assert not verify_max_flow(MaxFlowResult.model_validate(payload))

    with pytest.raises(ValidationError):
        EdgeDisjointPathsResult.model_validate(
            {
                "graph": {"vertex_count": 2, "edges": [[0, 1]]},
                "source": 0,
                "sink": 1,
                "path_count": 1,
                "paths": [[]],
            }
        )
    from jacobian.math.graphs.flows._models import EdgeDisjointPathsGraph

    degenerate = EdgeDisjointPathsResult.model_construct(
        graph=EdgeDisjointPathsGraph(vertex_count=2, edges=((0, 1),)),
        source=0,
        sink=1,
        path_count=1,
        paths=([],),
    )
    assert not verify_edge_disjoint_paths(degenerate)
