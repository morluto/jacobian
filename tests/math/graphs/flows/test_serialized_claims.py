"""Flow and cut witnesses stay bound to their source networks."""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.flows import (
    verify_edge_disjoint_paths,
    verify_max_flow,
    verify_min_cut,
)
from jacobian.math.graphs.flows._models import (
    EdgeDisjointPathsRequest,
    MaxFlowRequest,
    MinCutRequest,
)
from jacobian.math.graphs.flows._tools import (
    compute_edge_disjoint_paths,
    compute_max_flow,
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
