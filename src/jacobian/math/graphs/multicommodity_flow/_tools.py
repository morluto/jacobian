"""Exact multicommodity-flow operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.multicommodity_flow._models import (
    MulticommodityFlowProfileRequest,
    MulticommodityFlowProfileResult,
)
from jacobian.math.graphs.multicommodity_flow._operations import (
    _run_multicommodity_flow_profile,
)

TOOLS: MathTools = (
    MathTool(
        operation_id="network.multicommodity_flow.profile.compute",
        title="Compute an exact multicommodity-flow profile",
        description=(
            "Compute exact commodity conservation, aggregate edge loads, signed "
            "capacity slacks, capacity feasibility, and congestion for one "
            "canonical sparse rational multicommodity-flow tensor. The result "
            "retains the complete network, demands, and tensor; it "
            "does not search for a flow or solve an optimization problem."
        ),
        request_type=MulticommodityFlowProfileRequest,
        result_type=MulticommodityFlowProfileResult,
        run=_run_multicommodity_flow_profile,
        tags=(
            "network",
            "multicommodity-flow",
            "flow-profile",
            "conservation",
            "capacity",
            "exact",
            "bounded",
        ),
        examples=(
            example(
                "two_commodities_share_a_bottleneck",
                "Profile two exact commodity flows sharing a directed bottleneck; "
                "network edges, commodities, and nonzero entries must use their "
                "published canonical sort orders.",
                {
                    "flow": {
                        "network": {
                            "vertex_count": 4,
                            "edges": [
                                {
                                    "source": 0,
                                    "target": 2,
                                    "capacity": {"num": "2", "den": "1"},
                                },
                                {
                                    "source": 1,
                                    "target": 2,
                                    "capacity": {"num": "2", "den": "1"},
                                },
                                {
                                    "source": 2,
                                    "target": 3,
                                    "capacity": {"num": "3", "den": "1"},
                                },
                            ],
                        },
                        "commodities": [
                            {
                                "commodity_id": "a",
                                "source": 0,
                                "sink": 3,
                                "demand": {"num": "1", "den": "1"},
                            },
                            {
                                "commodity_id": "b",
                                "source": 1,
                                "sink": 3,
                                "demand": {"num": "2", "den": "1"},
                            },
                        ],
                        "entries": [
                            {
                                "commodity_id": "a",
                                "source": 0,
                                "target": 2,
                                "amount": {"num": "1", "den": "1"},
                            },
                            {
                                "commodity_id": "a",
                                "source": 2,
                                "target": 3,
                                "amount": {"num": "1", "den": "1"},
                            },
                            {
                                "commodity_id": "b",
                                "source": 1,
                                "target": 2,
                                "amount": {"num": "2", "den": "1"},
                            },
                            {
                                "commodity_id": "b",
                                "source": 2,
                                "target": 3,
                                "amount": {"num": "2", "den": "1"},
                            },
                        ],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
