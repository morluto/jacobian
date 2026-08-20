from __future__ import annotations

from jacobian.math.analysis._tools import TOOLS as ANALYSIS_TOOLS
from jacobian.math.optimization._tools import TOOLS as OPTIMIZATION_TOOLS
from jacobian.math.probability._tools import TOOLS as PROBABILITY_TOOLS


def test_subject_operation_groups_preserve_wire_contracts() -> None:
    assert tuple(
        tuple(operation.operation_id for operation in operations)
        for operations in (
            ANALYSIS_TOOLS,
            PROBABILITY_TOOLS,
            OPTIMIZATION_TOOLS,
        )
    ) == (
        (
            "analysis.real_function.point_enclosure.compute",
            "interval.compute.enclosure",
        ),
        (
            "probability.joint.mutual_information.compute",
            "probability.finite_distribution.raw_moment.compute",
            "probability.finite_distribution.event_probability.compute",
            "probability.finite_distribution.condition.compute",
            "probability.finite_distribution.pushforward.compute",
            "probability.finite_distribution.convolution.compute",
            "probability.gaussian_polynomial.moment.compute",
            "probability.graph_reliability.connection_probability.compute",
        ),
        ("optimization.linear.rational_optimum.compute",),
    )
