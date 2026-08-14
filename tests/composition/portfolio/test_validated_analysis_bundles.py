from __future__ import annotations

from jacobian.domains.analysis import build_real_analysis_bundle
from jacobian.domains.optimization import build_rational_optimization_bundle
from jacobian.domains.probability import build_finite_probability_bundle


def test_subject_operation_groups_preserve_wire_contracts() -> None:
    assert tuple(
        tuple(operation.operation_id for operation in operations)
        for operations in (
            build_real_analysis_bundle(),
            build_finite_probability_bundle(),
            build_rational_optimization_bundle(),
        )
    ) == (
        ("analysis.real_function.point_enclosure.compute",),
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
