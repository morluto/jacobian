from __future__ import annotations

from jacobian.domains.analysis import build_real_analysis_bundle
from jacobian.domains.optimization import build_rational_optimization_bundle
from jacobian.domains.probability import build_finite_probability_bundle


def test_subject_bundles_preserve_wire_contracts_and_report_one_backend() -> None:
    assert {
        bundle.domain_id: (
            bundle.provider_runtime.provider,
            bundle.schema_namespace,
            tuple(operation.spec.operation_id for operation in bundle.capabilities),
        )
        for bundle in (
            build_real_analysis_bundle(),
            build_finite_probability_bundle(),
            build_rational_optimization_bundle(),
        )
    } == {
        "analysis": (
            "python-flint",
            "jacobian.validated-analysis",
            ("analysis.real_function.point_enclosure.compute",),
        ),
        "probability": (
            "python-flint",
            "jacobian.validated-analysis",
            (
                "probability.finite_distribution.raw_moment.compute",
                "probability.finite_distribution.event_probability.compute",
                "probability.finite_distribution.condition.compute",
                "probability.finite_distribution.pushforward.compute",
                "probability.finite_distribution.convolution.compute",
                "probability.gaussian_polynomial.moment.compute",
                "probability.graph_reliability.connection_probability.compute",
            ),
        ),
        "optimization": (
            "jacobian.sympy",
            "jacobian.validated-analysis",
            ("optimization.linear.rational_optimum.compute",),
        ),
    }
