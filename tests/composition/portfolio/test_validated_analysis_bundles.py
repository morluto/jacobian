from __future__ import annotations

from jacobian.domains.analysis import REAL_ANALYSIS_BUNDLE
from jacobian.domains.optimization import RATIONAL_OPTIMIZATION_BUNDLE
from jacobian.domains.probability import FINITE_PROBABILITY_BUNDLE


def test_subject_bundles_preserve_wire_contracts_and_report_one_backend() -> None:
    assert {
        bundle.domain_id: (
            bundle.provider_runtime.provider,
            bundle.schema_namespace,
            tuple(operation.capability_id for operation in bundle.capabilities),
        )
        for bundle in (
            REAL_ANALYSIS_BUNDLE,
            FINITE_PROBABILITY_BUNDLE,
            RATIONAL_OPTIMIZATION_BUNDLE,
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
