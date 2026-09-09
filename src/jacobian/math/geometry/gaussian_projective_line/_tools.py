"""Gaussian projective-line operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry.gaussian_projective_line._models import (
    GaussianCrossRatioSource,
)
from jacobian.math.geometry.gaussian_projective_line.operations import (
    gaussian_rational_cross_ratio,
)
from jacobian.math.number_theory.number_fields import GaussianRational

TOOLS: MathTools = (
    MathTool(
        operation_id="geometry.projective_line.gaussian_rational.cross_ratio.compute",
        title="Compute a Gaussian-rational projective cross-ratio",
        description="Normalize homogeneous P1(Q(i)) points and compute their exact cross-ratio through homogeneous determinants.",
        request_type=GaussianCrossRatioSource,
        result_type=GaussianRational,
        run=gaussian_rational_cross_ratio,
        tags=(
            "geometry",
            "projective-line",
            "cross-ratio",
            "gaussian-rational",
            "exact",
        ),
        examples=(
            OperationExample(
                name="zero_one_two_infinity",
                description="Compute the cross-ratio of four rational points embedded in Q(i).",
                input={
                    name: {"coordinates": coordinates}
                    for name, coordinates in {
                        "first": [
                            {
                                "real": {"num": "0", "den": "1"},
                                "imaginary": {"num": "0", "den": "1"},
                            },
                            {
                                "real": {"num": "1", "den": "1"},
                                "imaginary": {"num": "0", "den": "1"},
                            },
                        ],
                        "second": [
                            {
                                "real": {"num": "1", "den": "1"},
                                "imaginary": {"num": "0", "den": "1"},
                            },
                            {
                                "real": {"num": "1", "den": "1"},
                                "imaginary": {"num": "0", "den": "1"},
                            },
                        ],
                        "third": [
                            {
                                "real": {"num": "2", "den": "1"},
                                "imaginary": {"num": "0", "den": "1"},
                            },
                            {
                                "real": {"num": "1", "den": "1"},
                                "imaginary": {"num": "0", "den": "1"},
                            },
                        ],
                        "fourth": [
                            {
                                "real": {"num": "1", "den": "1"},
                                "imaginary": {"num": "0", "den": "1"},
                            },
                            {
                                "real": {"num": "0", "den": "1"},
                                "imaginary": {"num": "0", "den": "1"},
                            },
                        ],
                    }.items()
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
