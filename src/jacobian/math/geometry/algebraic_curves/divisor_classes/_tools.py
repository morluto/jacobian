"""Published operations constructing plane-curve divisor classes."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.algebraic_curves.divisor_classes._models import (
    PlaneCurveStrictTransformRequest,
)
from jacobian.math.geometry.algebraic_curves.divisor_classes.operations import (
    plane_curve_strict_transform_class,
)
from jacobian.math.geometry.blowup_p2._models import BlowupDivisorClass


def _run(request: PlaneCurveStrictTransformRequest) -> BlowupDivisorClass:
    return plane_curve_strict_transform_class(request)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="algebraic_geometry.plane_curve.strict_transform_class.compute",
        title="Compute a plane curve's strict-transform divisor class",
        description=(
            "For a nonzero homogeneous rational plane curve and a labelled finite "
            "blow-up of P2 at distinct rational points, compute the exact class "
            "dH - sum m_i E_i. Each multiplicity is the curve's local order at "
            "that point, computed in a projective chart. The explicit coordinate "
            "variable permutation binds point coordinates to the polynomial axis. "
            "This slice admits degree 12, 64 source terms, 16 proper points, "
            "32-digit coefficients, and 16-digit projective coordinates."
        ),
        request_type=PlaneCurveStrictTransformRequest,
        result_type=BlowupDivisorClass,
        run=_run,
        tags=("algebraic-geometry", "plane-curve", "strict-transform", "blow-up"),
        discovery_terms=(
            "plane curve strict transform divisor class",
            "multiplicity of plane curve at rational point",
            "curve class on blowup of projective plane",
            "degree exceptional multiplicity divisor",
        ),
        examples=(
            OperationExample(
                name="cuspidal_cubic_class",
                description=(
                    "The cuspidal cubic has multiplicity two at its cusp and "
                    "multiplicity one at the supplied smooth point."
                ),
                input={
                    "polynomial": {
                        "domain": "QQ",
                        "variables": ["x", "y", "z"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [3, 0, 0],
                                },
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [0, 2, 1],
                                },
                            ]
                        },
                    },
                    "surface": {
                        "points": [
                            {
                                "label": "cusp",
                                "point": {
                                    "coordinates": [
                                        {"num": "0", "den": "1"},
                                        {"num": "0", "den": "1"},
                                        {"num": "1", "den": "1"},
                                    ]
                                },
                            },
                            {
                                "label": "smooth",
                                "point": {
                                    "coordinates": [
                                        {"num": "1", "den": "1"},
                                        {"num": "1", "den": "1"},
                                        {"num": "1", "den": "1"},
                                    ]
                                },
                            },
                        ]
                    },
                    "projective_coordinate_variables": ["x", "y", "z"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
