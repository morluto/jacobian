"""Public declarations for exact local Laurent-series profiles."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.polynomials.local_series._models import (
    ValuationProfileRequest,
    ValuationProfileResult,
)
from jacobian.math.polynomials.local_series.operations import (
    laurent_valuation_profile,
)


def compute_valuation_profile(
    request: ValuationProfileRequest,
) -> ValuationProfileResult:
    return laurent_valuation_profile(request.series)


TOOLS: MathTools = (
    MathTool(
        operation_id="local_series.laurent.valuation_profile.compute",
        title="Compute the valuation profile of a truncated Laurent window",
        description=(
            "Project one bounded dense Laurent window with QQ coefficients at "
            "a single center to ZERO_AT_PRECISION or to its exact integer "
            "valuation with leading coefficient, leading monomial exponent, "
            "and pole/zero orders. Admission caps the retained window at "
            "4096 terms before the leading-term scan."
        ),
        request_type=ValuationProfileRequest,
        result_type=ValuationProfileResult,
        run=compute_valuation_profile,
        tags=("local-series", "laurent", "valuation", "exact"),
        discovery_terms=(
            "Laurent series valuation",
            "pole order of a Laurent window",
            "leading term of a Laurent expansion",
        ),
        examples=(
            OperationExample(
                name="simple_pole_window",
                description=(
                    "Profile t^-1 + 2 + 3t at center 0; the window must be "
                    "dense with exactly precision - valuation_lower entries."
                ),
                input={
                    "series": {
                        "variable": "t",
                        "center": {"num": "0", "den": "1"},
                        "valuation_lower": -1,
                        "precision": 2,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                            {"num": "3", "den": "1"},
                        ],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS", "compute_valuation_profile"]
