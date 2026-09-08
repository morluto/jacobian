"""Publication of complete fixed-node Lebesgue interval profiles."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.analysis.approximation.lebesgue._models import (
    LebesgueIntervalProfile,
    LebesgueIntervalRequest,
)
from jacobian.math.analysis.approximation.lebesgue.operations import (
    lebesgue_interval_profile,
)


def _run(request: LebesgueIntervalRequest) -> LebesgueIntervalProfile:
    return lebesgue_interval_profile(request.source)


TOOLS: MathTools = (
    MathTool(
        operation_id="approximation.lebesgue.interval_profiles.compute",
        title="Compute exact fixed-node Lebesgue interval profiles",
        description=(
            "For distinct rational interpolation nodes and a closed rational query interval, "
            "return the complete Lagrange basis, sign-cell polynomials for the sum of its "
            "absolute values, all isolated derivative critical points, cell maxima, and the "
            "global maximum with every tied maximizing point or constant interval. "
            "The query may lie inside or outside the node hull. Exact real-algebraic values "
            "retain irreducible polynomials and real-root indices. Source-derived admission "
            "bounds algebraic degree, coefficient growth, root isolation and complete output."
        ),
        request_type=LebesgueIntervalRequest,
        result_type=LebesgueIntervalProfile,
        run=_run,
        tags=("approximation", "interpolation", "lebesgue", "exact", "maximum"),
        examples=(
            OperationExample(
                name="three_nodes",
                description="Nodes -1, 0, 1 on [-1,1] give maximum 5/4 at both -1/2 and 1/2. Nodes must be distinct and strictly increasing, and the query interval endpoints must be ordered.",
                input={
                    "source": {
                        "nodes": {
                            "nodes": [{"num": str(n), "den": "1"} for n in (-1, 0, 1)]
                        },
                        "interval": {
                            "lower": {"num": "-1", "den": "1"},
                            "upper": {"num": "1", "den": "1"},
                        },
                    }
                },
            ),
        ),
    ),
)
