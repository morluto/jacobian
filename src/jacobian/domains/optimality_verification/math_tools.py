"""MathTool declarations for optimality verification."""

from __future__ import annotations

from jacobian.contracts.optimality_verification import (
    RationalOptimalityVerifyRequest,
    RationalOptimalityVerifyResult,
)
from jacobian.domains._examples import example
from jacobian.domains.optimality_verification.operations import (
    verify_rational_optimality,
)
from jacobian.math_tools import MathTool


OPTIMALITY_VERIFICATION_OPERATIONS: tuple[MathTool, ...] = (
    MathTool(
        operation_id="optimality.verify.rational_lp",
        version="1",
        title="Verify an exact rational LP optimum",
        description=(
            "Independently verify a claimed rational LP optimum by "
            "checking primal feasibility, dual feasibility, and strong "
            "duality agreement on the objective value."
        ),
        request_type=RationalOptimalityVerifyRequest,
        result_type=RationalOptimalityVerifyResult,
        run=verify_rational_optimality,
        tags=(
            "optimization",
            "linear",
            "rational",
            "verify",
            "optimality",
            "certificate",
        ),
        examples=(
            example(
                "simple_lp_verify",
                "Verify the optimum of min(x) s.t. x >= 0, x <= 3.",
                {
                    "program": {
                        "variables": ["x"],
                        "objective": [{"num": "1", "den": "1"}],
                        "coefficients": [[{"num": "1", "den": "1"}]],
                        "rhs": [{"num": "3", "den": "1"}],
                    },
                    "claimed_objective": {"num": "0", "den": "1"},
                    "primal_candidate": [{"num": "0", "den": "1"}],
                    "dual_candidate": [{"num": "0", "den": "1"}],
                },
            ),
        ),
    ),
)

__all__ = ["OPTIMALITY_VERIFICATION_OPERATIONS"]
