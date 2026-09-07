"""Declaration for exact mutual information on canonical joint tables."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.probability.mutual_information import mutual_information
from jacobian.math.probability.values import FiniteJointTable, MutualInformationResult

MUTUAL_INFORMATION_OPERATION = MathTool(
    operation_id="probability.joint.mutual_information.compute",
    title="Compute an exact finite-table mutual information value",
    description=(
        "Compute ordered marginals and every positive-support likelihood "
        "ratio for one bounded normalized rational joint table. Return the "
        "exact logarithmic value scale*I=log_base(product), without floating point."
    ),
    request_type=FiniteJointTable,
    result_type=MutualInformationResult,
    run=mutual_information,
    tags=(
        "probability",
        "information-theory",
        "mutual-information",
        "finite",
        "exact",
        "exact-value",
    ),
    examples=(
        OperationExample(
            name="perfectly_correlated_fair_bits",
            description="Compute exact base-two mutual information for two identical fair bits.",
            input={
                "row_labels": ["0", "1"],
                "column_labels": ["0", "1"],
                "probabilities": [
                    [
                        {"num": "1", "den": "2"},
                        {"num": "0", "den": "1"},
                    ],
                    [
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "2"},
                    ],
                ],
                "log_base": 2,
            },
        ),
    ),
)


__all__ = ["MUTUAL_INFORMATION_OPERATION"]
