"""Operation declaration for exact proper hypergeometric shift quotients."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.ore_algebras.proper_hypergeometric_terms.shift_quotients import (
    proper_hypergeometric_shift_quotients,
)
from jacobian.math.ore_algebras.proper_hypergeometric_terms.shift_quotients_models import (
    ProperHypergeometricShiftQuotientsRequest,
    ProperHypergeometricShiftQuotientsResult,
)


def _run_shift_quotients(
    request: ProperHypergeometricShiftQuotientsRequest,
) -> ProperHypergeometricShiftQuotientsResult:
    return proper_hypergeometric_shift_quotients(request.term)


TOOLS = (
    MathTool(
        operation_id="ore.proper_hypergeometric.shift_quotients.compute",
        title="Compute proper hypergeometric shift quotients",
        description=(
            "Compute the reduced exact rational functions T(n+1,k)/T(n,k) "
            "and T(n,k+1)/T(n,k) over QQ(n,k). The equalities describe the "
            "generic nonzero-term locus; support-boundary values are not quotients."
        ),
        request_type=ProperHypergeometricShiftQuotientsRequest,
        result_type=ProperHypergeometricShiftQuotientsResult,
        run=_run_shift_quotients,
        tags=("ore-algebra", "holonomic", "hypergeometric", "exact"),
        discovery_terms=(
            "proper hypergeometric term shift quotient",
            "hypergeometric term ratio",
            "term ratio in n and k",
            "creative telescoping input",
        ),
        examples=(
            OperationExample(
                name="binomial_shift_quotients",
                description=(
                    "Compute both formal shift quotients of n!/(k!(n-k)!), "
                    "the summand of the binomial theorem."
                ),
                input={
                    "term": {
                        "polynomial": {
                            "variables": ["n", "k"],
                            "polynomial": {
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "exponents": [0, 0],
                                    }
                                ]
                            },
                        },
                        "factorial_factors": [
                            {
                                "n_coefficient": 0,
                                "k_coefficient": 1,
                                "offset": 0,
                                "power": -1,
                            },
                            {
                                "n_coefficient": 1,
                                "k_coefficient": -1,
                                "offset": 0,
                                "power": -1,
                            },
                            {
                                "n_coefficient": 1,
                                "k_coefficient": 0,
                                "offset": 0,
                                "power": 1,
                            },
                        ],
                    }
                },
            ),
        ),
    ),
)
