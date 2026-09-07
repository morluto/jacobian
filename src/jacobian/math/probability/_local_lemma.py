"""Declaration for canonical asymmetric local-lemma numerical witnesses."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.probability.local_lemma import (
    AsymmetricLocalLemmaWitness,
    AsymmetricLocalLemmaWitnessCheckResult,
    check_asymmetric_local_lemma_witness,
)

ASYMMETRIC_LOCAL_LEMMA_OPERATION = MathTool(
    operation_id="probability.local_lemma.asymmetric_witness.check",
    title="Check an exact asymmetric local-lemma numerical witness",
    description=(
        "Check every exact inequality p_i <= x_i * product over directed "
        "Gamma(i) of (1-x_j), returning the complete source-bound product, "
        "right-hand-side, and slack ledger. Listed self-neighbors contribute "
        "once. This checks only the numerical witness; it does not establish "
        "that the declared neighborhoods are a dependency graph or that any "
        "event-independence hypothesis holds."
    ),
    request_type=AsymmetricLocalLemmaWitness,
    result_type=AsymmetricLocalLemmaWitnessCheckResult,
    run=check_asymmetric_local_lemma_witness,
    tags=(
        "probability",
        "Lovasz-local-lemma",
        "asymmetric-local-lemma",
        "numerical-witness",
        "exact-rational",
        "directed-neighborhoods",
        "bounded",
    ),
    examples=(
        OperationExample(
            name="two_event_directed_witness",
            description="Check two exact directed asymmetric local-lemma inequalities; all arrays share the ordered event axis, neighborhoods are strictly increasing index sets, and each witness lies in [0,1).",
            input={
                "event_labels": ["A", "B"],
                "probability_upper_bounds": [
                    {"num": "1", "den": "4"},
                    {"num": "1", "den": "2"},
                ],
                "witness_parameters": [
                    {"num": "1", "den": "2"},
                    {"num": "1", "den": "2"},
                ],
                "neighborhoods": [[1], []],
            },
        ),
    ),
)


__all__ = ["ASYMMETRIC_LOCAL_LEMMA_OPERATION"]
