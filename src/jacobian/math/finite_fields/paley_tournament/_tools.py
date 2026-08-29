"""Typed declarations for the Paley tournament operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.finite_fields.paley_tournament._models import (
    PaleyTournamentRequest,
    PaleyTournamentResult,
)
from jacobian.math.finite_fields.paley_tournament.operations import (
    construct_paley_tournament,
)


def _construct(request: PaleyTournamentRequest) -> PaleyTournamentResult:
    return construct_paley_tournament(request.field_order)


TOOLS: MathTools = (
    MathTool(
        operation_id="finite_field.paley_tournament.construct",
        title="Construct the quadratic-residue Paley tournament of a finite field",
        description=(
            "For an admitted finite field F_q of odd order q ≡ 3 (mod 4), "
            "return the canonical directed Paley tournament on its elements: "
            "x -> y exactly when y - x is a nonzero quadratic residue in F_q."
        ),
        request_type=PaleyTournamentRequest,
        result_type=PaleyTournamentResult,
        run=_construct,
        tags=("finite_field", "paley", "tournament", "exact"),
        examples=(
            example(
                "f3",
                "Paley tournament of F_3.",
                {"field_order": 3},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
