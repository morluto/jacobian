"""Tool declaration for regular-domain restriction of transducers."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.automata.transducers.domain_restriction._models import (
    SubsequentialDomainRestrictionRequest,
)
from jacobian.math.logic.automata.transducers.domain_restriction.operations import (
    restrict_subsequential_domain,
)
from jacobian.math.logic.automata.transducers.values import SubsequentialTransducer


def _run(
    request: SubsequentialDomainRestrictionRequest,
) -> SubsequentialTransducer:
    return restrict_subsequential_domain(request)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="transducer.subsequential.restrict_domain.compute",
        title="Restrict a finite state function to a regular language",
        description=(
            "Intersect the domain of a partial subsequential transducer with a "
            "finite automaton language and return the exact restricted function "
            "as a subsequential transducer. The reachable product is trimmed to "
            "states that can reach an accepting product state; product work and "
            "the canonical transducer result limits are admitted before output "
            "construction."
        ),
        request_type=SubsequentialDomainRestrictionRequest,
        result_type=SubsequentialTransducer,
        run=_run,
        tags=("automata", "transducer", "regular-language", "exact"),
        discovery_terms=(
            "restrict a subsequential function to a regular input language",
            "intersection of a transducer domain with a DFA language",
        ),
        examples=(
            OperationExample(
                name="keep_even_length_identity_words",
                description=(
                    "Restrict the identity function on a one-symbol alphabet to "
                    "the regular language of even-length words."
                ),
                input={
                    "transducer": {
                        "input_alphabet_size": 1,
                        "output_alphabet_size": 1,
                        "input_alphabet_id": "letter",
                        "output_alphabet_id": "letter",
                        "input_alphabet": {"symbols": ["a"]},
                        "output_alphabet": {"symbols": ["a"]},
                        "state_count": 1,
                        "initial_state": 0,
                        "transitions": [
                            {
                                "source": 0,
                                "input_symbol": 0,
                                "target": 0,
                                "output": [0],
                            }
                        ],
                        "final_outputs": [{"state": 0, "output": []}],
                    },
                    "domain_dfa": {
                        "state_count": 2,
                        "alphabet_size": 1,
                        "alphabet_id": "letter",
                        "alphabet": {"symbols": ["a"]},
                        "initial_state": 0,
                        "accepting_states": [0],
                        "transitions": [
                            {"source": 0, "symbol": 0, "target": 1},
                            {"source": 1, "symbol": 0, "target": 0},
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
