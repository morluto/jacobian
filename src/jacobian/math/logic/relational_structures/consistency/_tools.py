"""Catalog declaration for finite CSP domain consistency."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.logic.relational_structures.consistency._models import (
    CspDomainConsistency,
    CspDomainRequest,
)
from jacobian.math.logic.relational_structures.consistency.operations import (
    generalized_arc_consistency,
)

TOOLS: MathTools = (
    MathTool(
        operation_id="relational.csp.generalized_arc_consistency.compute",
        title="Compute generalized arc consistency for a finite CSP",
        description=(
            "Return the greatest subdomains of the supplied variable domains "
            "in which every retained value has support in every incident "
            "constraint. Empty domains and false nullary constraints are "
            "reported exactly. A nonempty fixed point does not establish a "
            "global solution. Candidate rows and support-index work are bounded."
        ),
        request_type=CspDomainRequest,
        result_type=CspDomainConsistency,
        run=generalized_arc_consistency,
        tags=("finite-csp", "consistency", "exact"),
        discovery_terms=(
            "generalized arc consistency finite constraint satisfaction",
            "GAC domain pruning CSP",
            "local constraint consistency",
        ),
        examples=(
            OperationExample(
                name="binary_relation_domains",
                description="Prune values without support in the incident relation.",
                input={
                    "instance": {
                        "template": {
                            "carrier_size": 2,
                            "signature": [{"symbol_id": "E", "arity": 2}],
                            "relation_tables": [[[0, 1]]],
                        },
                        "variable_count": 2,
                        "constraints": [{"constraint_id": "edge", "symbol_id": "E", "scope": [0, 1]}],
                    },
                    "domains": [[0, 1], [0, 1]],
                },
            ),
        ),
    ),
)
