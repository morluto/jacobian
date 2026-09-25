"""Public operation declarations for relational carrier relabeling."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.relational_structures.relabeling._models import (
    CspTemplateCarrierRelabeling,
    CspTemplateCarrierRelabelingRequest,
    RelationalCarrierRelabeling,
    RelationalCarrierRelabelingRequest,
)
from jacobian.math.logic.relational_structures.relabeling.operations import (
    relabel_csp_template_carrier,
    relabel_structure_carrier,
)


def _run_structure(
    request: RelationalCarrierRelabelingRequest,
) -> RelationalCarrierRelabeling:
    return relabel_structure_carrier(request.source, request.old_to_new)


def _run_csp_template(
    request: CspTemplateCarrierRelabelingRequest,
) -> CspTemplateCarrierRelabeling:
    return relabel_csp_template_carrier(request.instance, request.old_to_new)


STRUCTURE_EXAMPLE = {
    "source": {
        "carrier_size": 3,
        "signature": [{"symbol_id": "E", "arity": 2}],
        "relation_tables": [[[0, 1], [1, 2]]],
    },
    "old_to_new": [2, 0, 1],
}

CSP_EXAMPLE = {
    "instance": {
        "template": {
            "carrier_size": 2,
            "signature": [{"symbol_id": "R", "arity": 2}],
            "relation_tables": [[[0, 1], [1, 0]]],
        },
        "variable_count": 3,
        "constraints": [
            {"constraint_id": "first", "symbol_id": "R", "scope": [0, 1]},
            {"constraint_id": "repeat", "symbol_id": "R", "scope": [1, 1]},
        ],
    },
    "old_to_new": [1, 0],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="relational_structure.relabel_carrier.compute",
        title="Relabel a finite relational structure carrier",
        description=(
            "Apply a bijection to every relation-tuple coordinate, preserving "
            "the exact ranked signature and returning both carrier maps. "
            "The complete source tuple tables and bounded tuple-sort work are "
            "checked before coordinate transport."
        ),
        request_type=RelationalCarrierRelabelingRequest,
        result_type=RelationalCarrierRelabeling,
        run=_run_structure,
        tags=("relational-structures", "carrier-map", "isomorphism", "exact"),
        examples=(
            OperationExample(
                name="permute_a_directed_relation",
                description="Transport every ordered edge through a carrier bijection.",
                input=STRUCTURE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="csp.instance.relabel_template_carrier.compute",
        title="Relabel a finite CSP template carrier",
        description=(
            "Apply a bijection to every relation tuple in a CSP template. "
            "Variable labels, relation symbols, constraint IDs, and ordered "
            "constraint scopes remain exact; the result returns both target "
            "carrier maps so assignments can be transported explicitly."
        ),
        request_type=CspTemplateCarrierRelabelingRequest,
        result_type=CspTemplateCarrierRelabeling,
        run=_run_csp_template,
        tags=("csp", "relational-structures", "carrier-map", "exact"),
        examples=(
            OperationExample(
                name="relabel_template_with_repeated_scope",
                description="Retain repeated CSP constraint occurrences and scopes.",
                input=CSP_EXAMPLE,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
