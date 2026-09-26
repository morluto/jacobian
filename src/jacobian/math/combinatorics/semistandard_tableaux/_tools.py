"""Public bounded semistandard-tableau enumeration operation."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.semistandard_tableaux._models import (
    SemistandardTableauEnumerationRequest,
    SemistandardTableauEnumerationResult,
)
from jacobian.math.combinatorics.semistandard_tableaux.enumeration import (
    enumerate_semistandard_young_tableaux,
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="combinatorics.semistandard_young_tableaux.enumerate",
        title="Enumerate semistandard Young tableaux",
        description=(
            "Return every semistandard tableau of a straight partition shape "
            "with entries in a supplied finite alphabet, in lexicographic row "
            "order. The hook-content count bounds output before construction."
        ),
        request_type=SemistandardTableauEnumerationRequest,
        result_type=SemistandardTableauEnumerationResult,
        run=lambda request: enumerate_semistandard_young_tableaux(
            request.partition, request.max_entry
        ),
        tags=("combinatorics", "tableau", "semistandard", "exact"),
        discovery_terms=(
            "enumerate semistandard Young tableaux",
            "all semistandard tableaux with bounded entries",
        ),
        examples=(
            OperationExample(
                name="semistandard_tableaux_shape_2_1_alphabet_2",
                description=(
                    "Enumerate the two semistandard tableaux of shape (2,1) "
                    "with entries from {1,2}."
                ),
                input={"partition": {"parts": [2, 1]}, "max_entry": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
