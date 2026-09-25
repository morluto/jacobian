"""Catalog declaration for exact ordinary-character degrees."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.characters.degree._models import (
    CharacterDegree,
    CharacterDegreeRequest,
)
from jacobian.math.groups.characters.degree.operations import character_degree

_TRIVIAL_GROUP = {"degree": 1, "generators": [[0]]}
_TRIVIAL_TABLE = {
    "partition": {
        "source": _TRIVIAL_GROUP,
        "classes": [[[0]]],
    },
    "axis": {
        "class_sizes": [1],
        "group_order": 1,
        "cyclotomic_order": 1,
        "group": _TRIVIAL_GROUP,
        "class_representatives": [[0]],
    },
    "rows": [
        {
            "label": "trivial",
            "degree": 1,
            "values": [{"order": 1, "coefficients": [{"num": "1", "den": "1"}]}],
        }
    ],
    "degree_square_sum": 1,
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="character.degree.compute",
        title="Compute the degree of a finite-group character",
        description=(
            "Return chi(1) as an exact nonnegative integer for an ordinary "
            "character represented in a canonical finite-group irreducible "
            "basis. The result retains the source character and its exact "
            "table. Group/table authentication, work, and output are bounded "
            "before conjugacy expansion."
        ),
        request_type=CharacterDegreeRequest,
        result_type=CharacterDegree,
        run=character_degree,
        tags=("group", "character", "degree", "exact"),
        discovery_terms=("dimension of a finite group character", "chi of identity"),
        examples=(
            OperationExample(
                name="three_copies_of_the_trivial_character",
                description=(
                    "Three copies of the trivial character of the trivial group "
                    "have degree three."
                ),
                input={
                    "character": {
                        "table": _TRIVIAL_TABLE,
                        "irreducible_multiplicities": [3],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
