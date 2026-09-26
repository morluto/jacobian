"""Public finite permutation-character operation."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.groups.characters.permutation._models import FiniteCharacter
from jacobian.math.groups.characters.permutation.operations import (
    PermutationCharacterRequest,
    permutation_character,
)

TOOLS: MathTools = (
    MathTool(
        operation_id="group.permutation_character.compute",
        title="Compute the permutation character of a finite action",
        description=(
            "Return the exact rational class function whose value on each "
            "conjugacy class is the number of fixed domain points. The "
            "FiniteCharacter result retains the action and complete generated "
            "group class partition, so the true-character claim is source-bound."
        ),
        request_type=PermutationCharacterRequest,
        result_type=FiniteCharacter,
        run=permutation_character,
        tags=("group", "character", "permutation", "fixed-points", "exact"),
        discovery_terms=(
            "finite group permutation character",
            "fixed-point character",
            "character of a finite permutation action",
        ),
        examples=(
            OperationExample(
                name="s3_natural_permutation_character",
                description=(
                    "Compute the character of S3 acting on three points; the "
                    "class values are the fixed-point counts 3, 1, 0."
                ),
                input={
                    "action": {
                        "domain": ["a", "b", "c"],
                        "generators": [[1, 0, 2], [1, 2, 0]],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
