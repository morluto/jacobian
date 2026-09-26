"""Published operations for Dirichlet-character coordinate transport."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.characters.coordinate_basis._models import (
    DirichletCharacterBasisChangeRequest,
    DirichletCharacterBasisChangeResult,
)
from jacobian.math.number_theory.characters.coordinate_basis.operations import (
    change_dirichlet_character_coordinate_basis,
)


def _run(
    request: DirichletCharacterBasisChangeRequest,
) -> DirichletCharacterBasisChangeResult:
    return change_dirichlet_character_coordinate_basis(
        request.character, request.coordinate_isomorphism
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="dirichlet_character.change_coordinate_basis.compute",
        title="Transport a Dirichlet character to another supplied unit basis",
        description=(
            "Express the same character of (Z/NZ)^* in a target finite-Abelian "
            "unit basis. The input supplies the target-generator images in "
            "source coordinates; the operation admits both complete unit "
            "decompositions and checks that this map fixes every canonical "
            "residue. Return the transported character and its complete exact "
            "residue-value table, including zero on nonunits. Modulus, table, "
            "rank, work, and coefficient field are bounded by the character "
            "group representation."
        ),
        request_type=DirichletCharacterBasisChangeRequest,
        result_type=DirichletCharacterBasisChangeResult,
        run=_run,
        tags=("number-theory", "dirichlet-character", "coordinates", "exact"),
        discovery_terms=(
            "change Dirichlet character unit generator basis",
            "transport Dirichlet character coordinates",
        ),
        examples=(
            OperationExample(
                name="swap-generators-of-units-modulo-eight",
                description=(
                    "Transport a quadratic character across the swapped generators 3 and 5 modulo 8."
                ),
                input={
                    "character": {
                        "group": {
                            "modulus": 8,
                            "unit_residues": [1, 3, 5, 7],
                            "character_count": 4,
                            "invariant_factors": [2, 2],
                            "generators": [3, 5],
                            "generator_orders": [2, 2],
                            "unit_coordinates": [[0, 0], [1, 0], [0, 1], [1, 1]],
                            "exponent": 2,
                        },
                        "coordinates": [1, 0],
                    },
                    "coordinate_isomorphism": {
                        "source_group": {
                            "modulus": 8,
                            "unit_residues": [1, 3, 5, 7],
                            "character_count": 4,
                            "invariant_factors": [2, 2],
                            "generators": [3, 5],
                            "generator_orders": [2, 2],
                            "unit_coordinates": [
                                [0, 0],
                                [1, 0],
                                [0, 1],
                                [1, 1],
                            ],
                            "exponent": 2,
                        },
                        "target_group": {
                            "modulus": 8,
                            "unit_residues": [1, 3, 5, 7],
                            "character_count": 4,
                            "invariant_factors": [2, 2],
                            "generators": [5, 3],
                            "generator_orders": [2, 2],
                            "unit_coordinates": [
                                [0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1],
                            ],
                            "exponent": 2,
                        },
                        "target_generator_images_in_source_coordinates": [
                            [0, 1],
                            [1, 0],
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
