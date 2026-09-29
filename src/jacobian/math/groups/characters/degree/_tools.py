"""Catalog declaration for exact finite-group character degrees."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.characters.degree._models import (
    CharacterDegree,
    CharacterDegreeRequest,
)
from jacobian.math.groups.characters.degree.operations import (
    MAX_CHARACTER_DEGREE_WORK,
    character_degree,
)

_S3_STANDARD_CHARACTER = {
    "table": {
        "partition": {
            "source": {"degree": 3, "generators": [[1, 2, 0], [1, 0, 2]]},
            "classes": [
                [[0, 1, 2]],
                [[0, 2, 1], [1, 0, 2], [2, 1, 0]],
                [[1, 2, 0], [2, 0, 1]],
            ],
        },
        "axis": {
            "class_sizes": [1, 3, 2],
            "group_order": 6,
            "cyclotomic_order": 6,
            "group": {"degree": 3, "generators": [[1, 2, 0], [1, 0, 2]]},
            "class_representatives": [[0, 1, 2], [0, 2, 1], [1, 2, 0]],
        },
        "rows": [
            {
                "label": "trivial",
                "degree": 1,
                "values": [
                    {
                        "order": 6,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    {
                        "order": 6,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    {
                        "order": 6,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                ],
            },
            {
                "label": "sign",
                "degree": 1,
                "values": [
                    {
                        "order": 6,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    {
                        "order": 6,
                        "coefficients": [
                            {"num": "-1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    {
                        "order": 6,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                ],
            },
            {
                "label": "standard",
                "degree": 2,
                "values": [
                    {
                        "order": 6,
                        "coefficients": [
                            {"num": "2", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    {
                        "order": 6,
                        "coefficients": [
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    {
                        "order": 6,
                        "coefficients": [
                            {"num": "-1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                ],
            },
        ],
        "degree_square_sum": 6,
    },
    "irreducible_multiplicities": ["3", "0", "0"],
}


def _run_character_degree(request: CharacterDegreeRequest) -> CharacterDegree:
    return character_degree(request.character)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="character.degree.compute",
        title="Compute the degree of a table-bound finite-group character",
        description=(
            "Return chi(1) for an ordinary character expressed in the "
            "irreducible basis of a finite group's canonical character table, "
            "that is the exact sum of each irreducible multiplicity times its "
            "degree. The result retains the source character and table, so it "
            "composes with the other table-bound character operations. The "
            "retained table must be the canonical one for its group. Supported "
            "groups are the trivial group, cyclic groups of order at most 60, "
            "and S3. Coefficient and work bounds are charged before the exact "
            f"degree is summed, against a {MAX_CHARACTER_DEGREE_WORK}-unit "
            "envelope."
        ),
        request_type=CharacterDegreeRequest,
        result_type=CharacterDegree,
        run=_run_character_degree,
        tags=("characters", "degree", "finite-group", "exact"),
        discovery_terms=(
            "character degree",
            "chi of the identity",
        ),
        examples=(
            OperationExample(
                name="three_times_trivial_character_degree",
                description=(
                    "Three times the trivial S3 character has degree 3, since "
                    "chi(1) sums each irreducible multiplicity times its "
                    "degree and the trivial character has degree 1."
                ),
                input={"character": _S3_STANDARD_CHARACTER},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
