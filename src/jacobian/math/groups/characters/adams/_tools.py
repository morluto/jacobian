"""Catalog declaration for finite-group Adams operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.characters._models import CharacterRingElement
from jacobian.math.groups.characters.adams._models import AdamsOperationRequest
from jacobian.math.groups.characters.adams.operations import character_adams_operation

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="character.adams_operation.compute",
        title="Compute an Adams operation of a finite-group virtual character",
        description=(
            "For a table-bound virtual character chi and positive integer k, "
            "return the exact virtual character psi^k(chi), defined by "
            "psi^k(chi)(g) = chi(g^k). The result retains the same concrete "
            "group and canonical irreducible-character table. Supported groups "
            "are the trivial group, cyclic groups of order at most 60, and S3. "
            "Repeated-power work is admitted from the bit length of k, the "
            "conjugacy-class count, and the permutation degree."
        ),
        request_type=AdamsOperationRequest,
        result_type=CharacterRingElement,
        run=character_adams_operation,
        tags=("finite-group", "character", "representation-ring", "exact"),
        discovery_terms=(
            "Adams operation on a character",
            "power operation in a representation ring",
            "character evaluated on kth powers",
        ),
        examples=(
            OperationExample(
                name="s3_standard_adams_square",
                description=(
                    "The second Adams operation of the standard S3 character is "
                    "the trivial character minus the sign character plus the standard character."
                ),
                input={
                    "character": {
                        "table": {
                            "partition": {
                                "source": {
                                    "degree": 3,
                                    "generators": [[1, 2, 0], [1, 0, 2]],
                                },
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
                                "group": {
                                    "degree": 3,
                                    "generators": [[1, 2, 0], [1, 0, 2]],
                                },
                                "class_representatives": [
                                    [0, 1, 2],
                                    [0, 2, 1],
                                    [1, 2, 0],
                                ],
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
                        "irreducible_multiplicities": [0, 0, 1],
                    },
                    "exponent": "2",
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
