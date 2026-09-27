"""Public declaration for exact Lie-algebra basis transport."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.lie_algebras.basis_transport._models import (
    LieBasisChangeRequest,
    LieBasisChangeResult,
)
from jacobian.math.lie_algebras.basis_transport.operations import (
    lie_algebra_change_basis,
)


def _run(request: LieBasisChangeRequest) -> LieBasisChangeResult:
    return lie_algebra_change_basis(request.algebra, request.basis, request.matrix)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="lie_algebra.basis_change.compute",
        title="Transport a Lie algebra to a new ordered basis",
        description=(
            "Compute exact QQ structure constants in a caller-specified new basis. "
            "The columns of the supplied square matrix are the new basis vectors "
            "in source coordinates. Return the source, transported target, and "
            "both inverse coordinate maps. Dimension is at most 8; rational "
            "matrix entries are limited to 128 decimal digits, and exact "
            "inversion, bracket transport, coefficient growth, and output are "
            "admitted before matrix expansion."
        ),
        request_type=LieBasisChangeRequest,
        result_type=LieBasisChangeResult,
        run=_run,
        tags=(
            "lie-algebra",
            "basis-change",
            "structure-constants",
            "exact",
            "rational",
        ),
        discovery_terms=(
            "change basis of a Lie algebra",
            "transport Lie structure constants",
            "Lie algebra basis isomorphism",
            "basis change matrix for structure constants",
        ),
        examples=(
            OperationExample(
                name="sl2_rational_basis_transport",
                description="Transport sl2 to the basis (e+h, f/2, h).",
                input={
                    "algebra": {
                        "basis": ["e", "f", "h"],
                        "structure_constants": [
                            {
                                "i": 0,
                                "j": 1,
                                "k": 2,
                                "coefficient": {"num": 1, "den": 1},
                            },
                            {
                                "i": 0,
                                "j": 2,
                                "k": 0,
                                "coefficient": {"num": -2, "den": 1},
                            },
                            {
                                "i": 1,
                                "j": 2,
                                "k": 1,
                                "coefficient": {"num": 2, "den": 1},
                            },
                        ],
                    },
                    "basis": ["u", "v", "w"],
                    "matrix": {
                        "domain": "QQ",
                        "row_count": 3,
                        "column_count": 3,
                        "entries": [
                            [
                                {"num": 1, "den": 1},
                                {"num": 0, "den": 1},
                                {"num": 0, "den": 1},
                            ],
                            [
                                {"num": 0, "den": 1},
                                {"num": 1, "den": 2},
                                {"num": 0, "den": 1},
                            ],
                            [
                                {"num": 1, "den": 1},
                                {"num": 0, "den": 1},
                                {"num": 1, "den": 1},
                            ],
                        ],
                    },
                },
            ),
        ),
    ),
)
