"""Finite-coset crossed-product operation declarations."""

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.crossed_products._models import (
    CrossedProductMultiplyRequest,
    CrossedProductMultiplyResult,
)
from jacobian.math.crossed_products._operations import compute_product

_INFINITE_DIHEDRAL_PRESENTATION = {
    "crossed_product_schema_version": "1",
    "characteristic": 5,
    "lattice_basis": ["t"],
    "cosets": ["e", "a"],
    "identity_coset": "e",
    "quotient_multiplication": [["e", "a"], ["a", "e"]],
    "action_matrices": [[["1"]], [["-1"]]],
    "cocycle_table": [[["0"], ["0"]], [["0"], ["0"]]],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="crossed_product.multiply.compute",
        version="1",
        title="Multiply finite-coset crossed-product elements",
        description=(
            "Multiply two canonical sparse elements of F_p[Z^d x_c Q] from a "
            "caller-supplied finite quotient table, left action Q -> GL_d(Z), and "
            "normalized cocycle. Validate all laws before applying "
            "(u,q)(v,r)=(u+rho(q)v+c(q,r),qr). Q<=16, d<=8, support pairs<=1024, "
            "and scalar work<=80000; the source-bound result replays both operands. "
            "This uses explicit normal forms, not a finitely presented group word "
            "problem or an invertibility decision."
        ),
        request_type=CrossedProductMultiplyRequest,
        result_type=CrossedProductMultiplyResult,
        run=compute_product,
        tags=(
            "algebra",
            "crossed-product",
            "group-ring",
            "laurent",
            "finite-field",
            "sparse",
            "exact",
        ),
        examples=(
            example(
                "infinite_dihedral_basis_product",
                "In F_5[Z semidirect C_2], multiply 2*t*a by 3*t^2*a. "
                "The action a(t)=t^-1 gives t^-1 in the identity coset.",
                {
                    "left": {
                        "presentation": _INFINITE_DIHEDRAL_PRESENTATION,
                        "terms": [{"coefficient": 2, "coset": "a", "exponents": ["1"]}],
                    },
                    "right": {
                        "presentation": _INFINITE_DIHEDRAL_PRESENTATION,
                        "terms": [{"coefficient": 3, "coset": "a", "exponents": ["2"]}],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
