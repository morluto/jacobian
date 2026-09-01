"""Public declaration for invariant integral-form lattices."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.lattices.invariant_forms._models import (
    InvariantBilinearFormLattice,
    InvariantBilinearFormLatticeRequest,
)
from jacobian.math.lattices.invariant_forms.operations import (
    compute_invariant_bilinear_form_lattice as _compute_native,
)


def compute_invariant_bilinear_form_lattice(
    request: InvariantBilinearFormLatticeRequest,
) -> InvariantBilinearFormLattice:
    """Unpack the wire request for the canonical native operation."""

    return _compute_native(request.action, request.kind)


INVARIANT_BILINEAR_FORM_LATTICE_OPERATION = MathTool(
    operation_id="lattice.invariant_bilinear_form_lattice.compute",
    title="Compute the lattice of invariant integral bilinear forms",
    description=(
        "For a bounded labelled action by rational or common-embedding real "
        "simple-number-field matrices and a bilinear, symmetric, or alternating "
        "form class, return the complete saturated row-Hermite basis of integral "
        "matrices Q satisfying A^T Q A = Q for every generator. The exact result "
        "retains the action, coefficient order, constraint rank, and the rank-zero "
        "lattice."
    ),
    request_type=InvariantBilinearFormLatticeRequest,
    result_type=InvariantBilinearFormLattice,
    run=compute_invariant_bilinear_form_lattice,
    tags=(
        "lattice",
        "bilinear-form",
        "matrix-action",
        "congruence",
        "exact-integer",
        "saturation",
    ),
    discovery_terms=(
        "invariant bilinear forms",
        "invariant alternating forms",
        "invariant symmetric forms",
        "fixed forms of a matrix action",
    ),
    examples=(
        OperationExample(
            name="invariant_area_form",
            description="Computes the row-Hermite basis of integral matrices Q "
            "satisfying A^T Q A = Q for the generator A. Every generator "
            "matrix must be square on coordinate_axis.",
            input={
                "action": {
                    "action_type": "RATIONAL",
                    "coordinate_axis": ["e1", "e2"],
                    "generators": [
                        {
                            "label": "T",
                            "matrix": {
                                "domain": "QQ",
                                "entries": [
                                    [
                                        {"num": "1", "den": "1"},
                                        {"num": "1", "den": "1"},
                                    ],
                                    [
                                        {"num": "0", "den": "1"},
                                        {"num": "1", "den": "1"},
                                    ],
                                ],
                            },
                        }
                    ],
                },
                "kind": "ALTERNATING",
            },
        ),
    ),
)

TOOLS: MathTools = (INVARIANT_BILINEAR_FORM_LATTICE_OPERATION,)

__all__ = ["TOOLS"]
