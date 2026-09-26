"""Public declaration for exact integral quadratic-form parity profiles."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.quadratic_forms.integral.parity._models import (
    ParityProfile,
    ParityProfileRequest,
)
from jacobian.math.number_theory.quadratic_forms.integral.parity.operations import (
    parity_profile,
)


def _run_parity_profile(request: ParityProfileRequest) -> ParityProfile:
    return parity_profile(request.form)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="quadratic_form.parity_profile.compute",
        title="Compute an integral quadratic-form parity profile",
        description=(
            "Reduce an integral polynomial coefficient-wise into F2. The "
            "diagonal residues are Q(e_i); each mixed coefficient residue "
            "is the polar pairing B(e_i,e_j)=Q(e_i+e_j)-Q(e_i)-Q(e_j). "
            "The all_basis_norms_even flag concerns basis vectors only and "
            "does not decide whether every lattice vector has even norm."
        ),
        request_type=ParityProfileRequest,
        result_type=ParityProfile,
        run=_run_parity_profile,
        tags=("quadratic-form", "integral", "parity", "exact"),
        discovery_terms=(
            "integral quadratic form coefficient residues modulo two",
            "quadratic form basis norm parity",
            "polar cross-term residues over F2",
        ),
        examples=(
            OperationExample(
                name="diagonal_norm_and_polar_residues",
                description=(
                    "For Q(x,y)=2x^2+3xy+y^2, Q(e_x) and Q(e_y) are even "
                    "while B(e_x,e_y) has residue one."
                ),
                input={
                    "form": {
                        "axis": ["x", "y"],
                        "diagonal_coefficients": ["2", "1"],
                        "cross_terms": [{"left": 0, "right": 1, "coefficient": "3"}],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
