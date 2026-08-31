"""Public operations for exact lattice-presented complex tori."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.geometry.complex_tori._models import (
    NeronSeveriLatticeRequest,
    RiemannFormProfile,
    RiemannFormProfileRequest,
)
from jacobian.math.geometry.complex_tori.operations import (
    compute_neron_severi_lattice as _compute_native,
)
from jacobian.math.geometry.complex_tori.operations import (
    compute_riemann_form_profile as _compute_riemann_form_profile_native,
)
from jacobian.math.lattices.invariant_forms import InvariantBilinearFormLattice


def compute_neron_severi_lattice(
    request: NeronSeveriLatticeRequest,
) -> InvariantBilinearFormLattice:
    return _compute_native(request.torus)


def compute_riemann_form_profile(
    request: RiemannFormProfileRequest,
) -> RiemannFormProfile:
    return _compute_riemann_form_profile_native(request.torus, request.form)


NERON_SEVERI_LATTICE_OPERATION = MathTool(
    operation_id="complex_torus.neron_severi_lattice.compute",
    title="Compute the exact Neron-Severi lattice of a complex torus",
    description=(
        "For a bounded labelled lattice and an exact rational or common-embedding "
        "real algebraic matrix J satisfying J^2 = -I, return the complete "
        "saturated row-Hermite lattice of integral alternating forms E satisfying "
        "J^T E J = E. This is the integral Hodge (1,1), or Neron-Severi, lattice "
        "of the selected exact torus; symbolic and very-general family claims are "
        "outside this operation's result."
    ),
    request_type=NeronSeveriLatticeRequest,
    result_type=InvariantBilinearFormLattice,
    run=compute_neron_severi_lattice,
    tags=("complex-torus", "Neron-Severi", "Hodge", "lattice", "exact"),
    discovery_terms=(
        "integral (1,1) classes",
        "Neron-Severi rank of a complex torus",
        "Hodge alternating forms",
    ),
    examples=(
        example(
            "elliptic_curve_hodge_lattice",
            "Compute the rank-one Hodge lattice for the standard elliptic torus.",
            {
                "torus": {
                    "coordinate_axis": ["e1", "e2"],
                    "complex_structure": {
                        "domain": "QQ",
                        "entries": [
                            [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                            [
                                {"num": "-1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                        ],
                    },
                }
            },
        ),
    ),
)


RIEMANN_FORM_PROFILE_OPERATION = MathTool(
    operation_id="complex_torus.riemann_form.profile.compute",
    title="Compute the exact profile of an integral Riemann-form candidate",
    description=(
        "For a bounded exact lattice complex structure J and a selected integral "
        "alternating form E, return its Smith and alternating elementary divisors, "
        "decide the Hodge identity J^T E J = E, and, when it holds, return the "
        "exact real and Hermitian inertia of G = J^T E = -EJ. A positive-definite "
        "G is a Riemann form and its elementary divisors are the polarization type."
    ),
    request_type=RiemannFormProfileRequest,
    result_type=RiemannFormProfile,
    run=compute_riemann_form_profile,
    tags=("complex-torus", "Riemann-form", "polarization", "Hermitian", "exact"),
    discovery_terms=(
        "Riemann bilinear relations",
        "polarization type of a complex torus",
        "Hermitian signature of an integral alternating form",
    ),
    examples=(
        example(
            "elliptic_degree_six_polarization",
            "Profile the degree-six positive form on the standard elliptic torus.",
            {
                "torus": {
                    "coordinate_axis": ["e1", "e2"],
                    "complex_structure": {
                        "domain": "QQ",
                        "entries": [
                            [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                            [
                                {"num": "-1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                        ],
                    },
                },
                "form": {
                    "coordinate_axis": ["e1", "e2"],
                    "kind": "ALTERNATING",
                    "matrix": {
                        "entries": [["0", "6"], ["-6", "0"]],
                    },
                },
            },
        ),
    ),
)

TOOLS: MathTools = (
    NERON_SEVERI_LATTICE_OPERATION,
    RIEMANN_FORM_PROFILE_OPERATION,
)

__all__ = ["TOOLS"]
