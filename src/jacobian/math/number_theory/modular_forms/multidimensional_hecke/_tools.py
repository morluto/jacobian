"""Public declaration for exact multidimensional character Hecke matrices."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.modular_forms.multidimensional_hecke.models import (
    ModularCharacterHeckeMatrixRequest,
    ModularCharacterHeckeMatrixResult,
)
from jacobian.math.number_theory.modular_forms.multidimensional_hecke.operations import (
    modular_character_hecke_matrix_multidimensional,
)


def _compute(
    request: ModularCharacterHeckeMatrixRequest,
) -> ModularCharacterHeckeMatrixResult:
    return modular_character_hecke_matrix_multidimensional(request.space, request.index)


_SPACE = {
    "group": "GAMMA0",
    "level": 13,
    "weight": 2,
    "kind": "M",
    "character": {
        "group": {
            "modulus": 13,
            "unit_residues": list(range(1, 13)),
            "character_count": 12,
            "invariant_factors": [12],
            "generators": [2],
            "generator_orders": [12],
            "unit_coordinates": [
                [0],
                [1],
                [4],
                [2],
                [9],
                [5],
                [11],
                [3],
                [8],
                [10],
                [7],
                [6],
            ],
            "exponent": 12,
        },
        "coordinates": [6],
    },
    "coefficient_domain": {
        "domain": "QQ_CYCLOTOMIC",
        "order": 6,
        "generator": "CLASS_OF_X",
    },
}

TOOLS: MathTools = (
    MathTool(
        operation_id="modular_form.character_hecke_matrix.multidimensional.compute",
        title="Compute a multidimensional character Hecke matrix",
        description=(
            "Return T_n in the exact q-Sturm RREF basis for an admitted weight-two "
            "Gamma0 character space of dimension at least two over Q(zeta_6), with "
            "Nebentypus values contained in that coefficient field. The operator index "
            "is at most eight and coprime to the level; every image is reconstructed "
            "through the complete Sturm prefix before returning the matrix."
        ),
        request_type=ModularCharacterHeckeMatrixRequest,
        result_type=ModularCharacterHeckeMatrixResult,
        run=_compute,
        tags=("modular-forms", "characters", "hecke", "matrix", "exact"),
        examples=(
            OperationExample(
                name="level13-weight2-character-t2-matrix",
                description=(
                    "Compute T_2 on the two-dimensional full space M2(Gamma0(13), chi_6)."
                ),
                input={"space": _SPACE, "index": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
