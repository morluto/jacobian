"""Public declaration of exact bounded character-valued basis construction."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.modular_forms.character_basis import (
    modular_character_basis_q_expansions,
    modular_character_coordinates_hecke,
    modular_character_coordinates_product,
    modular_character_coordinates_q_expansion,
    modular_character_hecke_matrix,
)
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    ModularCharacterBasis,
    ModularCharacterBasisRequest,
    ModularCharacterCoordinatesProductRequest,
    ModularCharacterCoordinatesRequest,
    ModularCharacterHeckeMatrix,
    ModularCharacterHeckeMatrixRequest,
    ModularCharacterHeckeRequest,
    ModularCharacterQExpansion,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormFieldQExpansion,
)


def _compute(request: ModularCharacterBasisRequest) -> ModularCharacterBasis:
    return modular_character_basis_q_expansions(request.space)


def _coordinates_q_expansion(
    request: ModularCharacterCoordinatesRequest,
) -> ModularCharacterQExpansion:
    return modular_character_coordinates_q_expansion(request.form)


def _hecke(request: ModularCharacterHeckeRequest) -> ModularFormCoordinates:
    return modular_character_coordinates_hecke(request.form, request.index)


def _hecke_matrix(
    request: ModularCharacterHeckeMatrixRequest,
) -> ModularCharacterHeckeMatrix:
    return modular_character_hecke_matrix(request.space, request.index)


def _product(
    request: ModularCharacterCoordinatesProductRequest,
) -> ModularFormFieldQExpansion:
    return modular_character_coordinates_product(request.left, request.right)


def _character_form_example(coordinate: int = 2) -> dict[str, object]:
    return {
        "space": {
            "group": "GAMMA0",
            "level": 13,
            "weight": 2,
            "kind": "S",
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
                "coordinates": [coordinate],
            },
            "coefficient_domain": {
                "domain": "QQ_CYCLOTOMIC",
                "order": 6,
                "generator": "CLASS_OF_X",
            },
        },
        "basis_id": "gamma0-13-even-order6-character-sturm-v1",
        "coordinates": [
            {
                "field": {
                    "domain": "QQ_CYCLOTOMIC",
                    "order": 6,
                    "generator": "CLASS_OF_X",
                },
                "coefficients_ascending": [
                    {"num": "1", "den": "1"},
                    {"num": "0", "den": "1"},
                ],
            }
        ],
    }


TOOLS: MathTools = (
    MathTool(
        operation_id="modular_form.character_hecke_matrix.compute",
        title="Compute a Hecke matrix on a character-valued modular-form space",
        description=(
            "Return the exact 1-by-1 T_n matrix in the canonical Sturm basis of "
            "either admitted one-dimensional S2(Gamma0(13), chi) space over "
            "Q(zeta_6), for n at most 32 coprime to 13. The matrix preserves "
            "the exact source space, basis identifier, and coefficient field."
        ),
        request_type=ModularCharacterHeckeMatrixRequest,
        result_type=ModularCharacterHeckeMatrix,
        run=_hecke_matrix,
        tags=("modular-forms", "characters", "hecke", "matrix", "exact"),
        examples=(
            OperationExample(
                name="hecke_matrix_t2_order6_character",
                description="Compute T_2 in the canonical basis of S2(Gamma0(13), chi).",
                input={"space": _character_form_example()["space"], "index": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.character_coordinates.product.compute",
        title="Multiply conjugate character-valued modular forms",
        description=(
            "Multiply rational scalar multiples of the two conjugate one-dimensional "
            "S2(Gamma0(13), chi) forms. Return the exact target q-prefix through "
            "the Sturm bound in S4(Gamma0(13)) with trivial character over Q(zeta_6)."
        ),
        request_type=ModularCharacterCoordinatesProductRequest,
        result_type=ModularFormFieldQExpansion,
        run=_product,
        tags=("modular-forms", "characters", "product", "exact"),
        examples=(
            OperationExample(
                name="multiply_conjugate_level13_character_forms",
                description=(
                    "Multiply the normalized order-6 character form by its Galois conjugate."
                ),
                input={
                    "left": _character_form_example(2),
                    "right": _character_form_example(10),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.character_coordinates.hecke.apply",
        title="Apply a Hecke operator to a character-valued modular form",
        description=(
            "Apply T_n for bounded indices coprime to 13 to either admitted "
            "one-dimensional S2(Gamma0(13), chi) space over Q(zeta_6). The "
            "PARI basis prefix is extended as required; the action is checked "
            "through the exact Sturm bound and returned in the same space."
        ),
        request_type=ModularCharacterHeckeRequest,
        result_type=ModularFormCoordinates,
        run=_hecke,
        tags=("modular-forms", "characters", "hecke", "exact"),
        examples=(
            OperationExample(
                name="hecke_t2_order6_character",
                description="Apply T_2 in S2(Gamma0(13), chi) over Q(zeta_6).",
                input={"form": _character_form_example(), "index": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.character_basis.compute",
        title="Construct an exact character-valued modular-form basis",
        description=(
            "Return the exact q-Sturm RREF basis over Q(zeta_6) for weight-two "
            "M or S spaces with an even character of conductor 13 and order "
            "dividing six at "
            "levels 13, 26, or 39. Dimensions are established by the bounded "
            "Cohen-Oesterle formula and checked against PARI; the result retains "
            "the exact character and coefficient-field parents."
        ),
        request_type=ModularCharacterBasisRequest,
        result_type=ModularCharacterBasis,
        run=_compute,
        tags=("modular-forms", "characters", "basis", "exact"),
        examples=(
            OperationExample(
                name="level13_order6_character_basis",
                description=(
                    "Construct the Sturm-determining q-prefix for an even "
                    "order-6 character modulo 13."
                ),
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 13,
                        "weight": 2,
                        "kind": "S",
                        "character": {
                            "group": {
                                "modulus": 13,
                                "unit_residues": [
                                    1,
                                    2,
                                    3,
                                    4,
                                    5,
                                    6,
                                    7,
                                    8,
                                    9,
                                    10,
                                    11,
                                    12,
                                ],
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
                            "coordinates": [2],
                        },
                        "coefficient_domain": {
                            "domain": "QQ_CYCLOTOMIC",
                            "order": 6,
                            "generator": "CLASS_OF_X",
                        },
                    }
                },
            ),
            OperationExample(
                name="level13_order3_character_basis",
                description=(
                    "Construct the exact q-Sturm basis for an even order-3 "
                    "character using the same explicit Q(zeta_6) parent."
                ),
                input={"space": _character_form_example(4)["space"]},
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.character_coordinates.q_expansion.compute",
        title="Compute a character form's exact Sturm prefix",
        description=(
            "Realize one represented form in the admitted S2(Gamma0(13), chi) "
            "basis over Q(zeta_6), returning its exact q^0 through q^2 "
            "coefficients with the source space and field retained."
        ),
        request_type=ModularCharacterCoordinatesRequest,
        result_type=ModularCharacterQExpansion,
        run=_coordinates_q_expansion,
        tags=("modular-forms", "characters", "q-expansion", "exact"),
        examples=(
            OperationExample(
                name="twice_level13_character_form",
                description="Return the exact Sturm prefix of twice the normalized character basis form.",
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 13,
                            "weight": 2,
                            "kind": "S",
                            "character": {
                                "group": {
                                    "modulus": 13,
                                    "unit_residues": [
                                        1,
                                        2,
                                        3,
                                        4,
                                        5,
                                        6,
                                        7,
                                        8,
                                        9,
                                        10,
                                        11,
                                        12,
                                    ],
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
                                "coordinates": [2],
                            },
                            "coefficient_domain": {
                                "domain": "QQ_CYCLOTOMIC",
                                "order": 6,
                                "generator": "CLASS_OF_X",
                            },
                        },
                        "basis_id": "gamma0-13-even-order6-character-sturm-v1",
                        "coordinates": [
                            {
                                "field": {
                                    "domain": "QQ_CYCLOTOMIC",
                                    "order": 6,
                                    "generator": "CLASS_OF_X",
                                },
                                "coefficients_ascending": [
                                    {"num": "2", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            }
                        ],
                    }
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
