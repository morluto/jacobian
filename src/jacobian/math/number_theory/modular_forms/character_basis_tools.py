"""Public declaration of exact bounded character-valued basis construction."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.modular_forms.character_basis import (
    modular_character_basis_q_expansions,
    modular_character_coordinates_hecke,
    modular_character_coordinates_product,
    modular_character_coordinates_q_expansion,
    modular_character_coordinates_u_prime,
    modular_character_hecke_matrix,
)
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    ModularCharacterBasis,
    ModularCharacterBasisRequest,
    ModularCharacterCoordinates,
    ModularCharacterCoordinatesProductRequest,
    ModularCharacterCoordinatesRequest,
    ModularCharacterCoordinatesTransportRequest,
    ModularCharacterEqualityRequest,
    ModularCharacterEqualityResult,
    ModularCharacterHeckeMatrix,
    ModularCharacterHeckeMatrixRequest,
    ModularCharacterHeckeRequest,
    ModularCharacterQExpansion,
    ModularCharacterTransportedForm,
    ModularCharacterUPrimeRequest,
)
from jacobian.math.number_theory.modular_forms.character_transport import (
    modular_character_coordinates_equal_in_common_space,
    modular_character_coordinates_transport,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormFieldQExpansion,
)


def _compute(request: ModularCharacterBasisRequest) -> ModularCharacterBasis:
    return modular_character_basis_q_expansions(request.space, request.precision)


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


def _u_prime(
    request: ModularCharacterUPrimeRequest,
) -> ModularCharacterCoordinates:
    return modular_character_coordinates_u_prime(request.form, request.prime)


def _product(
    request: ModularCharacterCoordinatesProductRequest,
) -> ModularFormFieldQExpansion:
    return modular_character_coordinates_product(request.left, request.right)


def _transport(
    request: ModularCharacterCoordinatesTransportRequest,
) -> ModularCharacterTransportedForm:
    return modular_character_coordinates_transport(request.form, request.inclusion)


def _global_equal(
    request: ModularCharacterEqualityRequest,
) -> ModularCharacterEqualityResult:
    return modular_character_coordinates_equal_in_common_space(
        request.left, request.right
    )


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


def _character_u_form_example() -> dict[str, object]:
    """One serialized generalized-coordinate value in S2(Gamma0(26), chi)."""
    field = {
        "domain": "QQ_CYCLOTOMIC",
        "order": 6,
        "generator": "CLASS_OF_X",
    }
    return {
        "form": {
            "space": {
                "group": "GAMMA0",
                "level": 26,
                "weight": 2,
                "kind": "S",
                "character": {
                    "group": {
                        "modulus": 26,
                        "unit_residues": [
                            1,
                            3,
                            5,
                            7,
                            9,
                            11,
                            15,
                            17,
                            19,
                            21,
                            23,
                            25,
                        ],
                        "character_count": 12,
                        "invariant_factors": [12],
                        "generators": [15],
                        "generator_orders": [12],
                        "unit_coordinates": [
                            [0],
                            [4],
                            [9],
                            [11],
                            [8],
                            [7],
                            [1],
                            [2],
                            [5],
                            [3],
                            [10],
                            [6],
                        ],
                        "exponent": 12,
                    },
                    "coordinates": [2],
                },
                "coefficient_domain": field,
            },
            "basis_id": "gamma0-cyclotomic-character-sturm-rref-v1",
            "coordinates": [
                {
                    "field": field,
                    "coefficients_ascending": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
                {
                    "field": field,
                    "coefficients_ascending": [
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
            ],
        },
        "prime": 2,
    }


TOOLS: MathTools = (
    MathTool(
        operation_id="modular_form.character_coordinates.transport.compute",
        title="Transport a character form through explicit inflation",
        description=(
            "Map an S2 character form from level 13, 26, or 39 into an explicit "
            "level-13 identity, nested level-26/39, or level-78 cusp space. The request carries "
            "the explicit Dirichlet-character inflation and identity Q(zeta_6) "
            "field map. For level 26 or 39, return target coordinates and the "
            "exact q-prefix; for levels 13 and 78, return a typed target-bound "
            "q-prefix through its exact Sturm precision without claiming target basis "
            "coordinates. Source expansion height, work, and output are admitted "
            "before any PARI basis materialization."
        ),
        request_type=ModularCharacterCoordinatesTransportRequest,
        result_type=ModularCharacterTransportedForm,
        run=_transport,
        tags=("modular-forms", "characters", "transport", "exact"),
    ),
    MathTool(
        operation_id="modular_form.character.equal.check",
        title="Check global equality in a common character space",
        description=(
            "Compare two source forms after checking their explicit order-six "
            "character inflations and identical Q(zeta_6) maps into their least "
            "common S2 target at level 13, 26, 39, or 78. Recompute each source "
            "expansion through the common target's full Sturm precision and "
            "compare exact coefficients; the retained target prefix must match "
            "the source inclusion. Levels 13 and 78 use typed target-bound prefixes, "
            "not unvalidated target coordinates."
        ),
        request_type=ModularCharacterEqualityRequest,
        result_type=ModularCharacterEqualityResult,
        run=_global_equal,
        tags=("modular-forms", "characters", "equality", "exact"),
    ),
    MathTool(
        operation_id="modular_form.character_coordinates.u_prime.apply",
        title="Apply a bad-prime U operator to a character cusp form",
        description=(
            "Apply U_p to generalized coordinates in the represented weight-two "
            "S(Gamma0(N), chi) spaces over Q(zeta_6), for (N,p) equal to "
            "(26,2), (26,13), (39,3), or (39,13). These primes divide the "
            "declared level, so the image remains in the same character space. "
            "The operation uses a_n(U_p f)=a_(p n)(f), admits exactly "
            "p*(B-1)+1 source coefficients for target Sturm precision B and "
            "the exact coefficient envelope before "
            "PARI basis expansion, then returns same-parent coordinates only "
            "after exact reconstruction through the target Sturm bound."
        ),
        request_type=ModularCharacterUPrimeRequest,
        result_type=ModularCharacterCoordinates,
        run=_u_prime,
        tags=("modular-forms", "characters", "u-operator", "exact"),
        examples=(
            OperationExample(
                name="u2_on_level26_character_form",
                description=(
                    "Apply U_2 to the first Sturm-basis form in the represented "
                    "S2(Gamma0(26), chi) space."
                ),
                input=_character_u_form_example(),
            ),
        ),
    ),
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
            "M or S spaces with an even order-6 character of conductor 13 at "
            "levels 13, 26, or 39. Dimensions are established by the bounded "
            "Cohen-Oesterle formula and checked against PARI; the result retains "
            "the exact character and coefficient-field parents. An optional "
            "precision may extend the canonical basis through a nested target's "
            "Sturm bound; it must be at least the source bound and at most 128."
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
