"""Public declaration of exact bounded character-valued basis construction."""

import json
from itertools import product
from math import gcd
from typing import cast

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_value,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms.character_basis import (
    modular_character_basis_q_expansions,
    modular_character_coordinates_hecke,
    modular_character_coordinates_product,
    modular_character_coordinates_q_expansion,
    modular_character_hecke_matrix,
)
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    CyclotomicCharacterMap,
    CyclotomicIdentityFieldMap,
    ModularCharacterBasis,
    ModularCharacterBasisRequest,
    ModularCharacterCoordinatesProductRequest,
    ModularCharacterCoordinatesRequest,
    ModularCharacterCoordinatesTransportRequest,
    ModularCharacterEqualityRequest,
    ModularCharacterEqualityResult,
    ModularCharacterHeckeMatrix,
    ModularCharacterHeckeMatrixRequest,
    ModularCharacterHeckeRequest,
    ModularCharacterQExpansion,
    ModularCharacterSpaceInclusion,
    ModularCharacterTransportedForm,
)
from jacobian.math.number_theory.modular_forms.character_transport import (
    modular_character_coordinates_equal_in_common_space,
    modular_character_coordinates_transport,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormFieldQExpansion,
    ModularFormSpace,
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


def _transport_example_values() -> tuple[
    ModularFormCoordinates,
    ModularFormSpace,
    ModularCharacterSpaceInclusion,
]:
    source_form = ModularFormCoordinates.model_validate_json(
        json.dumps(_character_form_example(2))
    )
    source_character = cast(DirichletCharacter, source_form.space.character)
    target_group = character_group(26)
    target_character = None
    for coordinates in product(
        *(range(order) for order in target_group.generator_orders)
    ):
        candidate = dirichlet_character(target_group, coordinates)
        if all(
            dirichlet_character_value(candidate, residue).value
            == dirichlet_character_value(source_character, residue).value
            for residue in range(26)
            if gcd(residue, 26) == 1
        ):
            target_character = candidate
            break
    if target_character is None:
        raise RuntimeError("the documented character has no level-26 inflation")
    target_space = source_form.space.model_copy(
        update={"level": 26, "character": target_character}
    )
    inclusion = ModularCharacterSpaceInclusion(
        source_space=source_form.space,
        target_space=target_space,
        character_map=CyclotomicCharacterMap(
            source=source_character, target=target_character
        ),
        coefficient_field_map=CyclotomicIdentityFieldMap(
            source=cast(RationalCyclotomicField, source_form.space.coefficient_domain),
            target=cast(RationalCyclotomicField, target_space.coefficient_domain),
        ),
    )
    return source_form, target_space, inclusion


def _transport_example() -> dict[str, object]:
    form, _target, inclusion = _transport_example_values()
    return {
        "form": form.model_dump(mode="json"),
        "inclusion": inclusion.model_dump(mode="json"),
    }


def _equality_example() -> dict[str, object]:
    form, target_space, inclusion = _transport_example_values()
    field = cast(RationalCyclotomicField, target_space.coefficient_domain)

    def element(real: int, zeta: int = 0) -> RationalCyclotomicElement:
        return RationalCyclotomicElement(
            field=field,
            coefficients_ascending=(
                CanonicalRational(num=real, den=1),
                CanonicalRational(num=zeta, den=1),
            ),
        )

    coordinates = ModularFormCoordinates(
        space=target_space,
        basis_id="gamma0-cyclotomic-character-sturm-rref-v1",
        coordinates=(element(1), element(-1, -1)),
    )
    q_coefficients = tuple(
        element(real, zeta)
        for real, zeta in (
            (0, 0),
            (1, 0),
            (-1, -1),
            (-2, 2),
            (0, 1),
            (1, -2),
            (4, -2),
            (0, 0),
        )
    )
    target_q_expansion = ModularCharacterQExpansion(
        space=target_space,
        basis_id="gamma0-cyclotomic-character-sturm-rref-v1",
        coefficients=q_coefficients,
    )
    from_level_13 = ModularCharacterTransportedForm(
        source_form=form,
        inclusion=inclusion,
        target_form=coordinates,
        target_q_expansion=target_q_expansion,
    )
    target_character = cast(DirichletCharacter, target_space.character)
    target_field = cast(RationalCyclotomicField, target_space.coefficient_domain)
    identity_inclusion = ModularCharacterSpaceInclusion(
        source_space=target_space,
        target_space=target_space,
        character_map=CyclotomicCharacterMap(
            source=target_character, target=target_character
        ),
        coefficient_field_map=CyclotomicIdentityFieldMap(
            source=target_field,
            target=target_field,
        ),
    )
    from_level_26 = ModularCharacterTransportedForm(
        source_form=coordinates,
        inclusion=identity_inclusion,
        target_form=coordinates,
        target_q_expansion=target_q_expansion,
    )
    return {
        "left": from_level_13.model_dump(mode="json"),
        "right": from_level_26.model_dump(mode="json"),
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
        examples=(
            OperationExample(
                name="inflate_level13_character_form_to_level26",
                description=(
                    "Transport the normalized level-13 order-six character form "
                    "through its explicit level-26 inflation."
                ),
                input=_transport_example(),
            ),
        ),
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
        examples=(
            OperationExample(
                name="compare_two_representations_of_one_character_form",
                description=(
                    "Check equality after explicitly transporting the same "
                    "level-13 form into one level-26 character space."
                ),
                input=_equality_example(),
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
            "M or S spaces with an even character of conductor 13 and order "
            "dividing six at "
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
