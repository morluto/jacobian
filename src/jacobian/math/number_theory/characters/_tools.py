"""Catalog declarations for bounded principal Dirichlet characters."""

from __future__ import annotations

from typing import Literal, cast

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.characters import operations as native
from jacobian.math.number_theory.characters._models import (
    CharacterGroupRequest,
    DirichletCharacterConductorRequest,
    DirichletCharacterConductorResult,
    DirichletCharacterConjugateRequest,
    DirichletCharacterEnumerationRequest,
    DirichletCharacterFourierMatrixRequest,
    DirichletCharacterGaussSumRequest,
    DirichletCharacterGeneralizedBernoulliPrefix,
    DirichletCharacterGeneralizedBernoulliPrefixRequest,
    DirichletCharacterGeneralizedBernoulliRequest,
    DirichletCharacterGeneralizedGaussSumRequest,
    DirichletCharacterInflationRequest,
    DirichletCharacterInverseRequest,
    DirichletCharacterJacobiSumRequest,
    DirichletCharacterKernelRequest,
    DirichletCharacterLValueNonpositiveRequest,
    DirichletCharacterOrderRequest,
    DirichletCharacterOrthogonalityRequest,
    DirichletCharacterParityRequest,
    DirichletCharacterPowerRequest,
    DirichletCharacterPrimitiveGaussNormRequest,
    DirichletCharacterProductRequest,
    DirichletCharacterRequest,
    DirichletCharacterResidueIndicatorExpansion,
    DirichletCharacterResidueIndicatorRequest,
    DirichletCharacterRestrictionRequest,
    DirichletCharacterSequenceTwistRequest,
    DirichletCharacterTableResult,
    DirichletCharacterValueRequest,
    DirichletCharacterValueResult,
    PrincipalDirichletCharacterRequest,
    PrincipalDirichletCharacterValueRequest,
    PrincipalDirichletCharacterValueResult,
)
from jacobian.math.number_theory.characters.values import (
    DirichletCharacter,
    DirichletCharacterFamily,
    DirichletCharacterFourierMatrix,
    DirichletCharacterGroup,
    DirichletCharacterInflation,
    DirichletCharacterKernel,
    DirichletCharacterRestrictionResult,
    PrincipalDirichletCharacter,
)
from jacobian.math.number_theory.sequences.core import FiniteCyclotomicSequence


def compute_principal_dirichlet_character(
    request: PrincipalDirichletCharacterRequest,
) -> PrincipalDirichletCharacter:
    """Materialize the complete exact principal character for one modulus."""

    return native.principal_dirichlet_character(request.modulus)


def compute_principal_dirichlet_character_value(
    request: PrincipalDirichletCharacterValueRequest,
) -> PrincipalDirichletCharacterValueResult:
    """Evaluate one principal Dirichlet character at a source-bound integer."""

    value = native.principal_dirichlet_character_value(
        request.character, request.integer_value()
    )
    residue = request.integer_value() % request.character.modulus
    return PrincipalDirichletCharacterValueResult._from_kernel(
        character=request.character,
        integer=request.integer,
        canonical_residue=residue,
        is_unit=value == 1,
        value=cast(Literal[0, 1], value),
    )


def compute_character_group(request: CharacterGroupRequest) -> DirichletCharacterGroup:
    """Compute the finite unit-group decomposition for one modulus."""

    return native.character_group(request.modulus)


def compute_character_family(
    request: DirichletCharacterEnumerationRequest,
) -> DirichletCharacterFamily:
    """Enumerate all exact dual coordinates of one canonical group."""
    return native.dirichlet_character_group_enumerate(request.group)


def compute_character_fourier_matrix(
    request: DirichletCharacterFourierMatrixRequest,
) -> DirichletCharacterFourierMatrix:
    """Compute the exact complete Fourier matrix for the supplied unit group."""

    return native.dirichlet_character_fourier_matrix(request.group)


def compute_residue_indicator_expansion(
    request: DirichletCharacterResidueIndicatorRequest,
) -> DirichletCharacterResidueIndicatorExpansion:
    return native.dirichlet_character_residue_indicator_expansion(
        request.group, request.residue
    )


def _compute_character_inflation(
    request: DirichletCharacterInflationRequest,
) -> DirichletCharacterInflation:
    return native.dirichlet_character_inflate(request.character, request.target_modulus)


def _compute_character_restriction(
    request: DirichletCharacterRestrictionRequest,
) -> DirichletCharacterRestrictionResult:
    return native.dirichlet_character_restrict_modulus(
        request.character, request.target_modulus
    )


_GROUP_MOD3 = {
    "modulus": 3,
    "unit_residues": [1, 2],
    "character_count": 2,
    "invariant_factors": [2],
    "generators": [2],
    "generator_orders": [2],
    "unit_coordinates": [[0], [1]],
    "exponent": 2,
}

_GROUP_MOD5 = {
    "modulus": 5,
    "unit_residues": [1, 2, 3, 4],
    "character_count": 4,
    "invariant_factors": [4],
    "generators": [2],
    "generator_orders": [4],
    "unit_coordinates": [[0], [1], [3], [2]],
    "exponent": 4,
}


def _compute_character(request: DirichletCharacterRequest) -> DirichletCharacter:
    return native.dirichlet_character(request.group, request.coordinates)


def _compute_character_value(
    request: DirichletCharacterValueRequest,
) -> DirichletCharacterValueResult:
    return native.dirichlet_character_value(request.character, int(request.integer))


def _compute_character_product(
    request: DirichletCharacterProductRequest,
) -> DirichletCharacter:
    return native.dirichlet_character_product(request.left, request.right)


def _compute_character_order(request: DirichletCharacterOrderRequest):
    return native.dirichlet_character_order(request.character)


def _compute_character_kernel(request: DirichletCharacterKernelRequest):
    return native.dirichlet_character_kernel(request.character)


def _compute_character_parity(request: DirichletCharacterParityRequest):
    return native.dirichlet_character_parity(request.character)


def _compute_character_power(request: DirichletCharacterPowerRequest):
    return native.dirichlet_character_power(request.character, int(request.exponent))


def _compute_character_conjugate(
    request: DirichletCharacterConjugateRequest,
) -> DirichletCharacter:
    return native.dirichlet_character_conjugate(request.character)


def _compute_character_inverse(
    request: DirichletCharacterInverseRequest,
) -> DirichletCharacter:
    return native.dirichlet_character_inverse(request.character)


def _compute_character_table(
    request: DirichletCharacterRequest,
) -> DirichletCharacterTableResult:
    return native.dirichlet_character_table(
        native.dirichlet_character(request.group, request.coordinates)
    )


def _compute_character_conductor(
    request: DirichletCharacterConductorRequest,
) -> DirichletCharacterConductorResult:
    return native.dirichlet_character_conductor(request.character)


def _compute_gauss_sum(request: DirichletCharacterGaussSumRequest):
    return native.dirichlet_character_gauss_sum(request.character)


def _compute_primitive_gauss_norm(request: DirichletCharacterPrimitiveGaussNormRequest):
    return native.dirichlet_character_primitive_gauss_norm(request.character)


def _compute_generalized_gauss_sum(
    request: DirichletCharacterGeneralizedGaussSumRequest,
):
    return native.dirichlet_character_generalized_gauss_sum(
        request.character, int(request.frequency)
    )


def _compute_jacobi_sum(
    request: DirichletCharacterJacobiSumRequest,
):
    return native.dirichlet_character_jacobi_sum(request.left, request.right)


def _compute_orthogonality(
    request: DirichletCharacterOrthogonalityRequest,
):
    return native.dirichlet_character_orthogonality(request.left, request.right)


def _compute_sequence_twist(
    request: DirichletCharacterSequenceTwistRequest,
) -> FiniteCyclotomicSequence:
    return native.dirichlet_character_sequence_twist(request)


def _compute_generalized_bernoulli(
    request: DirichletCharacterGeneralizedBernoulliRequest,
):
    return native.dirichlet_character_generalized_bernoulli(
        request.character, request.index
    )


def _compute_generalized_bernoulli_prefix(
    request: DirichletCharacterGeneralizedBernoulliPrefixRequest,
) -> DirichletCharacterGeneralizedBernoulliPrefix:
    return native.dirichlet_character_generalized_bernoulli_prefix(
        request.character, request.maximum_index
    )


def _compute_l_value_nonpositive(request: DirichletCharacterLValueNonpositiveRequest):
    return native.dirichlet_character_l_value_nonpositive_integer(
        request.character, request.bernoulli_index
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="dirichlet_character.compute",
        title="Construct an exact Dirichlet character",
        description="Construct one character from exact dual coordinates bound to a finite unit-group parent; coordinates must fit every generator order.",
        request_type=DirichletCharacterRequest,
        result_type=DirichletCharacter,
        run=_compute_character,
        tags=("number-theory", "dirichlet-character", "exact"),
        examples=(
            OperationExample(
                name="quadratic_mod3",
                description="Construct the nonprincipal character modulo 3; coordinates must use the supplied group's dual axis.",
                input={"group": _GROUP_MOD3, "coordinates": [1]},
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.value.compute",
        title="Evaluate an exact Dirichlet character",
        description="Evaluate a character exactly at an integer, returning zero off the unit group and a cyclotomic root with its modulus parent on units.",
        request_type=DirichletCharacterValueRequest,
        result_type=DirichletCharacterValueResult,
        run=_compute_character_value,
        tags=("number-theory", "dirichlet-character", "cyclotomic", "exact"),
        examples=(
            OperationExample(
                name="quadratic_value_mod3",
                description="Evaluate the quadratic character modulo 3 at 2; the character must retain the exact modulus and group parent.",
                input={
                    "character": {"group": _GROUP_MOD3, "coordinates": [1]},
                    "integer": "2",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.conjugate.compute",
        title="Conjugate an exact Dirichlet character",
        description=(
            "Return the complex conjugate character in the identical modulus "
            "and unit-group parent. On units this negates every dual coordinate; "
            "off units both characters remain zero."
        ),
        request_type=DirichletCharacterConjugateRequest,
        result_type=DirichletCharacter,
        run=_compute_character_conjugate,
        tags=("number-theory", "dirichlet-character", "exact"),
        examples=(
            OperationExample(
                name="conjugate_mod5_character",
                description="Conjugate a nontrivial character modulo 5.",
                input={
                    "character": {
                        "group": {
                            "modulus": 5,
                            "unit_residues": [1, 2, 3, 4],
                            "character_count": 4,
                            "invariant_factors": [4],
                            "generators": [2],
                            "generator_orders": [4],
                            "unit_coordinates": [[0], [1], [3], [2]],
                            "exponent": 4,
                        },
                        "coordinates": [1],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.table.compute",
        title="Materialize an exact Dirichlet character table",
        description="Materialize the complete extension-by-zero table of one exact character, retaining modulus and cyclotomic parent identity.",
        request_type=DirichletCharacterRequest,
        result_type=DirichletCharacterTableResult,
        run=_compute_character_table,
        tags=("number-theory", "dirichlet-character", "cyclotomic", "exact"),
        examples=(
            OperationExample(
                name="table_mod3",
                description="Materialize the character table modulo 3; the character coordinates must belong to the supplied exact group.",
                input={"group": _GROUP_MOD3, "coordinates": [1]},
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.conductor.compute",
        title="Compute an exact Dirichlet character conductor",
        description=(
            "Return the least positive divisor d of the source modulus N such "
            "that the character factors through reduction of unit groups "
            "(Z/NZ)^* to (Z/dZ)^*. The exact test checks triviality on each "
            "reduction-kernel subgroup and is bounded by the source unit table. "
            "Also return the inducing character modulo d in its canonical "
            "unit-group coordinates."
        ),
        request_type=DirichletCharacterConductorRequest,
        result_type=DirichletCharacterConductorResult,
        run=_compute_character_conductor,
        tags=("number-theory", "dirichlet-character", "conductor", "exact"),
        examples=(
            OperationExample(
                name="quadratic_character_mod_8_conductor",
                description=(
                    "Compute the conductor of the nonprincipal character modulo "
                    "8; the character must carry complete unit coordinates."
                ),
                input={
                    "character": {
                        "group": {
                            "modulus": 8,
                            "unit_residues": [1, 3, 5, 7],
                            "character_count": 4,
                            "invariant_factors": [2, 2],
                            "generators": [7, 5],
                            "generator_orders": [2, 2],
                            "unit_coordinates": [[0, 0], [1, 1], [0, 1], [1, 0]],
                            "exponent": 2,
                        },
                        "coordinates": [0, 1],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.multiply.compute",
        title="Multiply exact Dirichlet characters",
        description="Multiply two characters pointwise through their common finite dual-group coordinates; both operands must have the identical group parent.",
        request_type=DirichletCharacterProductRequest,
        result_type=DirichletCharacter,
        run=_compute_character_product,
        tags=("number-theory", "dirichlet-character", "exact"),
        examples=(
            OperationExample(
                name="quadratic_square_mod3",
                description="Multiply the quadratic character modulo 3 by itself; both operands must use the identical group parent.",
                input={
                    "left": {"group": _GROUP_MOD3, "coordinates": [1]},
                    "right": {"group": _GROUP_MOD3, "coordinates": [1]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.order.compute",
        title="Compute an exact Dirichlet character order",
        description=(
            "Return the multiplicative order of a character from its exact "
            "dual coordinates: on each cyclic axis of order m, coordinate c "
            "has order m/gcd(c,m), and the character order is their lcm."
        ),
        request_type=DirichletCharacterOrderRequest,
        result_type=native.DirichletCharacterOrderResult,
        run=_compute_character_order,
        tags=("number-theory", "dirichlet-character", "exact"),
        examples=(
            OperationExample(
                name="character_order_mod5",
                description="Compute the order of a generator character modulo 5.",
                input={
                    "character": {
                        "group": {
                            "modulus": 5,
                            "unit_residues": [1, 2, 3, 4],
                            "character_count": 4,
                            "invariant_factors": [4],
                            "generators": [2],
                            "generator_orders": [4],
                            "unit_coordinates": [[0], [1], [3], [2]],
                            "exponent": 4,
                        },
                        "coordinates": [1],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.kernel.compute",
        title="Compute an exact Dirichlet-character kernel",
        description=(
            "Return the complete subgroup of canonical unit residues on which "
            "the supplied character equals one. The result retains the exact "
            "character parent and subgroup index; unit-coordinate evaluation "
            "and serialized output are bounded before construction."
        ),
        request_type=DirichletCharacterKernelRequest,
        result_type=DirichletCharacterKernel,
        run=_compute_character_kernel,
        tags=("number-theory", "dirichlet-character", "exact"),
        examples=(
            OperationExample(
                name="quadratic_kernel_mod5",
                description="Return the kernel of the quadratic character modulo 5.",
                input={
                    "character": {
                        "group": {
                            "modulus": 5,
                            "unit_residues": [1, 2, 3, 4],
                            "character_count": 4,
                            "invariant_factors": [4],
                            "generators": [2],
                            "generator_orders": [4],
                            "unit_coordinates": [[0], [1], [3], [2]],
                            "exponent": 4,
                        },
                        "coordinates": [2],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.parity.compute",
        title="Determine exact Dirichlet-character parity",
        description=(
            "Return chi(-1) as an exact root of unity in the source character "
            "group's cyclotomic parent and derive the EVEN or ODD label from "
            "whether that value is 1 or -1."
        ),
        request_type=DirichletCharacterParityRequest,
        result_type=native.DirichletCharacterParityResult,
        run=_compute_character_parity,
        tags=("number-theory", "dirichlet-character", "exact"),
        examples=(
            OperationExample(
                name="odd_character_mod3",
                description="The nonprincipal character modulo 3 is odd because chi(-1)=-1.",
                input={"character": {"group": _GROUP_MOD3, "coordinates": [1]}},
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.power.compute",
        title="Raise an exact Dirichlet character to an integer power",
        description=(
            "Return chi^k by multiplying each dual coordinate by the bounded "
            "signed integer k modulo its cyclic-axis order. Exponent zero gives "
            "the principal character in the identical group parent."
        ),
        request_type=DirichletCharacterPowerRequest,
        result_type=DirichletCharacter,
        run=_compute_character_power,
        tags=("number-theory", "dirichlet-character", "exact"),
        examples=(
            OperationExample(
                name="character_inverse_power_mod3",
                description="Raise the nonprincipal character modulo 3 to exponent -1.",
                input={
                    "character": {"group": _GROUP_MOD3, "coordinates": [1]},
                    "exponent": "-1",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.principal.compute",
        title="Compute an exact principal Dirichlet character",
        description=(
            "Materialize the complete extension-by-zero table of the principal "
            "Dirichlet character modulo a bounded positive modulus. The returned "
            "canonical value composes directly with exact character evaluation."
        ),
        request_type=PrincipalDirichletCharacterRequest,
        result_type=PrincipalDirichletCharacter,
        run=compute_principal_dirichlet_character,
        tags=("number-theory", "dirichlet-character", "principal", "exact"),
        examples=(
            OperationExample(
                name="principal_character_mod_12",
                description="Compute the complete principal character modulo 12; the modulus must be positive and its full residue table must fit the 2,048-entry bound.",
                input={"modulus": 12},
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.principal.value.compute",
        title="Evaluate a principal Dirichlet character",
        description=(
            "Evaluate the exact principal Dirichlet character at an integer, "
            "retaining its source character and canonical residue."
        ),
        request_type=PrincipalDirichletCharacterValueRequest,
        result_type=PrincipalDirichletCharacterValueResult,
        run=compute_principal_dirichlet_character_value,
        tags=("number-theory", "dirichlet-character", "principal", "exact"),
        examples=(
            OperationExample(
                name="principal_character_value_mod_12",
                description="Evaluate the principal character modulo 12 at 5.",
                input={
                    "character": {
                        "modulus": 12,
                        "unit_residues": [1, 5, 7, 11],
                        "values": [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                    },
                    "integer": "5",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.group.compute",
        title="Compute a finite Dirichlet character group",
        description=(
            "Compute the finite unit group modulo a bounded positive modulus: "
            "the unit set, invariant factors, canonical generators with "
            "residue-coordinate maps, the common root-of-unity exponent, and "
            "the character count equal to phi(modulus)."
        ),
        request_type=CharacterGroupRequest,
        result_type=DirichletCharacterGroup,
        run=compute_character_group,
        tags=("number-theory", "dirichlet-character", "unit-group", "exact"),
        discovery_terms=(
            "Dirichlet character group",
            "unit group decomposition",
        ),
        examples=(
            OperationExample(
                name="character_group_mod_12",
                description="Compute the unit-group decomposition modulo 12; the modulus must be positive and its unit table must fit the 2,048-entry bound.",
                input={"modulus": 12},
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.group.enumerate.compute",
        title="Enumerate a complete finite Dirichlet character group",
        description=(
            "Enumerate every character in the dual of the supplied exact finite "
            "unit-group parent. The result stores the parent once and lists each "
            "dual coordinate tuple in lexicographic order. Complete group, work, "
            "and 10 MiB result bounds are checked before coordinate expansion."
        ),
        request_type=DirichletCharacterEnumerationRequest,
        result_type=DirichletCharacterFamily,
        run=compute_character_family,
        tags=("number-theory", "dirichlet-character", "enumeration", "exact"),
        discovery_terms=(
            "enumerate Dirichlet characters",
            "list every character of a finite unit group",
            "complete dual character group",
        ),
        examples=(
            OperationExample(
                name="complete_dual_mod3",
                description=(
                    "Enumerate both characters modulo 3 in the supplied cyclic "
                    "dual coordinate axis."
                ),
                input={"group": _GROUP_MOD3},
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.group.fourier_matrix.compute",
        title="Compute the exact Dirichlet character Fourier matrix",
        description=(
            "Return the complete matrix with rows indexed by lexicographic dual "
            "character coordinates and columns by canonical increasing unit "
            "residues. Entries are exact exponents in the group's common "
            "cyclotomic root-of-unity parent. Complete work and output bounds "
            "are checked before matrix construction."
        ),
        request_type=DirichletCharacterFourierMatrixRequest,
        result_type=DirichletCharacterFourierMatrix,
        run=compute_character_fourier_matrix,
        tags=("number-theory", "dirichlet-character", "fourier", "exact"),
        discovery_terms=(
            "Dirichlet character table",
            "finite Fourier matrix on units",
            "complete character orthogonality matrix",
        ),
        examples=(
            OperationExample(
                name="character_fourier_matrix_mod3",
                description="Build the exact two by two character matrix modulo 3.",
                input={"group": _GROUP_MOD3},
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.residue_class_indicator_expansion.compute",
        title="Expand a unit residue indicator in Dirichlet characters",
        description=(
            "Return the exact character-basis expansion of the indicator of one "
            "unit residue modulo the supplied group modulus. Coefficients are "
            "cyclotomic roots of unity divided by phi(modulus); work and output "
            "are admitted before constructing the dual coordinate axis."
        ),
        request_type=DirichletCharacterResidueIndicatorRequest,
        result_type=DirichletCharacterResidueIndicatorExpansion,
        run=compute_residue_indicator_expansion,
        tags=("number-theory", "dirichlet-character", "fourier", "exact"),
        discovery_terms=(
            "Dirichlet character expansion of a residue class indicator",
            "finite character orthogonality expansion",
        ),
        examples=(
            OperationExample(
                name="unit_indicator_mod3",
                description=(
                    "Expand the indicator of residue 2 on the units modulo 3; "
                    "its two coefficients are 1/2 and -1/2."
                ),
                input={"group": _GROUP_MOD3, "residue": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.inflate.compute",
        title="Inflate a Dirichlet character to a multiple modulus",
        description=(
            "Compose a source character with reduction of target units modulo "
            "the source modulus, then extend by zero on target nonunits. The "
            "result retains both characters and the canonical target-unit to "
            "source-unit reduction table."
        ),
        request_type=DirichletCharacterInflationRequest,
        result_type=DirichletCharacterInflation,
        run=_compute_character_inflation,
        tags=("number-theory", "dirichlet-character", "inflation", "exact"),
        discovery_terms=(
            "induce a Dirichlet character to a multiple modulus",
            "inflate a Dirichlet character",
            "imprimitive character from a lower modulus",
        ),
        examples=(
            OperationExample(
                name="quadratic_mod3_inflated_to_15",
                description=(
                    "Inflate the nonprincipal character modulo 3 to modulus 15; "
                    "the result is zero at target nonunits."
                ),
                input={
                    "character": {"group": _GROUP_MOD3, "coordinates": [1]},
                    "target_modulus": 15,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.restrict_modulus.compute",
        title="Factor a Dirichlet character through a divisor modulus",
        description=(
            "Descend a character from source modulus m to divisor d exactly "
            "when it is constant on every unit-reduction fiber. Success returns "
            "the target character and one canonical source-unit lift per target "
            "unit. Failure returns two source units in one fiber with distinct "
            "exact character values as an obstruction."
        ),
        request_type=DirichletCharacterRestrictionRequest,
        result_type=DirichletCharacterRestrictionResult,
        run=_compute_character_restriction,
        tags=("number-theory", "dirichlet-character", "restriction", "exact"),
        discovery_terms=(
            "restrict a Dirichlet character to a divisor modulus",
            "descend a character through unit-group reduction",
            "test whether a character factors through reduction",
        ),
        examples=(
            OperationExample(
                name="mod8_character_descends_to_mod4",
                description=(
                    "The character modulo 8 that is trivial on the reduction "
                    "kernel descends to modulus 4."
                ),
                input={
                    "character": {
                        "group": {
                            "modulus": 8,
                            "unit_residues": [1, 3, 5, 7],
                            "character_count": 4,
                            "invariant_factors": [2, 2],
                            "generators": [7, 5],
                            "generator_orders": [2, 2],
                            "unit_coordinates": [[0, 0], [1, 1], [0, 1], [1, 0]],
                            "exponent": 2,
                        },
                        "coordinates": [1, 0],
                    },
                    "target_modulus": 4,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.primitive_gauss_norm.compute",
        title="Compute the exact squared norm of a primitive character Gauss sum",
        description=(
            "Derive the character conductor from its unit-group data, require it "
            "to equal the source modulus, then compute tau(chi) times its exact "
            "cyclotomic conjugate. The returned cyclotomic value is |tau(chi)|^2 "
            "and equals the modulus. Caller-supplied primitive labels are not used."
        ),
        request_type=DirichletCharacterPrimitiveGaussNormRequest,
        result_type=native.DirichletCharacterPrimitiveGaussNormResult,
        run=_compute_primitive_gauss_norm,
        tags=("number-theory", "dirichlet-character", "gauss-sum", "exact"),
        examples=(
            OperationExample(
                name="quadratic_character_mod5_norm",
                description="Compute the exact squared norm, 5, of its primitive Gauss sum.",
                input={"character": {"group": _GROUP_MOD5, "coordinates": [2]}},
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.gauss_sum.compute",
        title="Compute an exact Dirichlet character Gauss sum",
        description=(
            "Return tau(chi) = sum over residues a modulo N of "
            "chi(a) exp(2*pi*i*a/N), using extension-by-zero character values. "
            "The result is source-bound in QQ[zeta_lcm(ord(chi), N)]; no "
            "primitivity assumption is made. Field order, work, and coefficient "
            "growth are admitted before cyclotomic construction and summation."
        ),
        request_type=DirichletCharacterGaussSumRequest,
        result_type=native.DirichletCharacterGaussSumResult,
        run=_compute_gauss_sum,
        tags=("number-theory", "dirichlet-character", "gauss-sum", "exact"),
        examples=(
            OperationExample(
                name="principal_gauss_sum_mod_3",
                description="Compute the principal character Gauss sum modulo 3, equal to -1.",
                input={"character": {"group": _GROUP_MOD3, "coordinates": [0]}},
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.generalized_gauss_sum.compute",
        title="Compute an exact generalized Dirichlet character Gauss sum",
        description=(
            "Return tau_n(chi) = sum over residues a modulo N of "
            "chi(a) exp(2*pi*i*n*a/N) at the supplied signed integer frequency n. "
            "Frequency is reduced modulo N; nonunit and zero frequencies are "
            "included. The result retains the source character, authored frequency, "
            "canonical residue, and exact value in QQ[zeta_lcm(ord(chi), N)]. "
            "Frequency digits, field order, work, coefficient growth, and output "
            "are bounded before the finite sum is expanded."
        ),
        request_type=DirichletCharacterGeneralizedGaussSumRequest,
        result_type=native.DirichletCharacterGeneralizedGaussSumResult,
        run=_compute_generalized_gauss_sum,
        tags=("number-theory", "dirichlet-character", "gauss-sum", "exact"),
        examples=(
            OperationExample(
                name="quadratic_character_mod5_zero_frequency",
                description=(
                    "The zero frequency sums all character values and is exactly zero."
                ),
                input={
                    "character": {"group": _GROUP_MOD5, "coordinates": [2]},
                    "frequency": "0",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.jacobi_sum.compute",
        title="Compute an exact Dirichlet character Jacobi sum",
        description=(
            "Return J(chi, psi) = sum over residues a modulo N of "
            "chi(a) psi(1-a), using extension-by-zero character values and "
            "the canonical rational cyclotomic field QQ[zeta_e] of the shared "
            "character-group exponent e. Both characters must carry the same "
            "complete group parent; field order and exact coefficient growth "
            "are bounded before summation."
        ),
        request_type=DirichletCharacterJacobiSumRequest,
        result_type=native.DirichletCharacterJacobiSumResult,
        run=_compute_jacobi_sum,
        tags=("number-theory", "dirichlet-character", "jacobi-sum", "exact"),
        examples=(
            OperationExample(
                name="quadratic_jacobi_sum_mod_5",
                description=(
                    "Compute J(chi, chi) for the quadratic character modulo 5; "
                    "the result is the exact rational cyclotomic element -1."
                ),
                input={
                    "left": {
                        "group": {
                            "modulus": 5,
                            "unit_residues": [1, 2, 3, 4],
                            "character_count": 4,
                            "invariant_factors": [4],
                            "generators": [2],
                            "generator_orders": [4],
                            "unit_coordinates": [[0], [1], [3], [2]],
                            "exponent": 4,
                        },
                        "coordinates": [2],
                    },
                    "right": {
                        "group": {
                            "modulus": 5,
                            "unit_residues": [1, 2, 3, 4],
                            "character_count": 4,
                            "invariant_factors": [4],
                            "generators": [2],
                            "generator_orders": [4],
                            "unit_coordinates": [[0], [1], [3], [2]],
                            "exponent": 4,
                        },
                        "coordinates": [2],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.orthogonality.compute",
        title="Compute an exact Dirichlet-character orthogonality sum",
        description=(
            "Return the exact integer sum over residues a modulo N of "
            "chi(a) conjugate(psi(a)), with nonunit values zero. The two "
            "characters must use the identical complete group parent. The "
            "result is bound to both source characters and is phi(N) for equal "
            "characters or zero for distinct characters."
        ),
        request_type=DirichletCharacterOrthogonalityRequest,
        result_type=native.DirichletCharacterOrthogonalityResult,
        run=_compute_orthogonality,
        tags=("number-theory", "dirichlet-character", "orthogonality", "exact"),
        examples=(
            OperationExample(
                name="quadratic_character_pairing_mod_3",
                description=(
                    "Compute the exact self-pairing of the nonprincipal "
                    "character modulo 3; the sum equals phi(3)=2."
                ),
                input={
                    "left": {"group": _GROUP_MOD3, "coordinates": [1]},
                    "right": {"group": _GROUP_MOD3, "coordinates": [1]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.generalized_bernoulli.compute",
        title="Compute an exact generalized Bernoulli number",
        description=(
            "Return B(k,chi)=N^(k-1) sum over a=1,...,N of "
            "chi(a) B_k(a/N) using B_1(x)=x-1/2. The result is a canonical "
            "QQ or cyclotomic value bound to the source character and index. "
            "The index, target field, rational coefficient growth, work, and "
            "result bytes are admitted before the finite sum is expanded."
        ),
        request_type=DirichletCharacterGeneralizedBernoulliRequest,
        result_type=native.DirichletCharacterGeneralizedBernoulliResult,
        run=_compute_generalized_bernoulli,
        tags=("number-theory", "dirichlet-character", "bernoulli", "exact"),
        examples=(
            OperationExample(
                name="principal_generalized_bernoulli_zero_mod_3",
                description=(
                    "Compute B(0,chi_0)=phi(3)/3=2/3 for the principal "
                    "character modulo 3."
                ),
                input={
                    "character": {"group": _GROUP_MOD3, "coordinates": [0]},
                    "index": 0,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.generalized_bernoulli_prefix.compute",
        title="Compute a generalized Bernoulli prefix",
        description=(
            "Return the exact values B(0,chi),...,B(k,chi) in the source "
            "character value field, using B_j(x) from t*exp(x*t)/(exp(t)-1). "
            "The full prefix work, output bytes, and coefficient growth are "
            "admitted before any value is evaluated."
        ),
        request_type=DirichletCharacterGeneralizedBernoulliPrefixRequest,
        result_type=DirichletCharacterGeneralizedBernoulliPrefix,
        run=_compute_generalized_bernoulli_prefix,
        tags=("number-theory", "dirichlet-character", "bernoulli", "exact"),
        examples=(
            OperationExample(
                name="principal_generalized_bernoulli_prefix_mod_3",
                description="Return B(0), B(1), and B(2) for the principal character modulo 3.",
                input={
                    "character": {"group": _GROUP_MOD3, "coordinates": [0]},
                    "maximum_index": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.l_value_nonpositive_integer.compute",
        title="Compute an exact Dirichlet-character L-value at a nonpositive integer",
        description=(
            "Return L(1-k, chi)=-B(k, chi)/k for 1 <= k <= 32 in the exact "
            "cyclotomic value field of chi. The denominator growth from division "
            "by k is admitted before the finite Bernoulli sum is expanded. This "
            "operation does not numerically evaluate the analytic L-function."
        ),
        request_type=DirichletCharacterLValueNonpositiveRequest,
        result_type=native.DirichletCharacterLValueNonpositiveResult,
        run=_compute_l_value_nonpositive,
        tags=("number-theory", "dirichlet-character", "l-value", "exact"),
        examples=(
            OperationExample(
                name="quadratic_mod3_at_zero",
                description="Compute L(0,chi)=-B(1,chi) for the nonprincipal character modulo 3.",
                input={
                    "character": {"group": _GROUP_MOD3, "coordinates": [1]},
                    "bernoulli_index": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.dirichlet_character_twist.compute",
        title="Twist a finite exact sequence by a Dirichlet character",
        description=(
            "Return b_(n)=chi(n)*a_(n) in a canonical rational cyclotomic "
            "sequence. Integer and rational sources supply index_origin; an "
            "existing cyclotomic source retains its authored origin. Repeated "
            "twists embed into the least common cyclotomic field. Field order, "
            "lookup work, coefficient growth, and output size are admitted "
            "before sequence expansion."
        ),
        request_type=DirichletCharacterSequenceTwistRequest,
        result_type=FiniteCyclotomicSequence,
        run=_compute_sequence_twist,
        tags=("sequence", "dirichlet-character", "cyclotomic", "exact"),
        discovery_terms=("Dirichlet character sequence twist",),
        examples=(
            OperationExample(
                name="quadratic_mod3_sequence_twist",
                description="Twist indices 1, 2, 3 by the quadratic character modulo 3.",
                input={
                    "sequence": {
                        "domain": "rational",
                        "values": [
                            {"num": "2", "den": "1"},
                            {"num": "3", "den": "1"},
                            {"num": "4", "den": "1"},
                        ],
                    },
                    "character": {
                        "group": _GROUP_MOD3,
                        "coordinates": [1],
                    },
                    "index_origin": 1,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
