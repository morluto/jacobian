"""Public declarations for exact bounded modular-form operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.modular_forms import operations as native
from jacobian.math.number_theory.modular_forms._models import (
    LevelOneNamedQExpansionRequest,
    ModularFormBasisFrameRequest,
    ModularFormBasisRequest,
    ModularFormCanonicalToFramedRequest,
    ModularFormCoordinatesAtkinLehnerRequest,
    ModularFormCoordinatesHeckeRequest,
    ModularFormCoordinatesProductRequest,
    ModularFormCoordinatesQExpansionRequest,
    ModularFormCoordinatesTransportRequest,
    ModularFormCoordinatesU2Request,
    ModularFormCoordinatesUPrimeRequest,
    ModularFormCoordinatesV2Request,
    ModularFormCoordinatesV3Request,
    ModularFormCoordinatesVDegeneracyRequest,
    ModularFormEqualityRequest,
    ModularFormEqualityResult,
    ModularFormFramedHeckeMatrixRequest,
    ModularFormFramedToCanonicalRequest,
    ModularFormHeckeMatrixRequest,
    ModularFormOperatorImagePrefixRequest,
    ModularFormOperatorImageRequest,
    SpaceDimensionRequest,
    SpaceDimensionResult,
)
from jacobian.math.number_theory.modular_forms.basis import (
    modular_form_basis_frame,
    modular_form_basis_q_expansions,
    modular_form_coordinates_atkin_lehner,
    modular_form_coordinates_equal,
    modular_form_coordinates_from_frame,
    modular_form_coordinates_hecke,
    modular_form_coordinates_product,
    modular_form_coordinates_q_expansion,
    modular_form_coordinates_to_frame,
    modular_form_coordinates_transport,
    modular_form_coordinates_u2,
    modular_form_coordinates_u_prime,
    modular_form_coordinates_v2,
    modular_form_coordinates_v3,
    modular_form_coordinates_v_degeneracy,
    modular_form_hecke_matrix,
    modular_form_hecke_matrix_in_frame,
    modular_form_operator_image,
    modular_form_operator_image_q_expansion,
)
from jacobian.math.number_theory.modular_forms.character_basis_tools import (
    TOOLS as CHARACTER_BASIS_TOOLS,
)
from jacobian.math.number_theory.modular_forms.field_coordinates_tools import (
    TOOLS as FIELD_COORDINATE_TOOLS,
)
from jacobian.math.number_theory.modular_forms.transform_tools import (
    TOOLS as TRANSFORM_TOOLS,
)
from jacobian.math.number_theory.modular_forms.values import (
    LevelOneModularQExpansion,
    ModularFormBasis,
    ModularFormChangeOfBasisFrame,
    ModularFormCoordinates,
    ModularFormFramedCoordinates,
    ModularFormFramedHeckeMatrix,
    ModularFormHeckeMatrix,
    ModularFormOperatorImage,
    ModularFormOperatorImagePrefix,
    ModularQExpansion,
)


def compute_level_one_named_q_expansion(
    request: LevelOneNamedQExpansionRequest,
) -> LevelOneModularQExpansion:
    return native.level_one_named_q_expansion(request.form, request.truncation_order)


def compute_space_dimension(request: SpaceDimensionRequest) -> SpaceDimensionResult:
    return native.space_dimension(request.space.as_value())


def compute_modular_form_basis(request: ModularFormBasisRequest) -> ModularFormBasis:
    return modular_form_basis_q_expansions(request.space, request.precision)


def compute_coordinate_q_expansion(
    request: ModularFormCoordinatesQExpansionRequest,
) -> ModularQExpansion:
    return modular_form_coordinates_q_expansion(request.form, request.precision)


def multiply_coordinate_forms(
    request: ModularFormCoordinatesProductRequest,
) -> ModularFormCoordinates:
    return modular_form_coordinates_product(request.left, request.right)


def transport_coordinates(
    request: ModularFormCoordinatesTransportRequest,
) -> ModularFormCoordinates:
    return modular_form_coordinates_transport(request.form, request.target_space)


def decide_coordinate_equality(
    request: ModularFormEqualityRequest,
) -> ModularFormEqualityResult:
    return ModularFormEqualityResult(
        equal=modular_form_coordinates_equal(request.left, request.right)
    )


def compute_basis_frame(
    request: ModularFormBasisFrameRequest,
) -> ModularFormChangeOfBasisFrame:
    return modular_form_basis_frame(request)


def convert_canonical_to_framed(
    request: ModularFormCanonicalToFramedRequest,
) -> ModularFormFramedCoordinates:
    return modular_form_coordinates_to_frame(request.frame, request.form)


def convert_framed_to_canonical(
    request: ModularFormFramedToCanonicalRequest,
) -> ModularFormCoordinates:
    return modular_form_coordinates_from_frame(request.form)


def apply_coordinate_hecke(
    request: ModularFormCoordinatesHeckeRequest,
) -> ModularFormCoordinates:
    return modular_form_coordinates_hecke(request.form, request.index)


def apply_coordinate_atkin_lehner(
    request: ModularFormCoordinatesAtkinLehnerRequest,
) -> ModularFormCoordinates:
    return modular_form_coordinates_atkin_lehner(request.form, request.divisor)


def compute_hecke_matrix(
    request: ModularFormHeckeMatrixRequest,
) -> ModularFormHeckeMatrix:
    return modular_form_hecke_matrix(request.space, request.index)


def compute_framed_hecke_matrix(
    request: ModularFormFramedHeckeMatrixRequest,
) -> ModularFormFramedHeckeMatrix:
    return modular_form_hecke_matrix_in_frame(request.frame, request.index)


def apply_coordinate_u2(
    request: ModularFormCoordinatesU2Request,
) -> ModularFormCoordinates:
    return modular_form_coordinates_u2(request.form)


def apply_coordinate_u_prime(
    request: ModularFormCoordinatesUPrimeRequest,
) -> ModularFormCoordinates:
    return modular_form_coordinates_u_prime(request.form, request.prime)


def apply_coordinate_v2(
    request: ModularFormCoordinatesV2Request,
) -> ModularFormCoordinates:
    return modular_form_coordinates_v2(request.form)


def apply_coordinate_v3(
    request: ModularFormCoordinatesV3Request,
) -> ModularFormCoordinates:
    return modular_form_coordinates_v3(request.form)


def apply_coordinate_v_degeneracy(
    request: ModularFormCoordinatesVDegeneracyRequest,
) -> ModularFormCoordinates:
    return modular_form_coordinates_v_degeneracy(request.form, request.d)


def compute_operator_image(
    request: ModularFormOperatorImageRequest,
) -> ModularFormOperatorImage:
    return modular_form_operator_image(
        request.source_form, request.operator, request.prime
    )


def compute_operator_image_prefix(
    request: ModularFormOperatorImagePrefixRequest,
) -> ModularFormOperatorImagePrefix:
    return modular_form_operator_image_q_expansion(request.image, request.precision)


TOOLS: MathTools = (
    MathTool(
        operation_id="modular_form.equal.check",
        title="Check global equality of modular forms",
        description=(
            "Decide exact equality of two globally represented forms. In one exact "
            "space, compare admitted canonical coordinates; across supported "
            "rational trivial-character spaces of equal weight, compare through "
            "the Sturm bound of their common Gamma0(lcm(levels)) ambient M space. "
            "The represented cyclotomic character space also uses its exact Sturm "
            "prefix; cyclotomic comparisons require the identical space and basis. "
            "Finite q-prefixes are not accepted as forms."
        ),
        request_type=ModularFormEqualityRequest,
        result_type=ModularFormEqualityResult,
        run=decide_coordinate_equality,
        tags=("modular-forms", "equality", "coordinates", "exact"),
        examples=(
            OperationExample(
                name="equal_weight_four_forms",
                description="The represented forms 2 A2^2 + 3 E4 are equal.",
                input={
                    "left": {
                        "space": {"level": 2, "weight": 4, "kind": "M"},
                        "basis_id": "gamma0-two-weight-2-4-monomials-v1",
                        "coordinates": [
                            {"num": "2", "den": "1"},
                            {"num": "3", "den": "1"},
                        ],
                    },
                    "right": {
                        "space": {"level": 2, "weight": 4, "kind": "M"},
                        "basis_id": "gamma0-two-weight-2-4-monomials-v1",
                        "coordinates": [
                            {"num": "2", "den": "1"},
                            {"num": "3", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.coordinates.transport.compute",
        title="Transport modular-form coordinates into a nested Gamma0 space",
        description=(
            "Express an exact QQ, trivial-character form in a same-weight target "
            "Gamma0 space when the source level divides the target level. M maps "
            "to M; S maps to S or M. The operation solves against the target "
            "canonical basis through its exact Sturm precision and returns "
            "target-bound coordinates. It admits both bases, combined work, "
            "coefficient growth, and output before backend materialization."
        ),
        request_type=ModularFormCoordinatesTransportRequest,
        result_type=ModularFormCoordinates,
        run=transport_coordinates,
        tags=("modular-forms", "coordinates", "transport", "exact"),
        examples=(
            OperationExample(
                name="level_one_e4_into_gamma0_two",
                description="Transport E4 coordinates from level one into M4(Gamma0(2)).",
                input={
                    "form": {
                        "space": {"level": 1, "weight": 4, "kind": "M"},
                        "basis_id": "level-one-e4-e6-monomials-v1",
                        "coordinates": [{"num": "1", "den": "1"}],
                    },
                    "target_space": {"level": 2, "weight": 4, "kind": "M"},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.level_one.named_q_expansion.compute",
        title="Compute an exact named level-one modular-form q-expansion",
        description=(
            "Construct the normalized exact q-prefix of E4, E6, or Ramanujan "
            "Delta in QQ[[q]]. The closed form family and requested finite "
            "precision are admitted before complete divisor scans and Delta's "
            "finite-series identity are evaluated."
        ),
        request_type=LevelOneNamedQExpansionRequest,
        result_type=LevelOneModularQExpansion,
        run=compute_level_one_named_q_expansion,
        tags=(
            "modular-forms",
            "q-expansion",
            "level-one",
            "eisenstein-series",
            "ramanujan-delta",
            "exact",
        ),
        examples=(
            OperationExample(
                name="delta_through_q5",
                description="Compute Delta through q^5; the form must be one of the closed normalized level-one family.",
                input={"form": "DELTA", "truncation_order": 6},
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.space.dimension.compute",
        title="Compute the exact dimension of a modular-form space",
        description=(
            "Return the exact complex dimension of a supported holomorphic "
            "M_k or cuspidal S_k space on Gamma0(N) with trivial character "
            "over QQ, or M_1/M_3(Gamma0(4), chi_{-4}), with the exact Gamma0 "
            "index, genus, cusp count, elliptic "
            "point counts, and Eisenstein/cusp projections. Levels through "
            "the bounded exact arithmetic envelope are admitted. No "
            "q-expansion coefficients are returned."
        ),
        request_type=SpaceDimensionRequest,
        result_type=SpaceDimensionResult,
        run=compute_space_dimension,
        tags=(
            "modular-forms",
            "dimension",
            "gamma0",
            "eisenstein-series",
            "cusp-forms",
            "exact",
        ),
        discovery_terms=(
            "dimension of M_k",
            "dimension of S_k",
            "level-one modular form dimension",
        ),
        examples=(
            OperationExample(
                name="dimension_of_m4_gamma0_10",
                description="Compute dim M_4(Gamma0(10)) = 7; the space must have trivial character over QQ.",
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 10,
                        "weight": 4,
                        "kind": "M",
                        "character": "TRIVIAL",
                        "coefficient_domain": "QQ",
                    }
                },
            ),
            OperationExample(
                name="dimension_of_m1_gamma0_4_chi_minus4",
                description="Compute dim M_1(Gamma0(4), chi_{-4}) = 1 using its exact modulus-4 character.",
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 4,
                        "weight": 1,
                        "kind": "M",
                        "character": {
                            "group": {
                                "modulus": 4,
                                "unit_residues": [1, 3],
                                "character_count": 2,
                                "invariant_factors": [2],
                                "generators": [3],
                                "generator_orders": [2],
                                "unit_coordinates": [[0], [1]],
                                "exponent": 2,
                            },
                            "coordinates": [1],
                        },
                        "coefficient_domain": "QQ",
                    }
                },
            ),
            OperationExample(
                name="dimension_of_m3_gamma0_4_chi_minus4",
                description="Compute dim M_3(Gamma0(4), chi_{-4}) = 2.",
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 4,
                        "weight": 3,
                        "kind": "M",
                        "character": {
                            "group": {
                                "modulus": 4,
                                "unit_residues": [1, 3],
                                "character_count": 2,
                                "invariant_factors": [2],
                                "generators": [3],
                                "generator_orders": [2],
                                "unit_coordinates": [[0], [1]],
                                "exponent": 2,
                            },
                            "coordinates": [1],
                        },
                        "coefficient_domain": "QQ",
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.space.basis_q_expansions.compute",
        title="Compute a deterministic modular-form basis q-prefix",
        description=(
            "Return every basis vector through the requested precision for "
            "rational spaces using the canonical E4/E6 and "
            "Delta bases at level one, A2/E4 monomials for M_k at Gamma0(2), "
            "reduced A2/B4star/S6 monomials for M_k at Gamma0(3), or B4/D4 "
            "monomials for even-weight M_k at Gamma0(4). Admission bounds "
            "weight, dimension, coefficient growth, work, and aggregate output "
            "before q-series expansion; the Gamma0(3) slice admits weights "
            "through 94 and prefixes through 63 terms. "
            "The nontrivial-character bases are M_1(Gamma0(4), chi_{-4}) "
            "and the two-dimensional M_3(Gamma0(4), chi_{-4}) space. For "
            "rational trivial-character M_k and S_k spaces at levels above "
            "four, a bounded PARI backend returns Jacobian's canonical "
            "q-expansion row-reduced frame through at least the Sturm bound. "
            "Its admission bounds level, weight, dimension, Sturm precision, "
            "estimated backend work, coefficient digits, and worst-case output."
        ),
        request_type=ModularFormBasisRequest,
        result_type=ModularFormBasis,
        run=compute_modular_form_basis,
        tags=("modular-forms", "basis", "q-expansion", "exact"),
        examples=(
            OperationExample(
                name="gamma0_five_m4_basis",
                description=(
                    "Return the exact q-Sturm frame of M_4(Gamma0(5)) "
                    "through its Sturm-determining three coefficients."
                ),
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 5,
                        "weight": 4,
                        "kind": "M",
                        "character": "TRIVIAL",
                        "coefficient_domain": "QQ",
                    },
                    "precision": 3,
                },
            ),
            OperationExample(
                name="basis_for_m12",
                description="Return E4^3 and E6^2 through q^3.",
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 1,
                        "weight": 12,
                        "kind": "M",
                        "character": "TRIVIAL",
                        "coefficient_domain": "QQ",
                    },
                    "precision": 4,
                },
            ),
            OperationExample(
                name="gamma0_two_m4_basis",
                description="Return A2^2 and E4 through q^2 in M_4(Gamma0(2)).",
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 2,
                        "weight": 4,
                        "kind": "M",
                        "character": "TRIVIAL",
                        "coefficient_domain": "QQ",
                    },
                    "precision": 3,
                },
            ),
            OperationExample(
                name="gamma0_three_m8_basis",
                description="Return the three exact reduced monomials A2^4, A2^2*B4star, and A2*S6 in M_8(Gamma0(3)).",
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 3,
                        "weight": 8,
                        "kind": "M",
                        "character": "TRIVIAL",
                        "coefficient_domain": "QQ",
                    },
                    "precision": 3,
                },
            ),
            OperationExample(
                name="gamma0_four_m4_basis",
                description="Return B4^2, B4*D4, and D4^2 through q^2 in M_4(Gamma0(4)).",
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 4,
                        "weight": 4,
                        "kind": "M",
                        "character": "TRIVIAL",
                        "coefficient_domain": "QQ",
                    },
                    "precision": 3,
                },
            ),
            OperationExample(
                name="gamma0_four_chi4_m1_basis",
                description="Return the one-element exact basis of M_1(Gamma0(4), chi_{-4}) through q^5.",
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 4,
                        "weight": 1,
                        "kind": "M",
                        "character": {
                            "group": {
                                "modulus": 4,
                                "unit_residues": [1, 3],
                                "character_count": 2,
                                "invariant_factors": [2],
                                "generators": [3],
                                "generator_orders": [2],
                                "unit_coordinates": [[0], [1]],
                                "exponent": 2,
                            },
                            "coordinates": [1],
                        },
                        "coefficient_domain": "QQ",
                    },
                    "precision": 6,
                },
            ),
            OperationExample(
                name="gamma0_four_chi4_m3_basis",
                description="Return a rational Eisenstein basis of M_3(Gamma0(4), chi_{-4}) through q^5.",
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 4,
                        "weight": 3,
                        "kind": "M",
                        "character": {
                            "group": {
                                "modulus": 4,
                                "unit_residues": [1, 3],
                                "character_count": 2,
                                "invariant_factors": [2],
                                "generators": [3],
                                "generator_orders": [2],
                                "unit_coordinates": [[0], [1]],
                                "exponent": 2,
                            },
                            "coordinates": [1],
                        },
                        "coefficient_domain": "QQ",
                    },
                    "precision": 6,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.coordinates.product.compute",
        title="Multiply exact modular-form coordinates",
        description=(
            "Multiply two trivial-character rational forms in supported "
            "native bases. The result is reconstructed through the target "
            "Sturm bound in Gamma0(lcm(levels)), with added weight and cusp "
            "space kind when either factor is cuspidal."
        ),
        request_type=ModularFormCoordinatesProductRequest,
        result_type=ModularFormCoordinates,
        run=multiply_coordinate_forms,
        tags=("modular-forms", "coordinates", "product", "exact"),
        examples=(
            OperationExample(
                name="e4_squared",
                description="Multiply E4 by E4 in the exact level-one weight-eight space.",
                input={
                    "left": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 1,
                            "weight": 4,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "level-one-e4-e6-monomials-v1",
                        "coordinates": [{"num": "1", "den": "1"}],
                    },
                    "right": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 1,
                            "weight": 4,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "level-one-e4-e6-monomials-v1",
                        "coordinates": [{"num": "1", "den": "1"}],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.coordinates.q_expansion.compute",
        title="Expand an exact form from modular-basis coordinates",
        description=(
            "Return the exact finite q-prefix of a rational form from its "
            "coordinates in a supported deterministic basis. Level one uses "
            "E4/E6 or Delta times that basis; Gamma0(2) supports holomorphic "
            "M_k using A2/E4 monomials; Gamma0(4) supports even-weight M_k "
            "using B4/D4 monomials. The vector remains bound to its space "
            "and basis convention."
        ),
        request_type=ModularFormCoordinatesQExpansionRequest,
        result_type=ModularQExpansion,
        run=compute_coordinate_q_expansion,
        tags=("modular-forms", "coordinates", "q-expansion", "exact"),
        examples=(
            OperationExample(
                name="expand_e4_cubed_coordinates",
                description="Expand E4^3 through q^3 from its exact basis coordinate.",
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 1,
                            "weight": 12,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "level-one-e4-e6-monomials-v1",
                        "coordinates": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "precision": 4,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.coordinates.hecke.apply",
        title="Apply a Hecke operator to exact modular-form coordinates",
        description=(
            "Apply T_n to a rational form in a supported canonical basis: "
            "level one for any bounded n; M_k(Gamma0(2)) for odd n; even-weight "
            "trivial-character M_k(Gamma0(4)) for odd n; M_k(Gamma0(3)) "
            "for n coprime to 3; or the reviewed M_1/M_3(Gamma0(4), "
            "chi_-4) spaces for odd n. The "
            "character-weighted coefficient formula is used for chi_-4. Exact "
            "output coordinates are reconstructed through the Sturm bound, "
            "after index, source precision, work, and growth admission."
        ),
        request_type=ModularFormCoordinatesHeckeRequest,
        result_type=ModularFormCoordinates,
        run=apply_coordinate_hecke,
        tags=("modular-forms", "hecke", "coordinates", "exact"),
        examples=(
            OperationExample(
                name="hecke_two_on_delta",
                description="T_2 acts on the weight-12 cusp form Delta by -24.",
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 1,
                            "weight": 12,
                            "kind": "S",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "level-one-e4-e6-monomials-v1",
                        "coordinates": [{"num": "1", "den": "1"}],
                    },
                    "index": 2,
                },
            ),
            OperationExample(
                name="hecke_three_gamma0_two_e4",
                description="T_3(E4) = 28 E4 in M_4(Gamma0(2)).",
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 2,
                            "weight": 4,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "gamma0-two-weight-2-4-monomials-v1",
                        "coordinates": [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                    "index": 3,
                },
            ),
            OperationExample(
                name="hecke_two_gamma0_three_weight_four",
                description="T_2 acts by 9 on the basis (A2^2, B4star) in M_4(Gamma0(3)).",
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 3,
                            "weight": 4,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "gamma0-three-weight-2-4-6-hypersurface-v1",
                        "coordinates": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "index": 2,
                },
            ),
            OperationExample(
                name="hecke_three_gamma0_four_b4",
                description="T_3(B4) = 4 B4 in M_2(Gamma0(4)).",
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 4,
                            "weight": 2,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "gamma0-four-weight-2-generators-v1",
                        "coordinates": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "index": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.coordinates.atkin_lehner.apply",
        title="Apply an Atkin-Lehner involution to modular-form coordinates",
        description=(
            "Apply the exact slash action |k W_Q to a globally represented even-weight "
            "QQ form with trivial character on Gamma0(N), where Q is an exact divisor "
            "of N. The output remains in the same space and basis. A bounded PARI "
            "worker transforms the canonical q-Sturm RREF representative; Jacobian "
            "reconstructs the output through the Sturm bound and admits work, exact "
            "coefficient digits, and output before backend work. Character-valued and "
            "odd-weight forms are outside this operation's current domain."
        ),
        request_type=ModularFormCoordinatesAtkinLehnerRequest,
        result_type=ModularFormCoordinates,
        run=apply_coordinate_atkin_lehner,
        tags=("modular-forms", "atkin-lehner", "coordinates", "exact"),
        examples=(
            OperationExample(
                name="fricke_on_e4_at_level_two",
                description="The normalized Fricke action sends E4(z) to 4 E4(2z).",
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 2,
                            "weight": 4,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "gamma0-two-weight-2-4-monomials-v1",
                        "coordinates": [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                    "divisor": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.hecke_matrix.compute",
        title="Compute a Hecke operator matrix in an exact modular-form basis",
        description=(
            "Return the exact T_n matrix on a represented modular-form space. "
            "entries[row][column] is the output coefficient at row_labels[row] "
            "for T_n applied to column_labels[column]. The same admitted basis "
            "expansion supplies all columns; each is reconstructed through the "
            "space's exact Sturm bound. Only indices coprime to the level are "
            "supported. In addition to the explicit low-level bases, rational "
            "trivial-character Gamma0 spaces use the bounded PARI basis path. "
            "Source order, work, coefficient growth, and matrix output are "
            "admitted before basis generation."
        ),
        request_type=ModularFormHeckeMatrixRequest,
        result_type=ModularFormHeckeMatrix,
        run=compute_hecke_matrix,
        tags=("modular-forms", "hecke", "matrix", "exact"),
        examples=(
            OperationExample(
                name="hecke_matrix_gamma0_two_weight_four_t3",
                description="T_3 matrix in the ordered basis (A2^2, E4) of M_4(Gamma0(2)).",
                input={
                    "space": {
                        "group": "GAMMA0",
                        "level": 2,
                        "weight": 4,
                        "kind": "M",
                        "character": "TRIVIAL",
                        "coefficient_domain": "QQ",
                    },
                    "index": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.hecke_matrix.in_frame.compute",
        title="Compute a Hecke matrix in a rational basis frame",
        description=(
            "Compute the exact canonical T_n matrix and conjugate it into the "
            "supplied invertible rational basis frame. The result retains the "
            "frame and uses entries[row][column] for the action on the framed "
            "basis. Canonical Hecke admission and frame conjugation each bound "
            "their work, coefficient growth, and serialized output."
        ),
        request_type=ModularFormFramedHeckeMatrixRequest,
        result_type=ModularFormFramedHeckeMatrix,
        run=compute_framed_hecke_matrix,
        tags=("modular-forms", "hecke", "basis-frame", "exact"),
        discovery_terms=(
            "Hecke operator matrix in a changed basis",
            "conjugate Hecke matrix into modular-form coordinates",
        ),
        examples=(
            OperationExample(
                name="hecke_matrix_level_one_weight_twelve_shifted_basis",
                description=(
                    "Conjugate T_2 from (E4^3, E6^2) into (c1=E4^3+E6^2, c2=E6^2)."
                ),
                input={
                    "frame": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 1,
                            "weight": 12,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "source_basis_id": "level-one-e4-e6-monomials-v1",
                        "source_labels": ["E4^3", "E6^2"],
                        "labels": ["c1", "c2"],
                        "entries": [
                            [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "1", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                        ],
                    },
                    "index": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.coordinates.u2.apply",
        title="Apply U_2 to exact supported modular-form coordinates",
        description=(
            "Apply U_2(sum a_n q^n) = sum a_(2n) q^n to an exact "
            "coordinate-defined form in M_k(Gamma0(2)), even-weight "
            "trivial-character M_k(Gamma0(4)), or in the exact "
            "M_1/M_3(Gamma0(4), chi_-4) bases. Reconstruct the result in "
            "the same deterministic basis through the exact Sturm bound; "
            "work and rational growth are admitted before expansion."
        ),
        request_type=ModularFormCoordinatesU2Request,
        result_type=ModularFormCoordinates,
        run=apply_coordinate_u2,
        tags=("modular-forms", "u-operator", "coordinates", "exact"),
        examples=(
            OperationExample(
                name="u2_gamma0_two_e4",
                description="U_2(E4) = -10 A2^2 + 11 E4 in M_4(Gamma0(2)).",
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 2,
                            "weight": 4,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "gamma0-two-weight-2-4-monomials-v1",
                        "coordinates": [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    }
                },
            ),
            OperationExample(
                name="u2.chi-minus4.weight-three-a",
                description="U_2(A3_chi_minus4) = A3_chi_minus4 in M_3(Gamma0(4), chi_-4).",
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 4,
                            "weight": 3,
                            "kind": "M",
                            "character": {
                                "group": {
                                    "modulus": 4,
                                    "unit_residues": [1, 3],
                                    "character_count": 2,
                                    "invariant_factors": [2],
                                    "generators": [3],
                                    "generator_orders": [2],
                                    "unit_coordinates": [[0], [1]],
                                    "exponent": 2,
                                },
                                "coordinates": [1],
                            },
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "gamma0-four-chi4-weight-three-v1",
                        "coordinates": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.u_operator.apply",
        title="Apply U_p to exact Gamma0 coordinates",
        description=(
            "Apply U_p(sum a_n q^n) = sum a_(pn) q^n to exact QQ "
            "trivial-character M or S coordinates on Gamma0(N), when prime "
            "p divides N. The result stays in the same represented space and "
            "is reconstructed through its Sturm bound after admission of both "
            "basis precisions, exact work, rational growth, and output size."
        ),
        request_type=ModularFormCoordinatesUPrimeRequest,
        result_type=ModularFormCoordinates,
        run=apply_coordinate_u_prime,
        tags=("modular-forms", "u-operator", "coordinates", "exact"),
        examples=(
            OperationExample(
                name="u11_gamma0_11_s2",
                description=(
                    "Apply U_11 to the normalized cusp form in "
                    "S_2(Gamma0(11)); the image remains in that space."
                ),
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 11,
                            "weight": 2,
                            "kind": "S",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "gamma0-rational-gamma0-sturm-rref-v1",
                        "coordinates": [{"num": "1", "den": "1"}],
                    },
                    "prime": 11,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.coordinates.v2.apply",
        title="Apply V_2 into exact higher-level coordinates",
        description=(
            "Apply V_2(sum a_n q^n) = sum a_n q^(2n) from exact level-one "
            "M_k coordinates into the canonical M_k(Gamma0(2)) basis, or "
            "from M_k(Gamma0(2)) into the canonical even-weight "
            "M_k(Gamma0(4)) basis. Reconstruct through the target Sturm "
            "bound; source precision, work, and growth are admitted first."
        ),
        request_type=ModularFormCoordinatesV2Request,
        result_type=ModularFormCoordinates,
        run=apply_coordinate_v2,
        tags=("modular-forms", "v-operator", "coordinates", "exact"),
        examples=(
            OperationExample(
                name="v2_a2_to_gamma0_four",
                description="V_2(A2) = B4 in M_2(Gamma0(4)).",
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 2,
                            "weight": 2,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "gamma0-two-weight-2-4-monomials-v1",
                        "coordinates": [{"num": "1", "den": "1"}],
                    }
                },
            ),
            OperationExample(
                name="v2_level_one_e4_squared",
                description="V_2(E4^2) is returned in M_8(Gamma0(2)) coordinates.",
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 1,
                            "weight": 8,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "level-one-e4-e6-monomials-v1",
                        "coordinates": [{"num": "1", "den": "1"}],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.coordinates.v3.apply",
        title="Apply V_3 into exact Gamma0(3) coordinates",
        description=(
            "Apply f(q) -> f(q^3) from exact level-one M_k coordinates into "
            "the canonical trivial-character M_k(Gamma0(3)) basis. Reconstruct "
            "through the target Sturm bound after admitting source precision, "
            "work, coefficient growth, and output size."
        ),
        request_type=ModularFormCoordinatesV3Request,
        result_type=ModularFormCoordinates,
        run=apply_coordinate_v3,
        tags=("modular-forms", "degeneracy-map", "coordinates", "exact"),
        examples=(
            OperationExample(
                name="v3_level_one_e4",
                description="Return V_3(E4) in exact M_4(Gamma0(3)) coordinates.",
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 1,
                            "weight": 4,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "level-one-e4-e6-monomials-v1",
                        "coordinates": [{"num": "1", "den": "1"}],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.coordinates.v_degeneracy.apply",
        title="Apply exact V_d degeneracy coordinates",
        description=(
            "Apply V_d(f)(q)=f(q^d) from a represented trivial-character QQ "
            "Gamma0(M) space into the exact same-kind Gamma0(Md) basis. Source "
            "and target basis, work, coefficient growth, and output are admitted "
            "before q-expansion materialization."
        ),
        request_type=ModularFormCoordinatesVDegeneracyRequest,
        result_type=ModularFormCoordinates,
        run=apply_coordinate_v_degeneracy,
        tags=("modular-forms", "degeneracy-map", "v-operator", "coordinates", "exact"),
        examples=(
            OperationExample(
                name="v2_gamma0_two_weight_four",
                description="Apply V_2 to E4 in M_4(Gamma0(2)) into M_4(Gamma0(4)).",
                input={
                    "form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 2,
                            "weight": 4,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "gamma0-two-weight-2-4-monomials-v1",
                        "coordinates": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "d": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.coordinates.operator_image.compute",
        title="Bind a level-one form to an exact U_p or V_p image",
        description=(
            "Return a typed exact operator image for U_p or V_p of a "
            "coordinate-defined level-one form. The value retains its exact "
            "source coordinates and has the matching trivial-character "
            "Gamma0(p) codomain. The operator index must be prime."
        ),
        request_type=ModularFormOperatorImageRequest,
        result_type=ModularFormOperatorImage,
        run=compute_operator_image,
        tags=("modular-forms", "u-operator", "v-operator", "exact"),
        examples=(
            OperationExample(
                name="u2_image_of_delta",
                description="Bind U_2(Delta) as an exact form on Gamma0(2).",
                input={
                    "source_form": {
                        "space": {
                            "group": "GAMMA0",
                            "level": 1,
                            "weight": 12,
                            "kind": "S",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "level-one-e4-e6-monomials-v1",
                        "coordinates": [{"num": "1", "den": "1"}],
                    },
                    "operator": "U",
                    "prime": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.coordinates.operator_image.q_expansion.compute",
        title="Compute an exact U_p or V_p image q-prefix",
        description=(
            "Evaluate a finite q-prefix from the operator image's exact "
            "level-one source coordinates. U_p returns a_(p n); V_p places "
            "a_n at q^(p n). Source precision, exact coefficient work, and "
            "output size are admitted before q-series expansion."
        ),
        request_type=ModularFormOperatorImagePrefixRequest,
        result_type=ModularFormOperatorImagePrefix,
        run=compute_operator_image_prefix,
        tags=("modular-forms", "u-operator", "v-operator", "q-expansion", "exact"),
        examples=(
            OperationExample(
                name="v2_delta_prefix",
                description="Compute V_2(Delta) through q^8 on Gamma0(2).",
                input={
                    "image": {
                        "source_form": {
                            "space": {
                                "group": "GAMMA0",
                                "level": 1,
                                "weight": 12,
                                "kind": "S",
                                "character": "TRIVIAL",
                                "coefficient_domain": "QQ",
                            },
                            "basis_id": "level-one-e4-e6-monomials-v1",
                            "coordinates": [{"num": "1", "den": "1"}],
                        },
                        "operator": "V",
                        "prime": 2,
                        "codomain": {
                            "group": "GAMMA0",
                            "level": 2,
                            "weight": 12,
                            "kind": "S",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                    },
                    "precision": 9,
                },
            ),
        ),
    ),
)

TOOLS = TOOLS + TRANSFORM_TOOLS

TOOLS = (
    *TOOLS,
    MathTool(
        operation_id="modular_form.basis_frame.create",
        title="Create a source-bound modular-form basis frame",
        description=(
            "Validate a square exact rational basis matrix, ordered unique labels, "
            "and its source space and canonical basis. The columns express the "
            "caller basis vectors in canonical coordinates. Creation proves "
            "invertibility, and conversions recheck it on their input value."
        ),
        request_type=ModularFormBasisFrameRequest,
        result_type=ModularFormChangeOfBasisFrame,
        run=compute_basis_frame,
        tags=("modular-forms", "basis", "change-of-basis", "exact"),
        examples=(
            OperationExample(
                name="gamma0_two_m4_shifted_basis",
                description="Declare c1=A2^2+E4 and c2=E4 in M4(Gamma0(2)).",
                input={
                    "space": {"level": 2, "weight": 4, "kind": "M"},
                    "source_basis_id": "gamma0-two-weight-2-4-monomials-v1",
                    "source_labels": ["A2^2", "E4"],
                    "labels": ["c1", "c2"],
                    "entries": [
                        [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                        [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.coordinates.to_frame.compute",
        title="Convert canonical modular-form coordinates to a framed basis",
        description=(
            "Solve the source-bound exact rational frame matrix for coordinates "
            "in its labeled basis. Requires the exact same modular space and "
            "canonical basis and rejects singular frames."
        ),
        request_type=ModularFormCanonicalToFramedRequest,
        result_type=ModularFormFramedCoordinates,
        run=convert_canonical_to_framed,
        tags=("modular-forms", "coordinates", "basis", "exact"),
        examples=(
            OperationExample(
                name="convert_m4_coordinates_to_shifted_basis",
                description="Convert 2*A2^2+3*E4 to 2*c1+c2.",
                input={
                    "frame": {
                        "space": {"level": 2, "weight": 4, "kind": "M"},
                        "source_basis_id": "gamma0-two-weight-2-4-monomials-v1",
                        "source_labels": ["A2^2", "E4"],
                        "labels": ["c1", "c2"],
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
                        ],
                    },
                    "form": {
                        "space": {"level": 2, "weight": 4, "kind": "M"},
                        "basis_id": "gamma0-two-weight-2-4-monomials-v1",
                        "coordinates": [
                            {"num": "2", "den": "1"},
                            {"num": "3", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.coordinates.from_frame.compute",
        title="Convert framed modular-form coordinates to the canonical basis",
        description=(
            "Multiply by the exact frame matrix and return the existing canonical "
            "ModularFormCoordinates value, after checking that the frame is invertible."
        ),
        request_type=ModularFormFramedToCanonicalRequest,
        result_type=ModularFormCoordinates,
        run=convert_framed_to_canonical,
        tags=("modular-forms", "coordinates", "basis", "exact"),
        examples=(
            OperationExample(
                name="restore_m4_coordinates_from_shifted_basis",
                description="Return the canonical coordinates of 2*c1+c2.",
                input={
                    "form": {
                        "frame": {
                            "space": {"level": 2, "weight": 4, "kind": "M"},
                            "source_basis_id": "gamma0-two-weight-2-4-monomials-v1",
                            "source_labels": ["A2^2", "E4"],
                            "labels": ["c1", "c2"],
                            "entries": [
                                [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                                [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
                            ],
                        },
                        "coordinates": [
                            {"num": "2", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
TOOLS = TOOLS + CHARACTER_BASIS_TOOLS
TOOLS = TOOLS + FIELD_COORDINATE_TOOLS
