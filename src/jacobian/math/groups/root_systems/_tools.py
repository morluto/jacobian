"""Root system operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.root_systems._dynkin_models import FiniteDynkinDiagram
from jacobian.math.groups.root_systems._highest_coroot_models import (
    HighestCorootsResult,
)
from jacobian.math.groups.root_systems._highest_coroot_operations import highest_coroots
from jacobian.math.groups.root_systems._models import (
    MAX_RANK,
    CartanDatumRequest,
    CartanMatrixRequest,
    CartanTypeRequest,
    CartanTypeResult,
    CorootLatticeVector,
    CorootToCoweightLatticeRequest,
    CoweightLatticeVector,
    FiniteCartanDatum,
    LatticeVectorCreateRequest,
    PositiveCorootsResult,
    PositiveRootProfileResult,
    RootLatticeVector,
    RootLengthProfileResult,
    RootPosetResult,
    RootSystemDataResult,
    RootToCorootRequest,
    RootToCorootResult,
    RootToWeightLatticeRequest,
    SimpleReflectionRequest,
    SimpleReflectionResult,
    SimpleReflectionsResult,
    WeightLatticeVector,
    WeylDescentsResult,
    WeylDimensionRequest,
    WeylDimensionResult,
    WeylElement,
    WeylElementComposeRequest,
    WeylElementInverseRequest,
    WeylElementLengthResult,
    WeylElementOrderResult,
    WeylElementRequest,
    WeylElementRootActionRequest,
    WeylElementWeightActionRequest,
    WeylExponentsResult,
    WeylGroupOrderResult,
    WeylLongestElementResult,
    WeylParabolicRequest,
    WeylParabolicResult,
    WeylPoincarePolynomialResult,
    WeylVectorActionRequest,
    WeylVectorActionResult,
    WeylWeightOrbitRequest,
    WeylWeightOrbitResult,
)
from jacobian.math.groups.root_systems.operations import (
    cartan_datum,
    cartan_matrix_from_type,
    coroot_lattice_vector,
    coroot_to_coweight_lattice,
    coweight_lattice_vector,
    coxeter_polynomial,
    dynkin_diagram,
    positive_coroots,
    positive_root_profile,
    root_lattice_vector,
    root_length_profile,
    root_poset,
    root_system_data,
    root_to_coroot,
    root_to_weight_lattice,
    simple_reflection,
    simple_reflections,
    weight_lattice_vector,
    weyl_element_compose,
    weyl_element_descents,
    weyl_element_from_word,
    weyl_element_inverse,
    weyl_element_length,
    weyl_element_order,
    weyl_exponents,
    weyl_group_order,
    weyl_longest_element,
    weyl_parabolic,
    weyl_poincare_polynomial,
    weyl_weight_orbit,
    weyl_word_act_on_root_vector,
)
from jacobian.math.groups.root_systems.root_actions import weyl_element_act_on_root
from jacobian.math.groups.root_systems.weight_actions import weyl_element_act_on_weight
from jacobian.math.groups.root_systems.weyl_dimension import weyl_dimension
from jacobian.math.polynomials._models import IntegerPolynomial


def _run_cartan_datum(request: CartanDatumRequest) -> FiniteCartanDatum:
    return cartan_datum(request.matrix)


def _run_root_lattice_vector(request: LatticeVectorCreateRequest) -> RootLatticeVector:
    return root_lattice_vector(request.matrix, request.coordinates)


def _run_coroot_lattice_vector(
    request: LatticeVectorCreateRequest,
) -> CorootLatticeVector:
    return coroot_lattice_vector(request.matrix, request.coordinates)


def _run_weight_lattice_vector(
    request: LatticeVectorCreateRequest,
) -> WeightLatticeVector:
    return weight_lattice_vector(request.matrix, request.coordinates)


def _run_coweight_lattice_vector(
    request: LatticeVectorCreateRequest,
) -> CoweightLatticeVector:
    return coweight_lattice_vector(request.matrix, request.coordinates)


def _run_root_to_weight_lattice(
    request: RootToWeightLatticeRequest,
) -> WeightLatticeVector:
    return root_to_weight_lattice(request.vector)


def _run_coroot_to_coweight_lattice(
    request: CorootToCoweightLatticeRequest,
) -> CoweightLatticeVector:
    return coroot_to_coweight_lattice(request.vector)


def _run_cartan_matrix_from_type(request: CartanTypeRequest) -> CartanTypeResult:
    return cartan_matrix_from_type(request.cartan_type, request.rank)


def _run_dynkin_diagram(request: CartanMatrixRequest) -> FiniteDynkinDiagram:
    return dynkin_diagram(request.matrix)


def _run_root_system_data(request: CartanMatrixRequest) -> RootSystemDataResult:
    return root_system_data(request.matrix)


def _run_coxeter_polynomial(request: CartanMatrixRequest) -> IntegerPolynomial:
    return coxeter_polynomial(request.matrix)


def _run_positive_coroots(request: CartanMatrixRequest) -> PositiveCorootsResult:
    return positive_coroots(request.matrix)


def _run_root_to_coroot(request: RootToCorootRequest) -> RootToCorootResult:
    return root_to_coroot(request.matrix, request.root_coefficients)


def _run_root_length_profile(request: CartanMatrixRequest) -> RootLengthProfileResult:
    return root_length_profile(request.matrix)


def _run_highest_coroots(request: CartanMatrixRequest) -> HighestCorootsResult:
    return highest_coroots(request.matrix)


def _run_positive_root_profile(
    request: CartanMatrixRequest,
) -> PositiveRootProfileResult:
    return positive_root_profile(request.matrix)


def _run_weyl_exponents(request: CartanMatrixRequest) -> WeylExponentsResult:
    return weyl_exponents(request.matrix)


def _run_weyl_poincare_polynomial(
    request: CartanMatrixRequest,
) -> WeylPoincarePolynomialResult:
    return weyl_poincare_polynomial(request.matrix)


def _run_root_poset(request: CartanMatrixRequest) -> RootPosetResult:
    return root_poset(request.matrix)


def _run_simple_reflections(request: CartanMatrixRequest) -> SimpleReflectionsResult:
    return simple_reflections(request.matrix)


def _run_simple_reflection(request: SimpleReflectionRequest) -> SimpleReflectionResult:
    return simple_reflection(request.matrix, request.vector, request.simple_index)


def _run_weyl_group_order(request: CartanMatrixRequest) -> WeylGroupOrderResult:
    return weyl_group_order(request.matrix)


def _run_weyl_parabolic(request: WeylParabolicRequest) -> WeylParabolicResult:
    return weyl_parabolic(request.matrix, request.simple_root_indices)


def _run_weyl_element_length(request: WeylElementRequest) -> WeylElementLengthResult:
    return weyl_element_length(request.matrix, request.word)


def _run_weyl_element_order(request: WeylElementRequest) -> WeylElementOrderResult:
    return weyl_element_order(request.matrix, request.word)


def _run_weyl_element_from_word(request: WeylElementRequest) -> WeylElement:
    return weyl_element_from_word(request.matrix, request.word)


def _run_weyl_element_compose(request: WeylElementComposeRequest) -> WeylElement:
    return weyl_element_compose(request.first, request.then)


def _run_weyl_element_inverse(request: WeylElementInverseRequest) -> WeylElement:
    return weyl_element_inverse(request.element)


def _run_weyl_element_weight_action(
    request: WeylElementWeightActionRequest,
) -> WeightLatticeVector:
    return weyl_element_act_on_weight(request)


def _run_weyl_element_root_action(
    request: WeylElementRootActionRequest,
) -> RootLatticeVector:
    return weyl_element_act_on_root(request)


def _run_weyl_element_descents(
    request: WeylElementRequest,
) -> WeylDescentsResult:
    return weyl_element_descents(request.matrix, request.word)


def _run_weyl_longest_element(
    request: CartanMatrixRequest,
) -> WeylLongestElementResult:
    return weyl_longest_element(request.matrix)


def _run_weyl_vector_action(
    request: WeylVectorActionRequest,
) -> WeylVectorActionResult:
    return weyl_word_act_on_root_vector(request.matrix, request.word, request.vector)


def _run_weyl_weight_orbit(
    request: WeylWeightOrbitRequest,
) -> WeylWeightOrbitResult:
    return weyl_weight_orbit(request.matrix, request.weight)


def _run_weyl_dimension(request: WeylDimensionRequest) -> WeylDimensionResult:
    return weyl_dimension(request.matrix, request.highest_weight)


_A2 = {
    "matrix": {
        "matrix": {
            "domain": "ZZ",
            "row_count": 2,
            "column_count": 2,
            "entries": [["2", "-1"], ["-1", "2"]],
        },
        "simple_root_axis": [0, 1],
    }
}
_A3 = {
    "matrix": {
        "matrix": {
            "domain": "ZZ",
            "row_count": 3,
            "column_count": 3,
            "entries": [["2", "-1", "0"], ["-1", "2", "-1"], ["0", "-1", "2"]],
        },
        "simple_root_axis": [0, 1, 2],
    }
}
_A2_LATTICE_DATUM = {
    "cartan_matrix": _A2["matrix"],
    "symmetrizer": [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
    "root_to_weight": {
        "domain": "ZZ",
        "row_count": 2,
        "column_count": 2,
        "entries": [["2", "-1"], ["-1", "2"]],
    },
    "coroot_to_coweight": {
        "domain": "ZZ",
        "row_count": 2,
        "column_count": 2,
        "entries": [["2", "-1"], ["-1", "2"]],
    },
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="root_system.coxeter_polynomial.compute",
        title="Compute the Coxeter polynomial of a finite Cartan datum",
        description=(
            "Return det(tI-c) for the ordered simple-reflection product "
            "c=s_(r-1)...s_1 s_0 acting on the simple-root lattice; factors "
            "are applied left to right. Returns a canonical integer polynomial."
        ),
        request_type=CartanMatrixRequest,
        result_type=IntegerPolynomial,
        run=_run_coxeter_polynomial,
        tags=("algebra", "root-system", "weyl-group", "coxeter", "exact"),
        discovery_terms=(
            "Coxeter polynomial",
            "characteristic polynomial Coxeter element",
        ),
        examples=(
            OperationExample(
                name="a2_coxeter_polynomial",
                description="Compute t^2+t+1 for the stated A2 reflection order.",
                input={"matrix": _A2["matrix"]},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.cartan_datum.compute",
        title="Construct finite Cartan root, coroot, and weight lattice data",
        description=(
            "Construct the exact finite Cartan datum for a canonical Cartan matrix, "
            "including a positive symmetrizer and explicit root-to-weight and "
            "coroot-to-coweight basis maps. Nonsymmetric B/C/G matrices retain "
            "their ordered basis transport."
        ),
        request_type=CartanDatumRequest,
        result_type=FiniteCartanDatum,
        run=_run_cartan_datum,
        tags=("algebra", "root-system", "cartan", "lattice", "exact"),
        discovery_terms=(
            "Cartan datum",
            "root coroot lattice",
            "weight basis transport",
        ),
        examples=(
            OperationExample(
                name="g2_cartan_datum",
                description=(
                    "Construct root/coroot/weight basis data for G2; the matrix "
                    "must be a finite-type Cartan matrix."
                ),
                input={
                    "matrix": {
                        "matrix": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["2", "-3"], ["-1", "2"]],
                        },
                        "simple_root_axis": [0, 1],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.root_lattice.vector.compute",
        title="Create a simple-root lattice vector",
        description=(
            "Return the exact integral coordinates in the simple-root basis, "
            "bound to a canonical finite Cartan datum. Coordinates are admitted "
            "before the datum value is constructed."
        ),
        request_type=LatticeVectorCreateRequest,
        result_type=RootLatticeVector,
        run=_run_root_lattice_vector,
        tags=("algebra", "root-system", "root-lattice", "exact"),
        discovery_terms=("root lattice vector", "simple-root coordinates"),
        examples=(
            OperationExample(
                name="a2_simple_root_vector",
                description="Create the first simple root in the A2 root lattice.",
                input={"matrix": _A2["matrix"], "coordinates": [1, 0]},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.coroot_lattice.vector.compute",
        title="Create a simple-coroot lattice vector",
        description=(
            "Return the exact integral coordinates in the simple-coroot basis, "
            "bound to a canonical finite Cartan datum."
        ),
        request_type=LatticeVectorCreateRequest,
        result_type=CorootLatticeVector,
        run=_run_coroot_lattice_vector,
        tags=("algebra", "root-system", "coroot-lattice", "exact"),
        discovery_terms=("coroot lattice vector", "simple-coroot coordinates"),
        examples=(
            OperationExample(
                name="a2_simple_coroot_vector",
                description="Create the first simple coroot in the A2 coroot lattice.",
                input={"matrix": _A2["matrix"], "coordinates": [1, 0]},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.weight_lattice.vector.compute",
        title="Create a fundamental-weight lattice vector",
        description=(
            "Return the exact integral coordinates in the fundamental-weight "
            "basis, bound to a canonical finite Cartan datum."
        ),
        request_type=LatticeVectorCreateRequest,
        result_type=WeightLatticeVector,
        run=_run_weight_lattice_vector,
        tags=("algebra", "root-system", "weight-lattice", "exact"),
        discovery_terms=("weight lattice vector", "fundamental-weight coordinates"),
        examples=(
            OperationExample(
                name="a2_fundamental_weight_vector",
                description="Create the first fundamental weight of A2.",
                input={"matrix": _A2["matrix"], "coordinates": [1, 0]},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.coweight_lattice.vector.compute",
        title="Create a fundamental-coweight lattice vector",
        description=(
            "Return the exact integral coordinates in the fundamental-coweight "
            "basis, bound to a canonical finite Cartan datum."
        ),
        request_type=LatticeVectorCreateRequest,
        result_type=CoweightLatticeVector,
        run=_run_coweight_lattice_vector,
        tags=("algebra", "root-system", "coweight-lattice", "exact"),
        discovery_terms=("coweight lattice vector", "fundamental-coweight coordinates"),
        examples=(
            OperationExample(
                name="a2_fundamental_coweight_vector",
                description="Create the first fundamental coweight of A2.",
                input={"matrix": _A2["matrix"], "coordinates": [1, 0]},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.root_to_weight_lattice.compute",
        title="Embed a root-lattice vector in the weight lattice",
        description=(
            "Apply the exact Q-to-P inclusion using A in the retained ordered "
            "simple-root and fundamental-weight bases. The source datum is "
            "recomputed and checked before the map is applied."
        ),
        request_type=RootToWeightLatticeRequest,
        result_type=WeightLatticeVector,
        run=_run_root_to_weight_lattice,
        tags=("algebra", "root-system", "root-lattice", "weight-lattice", "exact"),
        discovery_terms=("root to weight lattice embedding", "Q to P inclusion"),
        examples=(
            OperationExample(
                name="a2_root_to_weight_inclusion",
                description="Express the first A2 simple root in fundamental weights.",
                input={"vector": {"datum": _A2_LATTICE_DATUM, "coordinates": [1, 0]}},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.coroot_to_coweight_lattice.compute",
        title="Embed a coroot-lattice vector in the coweight lattice",
        description=(
            "Apply the exact Q^vee-to-P^vee inclusion using A transpose in the "
            "retained ordered simple-coroot and fundamental-coweight bases. "
            "The source datum is recomputed before the map is applied."
        ),
        request_type=CorootToCoweightLatticeRequest,
        result_type=CoweightLatticeVector,
        run=_run_coroot_to_coweight_lattice,
        tags=("algebra", "root-system", "coroot-lattice", "coweight-lattice", "exact"),
        discovery_terms=(
            "coroot to coweight lattice embedding",
            "Qvee to Pvee inclusion",
        ),
        examples=(
            OperationExample(
                name="a2_coroot_to_coweight_inclusion",
                description="Express the first A2 simple coroot in fundamental coweights.",
                input={"vector": {"datum": _A2_LATTICE_DATUM, "coordinates": [1, 0]}},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.cartan_matrix.from_type.compute",
        title="Build a Cartan matrix from a finite Dynkin type and rank",
        description="Build the exact Cartan matrix of a finite irreducible root "
        "system from its Dynkin type (A_n, B_n, C_n, D_n, E_6/E_7/E_8, F_4, "
        "G_2) and rank, using the ordered simple-root convention "
        "A[i,j]=<alpha_i^vee,alpha_j> and the standard B_n/C_n short/long "
        "end-node orientation. The kernel asserts its own output is finite-type; "
        "the returned canonical matrix feeds root-system and Weyl-group "
        "operations directly.",
        request_type=CartanTypeRequest,
        result_type=CartanTypeResult,
        run=_run_cartan_matrix_from_type,
        tags=("algebra", "root-system", "cartan-matrix", "exact"),
        discovery_terms=(
            "cartan matrix by Lie type",
            "dynkin type constructor",
            "finite root system Cartan matrix",
        ),
        examples=(
            OperationExample(
                name="a2_cartan_from_type",
                description="Build the A2 Cartan matrix [[2, -1], [-1, 2]]; "
                "the (type, rank) pair must be a valid finite Dynkin type.",
                input={"cartan_type": "A", "rank": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.dynkin_diagram.compute",
        title="Construct the exact labeled Dynkin diagram",
        description=(
            "Return all simple-root nodes and the exact Cartan-labeled edges of a "
            "finite root datum. Each edge stores the ordered pair (A[i,j], A[j,i]) "
            "for i<j, preserving multiple-bond direction as well as multiplicity. "
            "Isolated nodes are retained. Rank is at most 8 and the result has at "
            "most 28 edges."
        ),
        request_type=CartanMatrixRequest,
        result_type=FiniteDynkinDiagram,
        run=_run_dynkin_diagram,
        tags=("algebra", "root-system", "dynkin-diagram", "exact"),
        discovery_terms=("Dynkin diagram", "Cartan graph", "multiple bond direction"),
        examples=(
            OperationExample(
                name="g2_directed_multiple_edge",
                description=(
                    "Preserve the G2 Cartan pair (-3,-1) on its triple bond and "
                    "report edge multiplicity 3."
                ),
                input={
                    "matrix": {
                        "matrix": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["2", "-3"], ["-1", "2"]],
                        },
                        "simple_root_axis": [0, 1],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.positive_roots.compute",
        title="Compute positive roots from a Cartan matrix",
        description="Compute all positive roots of a finite crystallographic root "
        "system from its Cartan matrix, using closure under simple "
        "reflections. Returns simple and positive roots plus highest-root "
        "and Coxeter data for each irreducible component.",
        request_type=CartanMatrixRequest,
        result_type=RootSystemDataResult,
        run=_run_root_system_data,
        tags=("algebra", "root-system", "exact"),
        examples=(
            OperationExample(
                name="a2_cartan",
                description="Compute root system data for A2; "
                "the matrix must be a valid finite-type Cartan matrix.",
                input={"matrix": _A2["matrix"]},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.coroots.compute",
        title="Compute all positive coroots",
        description=(
            "Return every positive root with its exact coroot coordinates in the "
            "matching simple-coroot basis and its squared root length, bound to "
            "the canonical finite Cartan datum. Rank is at most 8 and the complete "
            "positive-root family is bounded by 120 entries."
        ),
        request_type=CartanMatrixRequest,
        result_type=PositiveCorootsResult,
        run=_run_positive_coroots,
        tags=("algebra", "root-system", "coroot", "exact"),
        discovery_terms=("positive coroots", "roots and coroots"),
        examples=(
            OperationExample(
                name="b2_positive_coroots",
                description=(
                    "Compute all positive coroots of B2 with their root lengths; "
                    "the matrix must be finite type and rank at most 8."
                ),
                input={
                    "matrix": {
                        "matrix": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["2", "-2"], ["-1", "2"]],
                        },
                        "simple_root_axis": [0, 1],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.highest_coroots.compute",
        title="Compute the highest positive coroot in every factor",
        description=(
            "Return one componentwise maximal positive coroot for every connected "
            "Cartan factor, with its comarks and the index of its root-coroot pair "
            "in the canonical positive-root order. Coroot dominance is coordinatewise "
            "in the simple-coroot basis. Rank is at most 8 and the positive-coroot "
            "table has at most 120 entries."
        ),
        request_type=CartanMatrixRequest,
        result_type=HighestCorootsResult,
        run=_run_highest_coroots,
        tags=("algebra", "root-system", "highest-coroot", "comarks", "exact"),
        discovery_terms=("highest coroot", "comarks", "dual highest root"),
        examples=(
            OperationExample(
                name="g2_highest_coroot",
                description=(
                    "The stated G2 ordering has highest coroot coefficients (2,3) "
                    "and comarks (2,3); the result also retains its preimage root "
                    "and source pair index."
                ),
                input={
                    "matrix": {
                        "matrix": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["2", "-3"], ["-1", "2"]],
                        },
                        "simple_root_axis": [0, 1],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.root_to_coroot.compute",
        title="Convert one positive root to its coroot",
        description=(
            "Map one positive root from the supplied finite Cartan datum's "
            "simple-root basis to its coroot in the matching simple-coroot "
            "basis. Return the exact squared root length and retain the full "
            "datum. The input must be a positive root of that datum."
        ),
        request_type=RootToCorootRequest,
        result_type=RootToCorootResult,
        run=_run_root_to_coroot,
        tags=("algebra", "root-system", "coroot", "exact"),
        discovery_terms=("root to coroot", "coroot of a positive root"),
        examples=(
            OperationExample(
                name="b2_short_root_coroot",
                description=(
                    "In B2, the positive root (1,1) has squared length 2 and "
                    "coroot coordinates (1,2) in the simple-coroot basis."
                ),
                input={
                    "matrix": {
                        "matrix": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["2", "-2"], ["-1", "2"]],
                        },
                        "simple_root_axis": [0, 1],
                    },
                    "root_coefficients": [1, 1],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.root_length_profile.compute",
        title="Group positive roots by exact squared length",
        description=(
            "Return the complete positive-root family grouped into exact squared-length "
            "classes separately for each irreducible component. Length ratios are local "
            "to each component's normalization, so unrelated factors are not compared. "
            "The finite Cartan rank is at most 8 and the root family at most 120."
        ),
        request_type=CartanMatrixRequest,
        result_type=RootLengthProfileResult,
        run=_run_root_length_profile,
        tags=("algebra", "root-system", "root-length", "exact"),
        discovery_terms=("root lengths", "long and short roots", "root length ratio"),
        examples=(
            OperationExample(
                name="b2_root_length_profile",
                description=(
                    "For the stated B2 ordering, the two squared-root-length classes "
                    "are 2 and 4, with ratio 2; the component normalization sets "
                    "the first simple-root squared length to 2."
                ),
                input={
                    "matrix": {
                        "matrix": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["2", "-2"], ["-1", "2"]],
                        },
                        "simple_root_axis": [0, 1],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.weyl_poincare_polynomial.compute",
        title="Compute the Weyl-group Poincare polynomial",
        description=(
            "Return the exact length-generating polynomial of a finite Weyl "
            "group in descending-degree integer coefficient order. The operation "
            "uses the Weyl exponents and does not enumerate group elements. "
            "Rank is at most 8 and degree is at most 120."
        ),
        request_type=CartanMatrixRequest,
        result_type=WeylPoincarePolynomialResult,
        run=_run_weyl_poincare_polynomial,
        tags=("algebra", "root-system", "weyl-group", "polynomial", "exact"),
        discovery_terms=(
            "Weyl group length generating polynomial",
            "Poincare polynomial of a Weyl group",
            "Weyl group length profile",
        ),
        examples=(
            OperationExample(
                name="a2_weyl_poincare_polynomial",
                description="Compute the length-generating polynomial of W(A2).",
                input={
                    "matrix": {
                        "matrix": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["2", "-1"], ["-1", "2"]],
                        },
                        "simple_root_axis": [0, 1],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.weyl_exponents.compute",
        title="Compute finite Weyl-group exponents",
        description=(
            "Return the exact Weyl exponents for each connected component of a "
            "finite Cartan datum. The result preserves the component's simple-root "
            "indices, including for reducible systems. Rank is at most 8."
        ),
        request_type=CartanMatrixRequest,
        result_type=WeylExponentsResult,
        run=_run_weyl_exponents,
        tags=("algebra", "root-system", "weyl-group", "exact"),
        discovery_terms=(
            "Weyl exponents",
            "degrees of invariant polynomials",
            "Coxeter exponents",
        ),
        examples=(
            OperationExample(
                name="g2_weyl_exponents",
                description="Compute the two exponents of the finite G2 Weyl group.",
                input={
                    "matrix": {
                        "matrix": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["2", "-3"], ["-1", "2"]],
                        },
                        "simple_root_axis": [0, 1],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.positive_root_profile.compute",
        title="Compute positive-root heights, supports, and highest roots",
        description=(
            "Return every positive root in canonical lexicographic simple-root "
            "coordinates with its exact height, simple-root support, and factor "
            "index. For each irreducible factor, return the index of the unique "
            "highest positive root on that shared root axis. Rank is at most 8 "
            "and the complete root family is bounded by 120 entries."
        ),
        request_type=CartanMatrixRequest,
        result_type=PositiveRootProfileResult,
        run=_run_positive_root_profile,
        tags=("algebra", "root-system", "positive-roots", "exact"),
        discovery_terms=(
            "positive root heights",
            "simple root support",
            "highest root",
            "root marks",
        ),
        examples=(
            OperationExample(
                name="g2_positive_root_profile",
                description=(
                    "Compute heights and supports for all six positive roots of G2; "
                    "the matrix must be finite type and rank at most 8."
                ),
                input={
                    "matrix": {
                        "matrix": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["2", "-3"], ["-1", "2"]],
                        },
                        "simple_root_axis": [0, 1],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.root_poset.compute",
        title="Construct the positive-root poset",
        description=(
            "Return the complete poset of positive roots ordered by coordinatewise "
            "difference in the datum's ordered simple-root basis. The result binds "
            "each poset label to an exact root-coordinate vector and the finite "
            "Cartan datum. At most 64 positive roots are admitted; larger families "
            "are rejected before pair comparisons or poset expansion."
        ),
        request_type=CartanMatrixRequest,
        result_type=RootPosetResult,
        run=_run_root_poset,
        tags=("algebra", "root-system", "root-poset", "exact"),
        discovery_terms=("positive root poset", "root order", "root height poset"),
        examples=(
            OperationExample(
                name="a2_root_poset",
                description=(
                    "Construct the three-element positive-root poset of A2; "
                    "the Cartan matrix must be finite type and have at most "
                    "64 positive roots."
                ),
                input={"matrix": _A2["matrix"]},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.simple_reflection.compute",
        title="Apply a simple reflection to a root lattice vector",
        description="Apply the simple reflection s_i to a vector in the root lattice "
        "of a finite crystallographic root system defined by its Cartan "
        f"matrix of rank at most {MAX_RANK}. The vector uses that same "
        "simple-root axis and the index is zero-based.",
        request_type=SimpleReflectionRequest,
        result_type=SimpleReflectionResult,
        run=_run_simple_reflection,
        tags=("algebra", "root-system", "exact"),
        examples=(
            OperationExample(
                name="a2_reflection",
                description="Apply s_0 to the simple root alpha_0 in A2. The matrix is "
                "a finite-type generalized Cartan matrix, the vector has its "
                "two simple-root coordinates, and index 0 is below its rank.",
                input={"matrix": _A2["matrix"], "vector": [1, 0], "simple_index": 0},
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.simple_reflections.compute",
        title="Compute all simple-reflection matrices of a Weyl group",
        description=(
            "Compute the exact action matrix of every simple reflection s_i on "
            "the root, coroot, weight, and coweight lattices of a finite "
            "crystallographic root system from its Cartan matrix of rank at "
            f"most {MAX_RANK}. Each returned reflection squares to the "
            "identity, replayed by the kernel before construction."
        ),
        request_type=CartanMatrixRequest,
        result_type=SimpleReflectionsResult,
        run=_run_simple_reflections,
        tags=("algebra", "weyl-group", "root-system", "reflection", "exact"),
        discovery_terms=(
            "simple reflection matrices",
            "Weyl group generators",
            "reflection action on weights",
            "coroot reflection",
        ),
        examples=(
            OperationExample(
                name="a2_simple_reflections",
                description="Compute the two simple-reflection matrices of A2 on all "
                "four lattices; the matrix must be a valid finite-type Cartan "
                "matrix of rank at most 8.",
                input={"matrix": _A2["matrix"]},
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.parabolic.compute",
        title="Construct a standard parabolic Weyl subgroup profile",
        description=(
            "Return the selected simple-reflection generators, their induced "
            "principal Cartan submatrix, and the exact subgroup order, all "
            "embedded in the parent Cartan datum. The empty generator set is "
            "the trivial subgroup. Rank is at most 8."
        ),
        request_type=WeylParabolicRequest,
        result_type=WeylParabolicResult,
        run=_run_weyl_parabolic,
        tags=("algebra", "root-system", "weyl-group", "parabolic", "exact"),
        discovery_terms=(
            "standard parabolic subgroup of a Weyl group",
            "Weyl subgroup generated by simple reflections",
            "parabolic Cartan submatrix and order",
        ),
        examples=(
            OperationExample(
                name="a3_standard_parabolic",
                description=(
                    "The A3 parabolic generated by simple reflections 0 and 1 "
                    "has Cartan type A2 and order 6."
                ),
                input={**_A3, "simple_root_indices": [0, 1]},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.weyl_group_order.compute",
        title="Compute the exact order of a Weyl group",
        description="Compute the exact order of the Weyl group of a finite "
        "crystallographic root system from its Cartan matrix. The kernel "
        "constructs the bounded complete signed-root action and uses SymPy's "
        "Schreier-Sims order algorithm; it never enumerates Weyl-group elements.",
        request_type=CartanMatrixRequest,
        result_type=WeylGroupOrderResult,
        run=_run_weyl_group_order,
        tags=("algebra", "root-system", "exact"),
        examples=(
            OperationExample(
                name="a2_weyl_group_order",
                description="Compute the order of the A2 Weyl group, which is 6; "
                "the matrix must be a finite-type generalized Cartan "
                "matrix of rank at most 8.",
                input={"matrix": _A2["matrix"]},
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.element.length.compute",
        title="Compute the length and inversion set of a Weyl-group word",
        description="Apply a word in the simple reflections to every positive "
        "root of a finite crystallographic root system and count the "
        "inversions: positive roots sent negative. That count is the "
        "element's length, and the word is reduced exactly when the count "
        "equals its factor count.",
        request_type=WeylElementRequest,
        result_type=WeylElementLengthResult,
        run=_run_weyl_element_length,
        tags=("algebra", "weyl-group", "root-system", "exact"),
        discovery_terms=(
            "weyl group element length",
            "inversion set of a weyl word",
            "reduced word check",
        ),
        examples=(
            OperationExample(
                name="a2_longest_word_length",
                description="Compute length 3 and the full inversion set of "
                "s_0 s_1 s_0 in A2; each word index must be below the "
                "Cartan rank and the word within the length budget.",
                input={"matrix": _A2["matrix"], "word": [0, 1, 0]},
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.element.order.compute",
        title="Compute the exact order of a Weyl-group element",
        description=(
            "Compute the order of the element represented by the supplied bounded "
            "simple-reflection word. The kernel forms its permutation on the complete "
            "signed-root set, whose action is faithful, and returns the lcm of the "
            "cycle lengths. Rank is at most 8, the word at most 1024 factors, and "
            "the signed-root set at most 240 elements."
        ),
        request_type=WeylElementRequest,
        result_type=WeylElementOrderResult,
        run=_run_weyl_element_order,
        tags=("algebra", "weyl-group", "root-system", "element-order", "exact"),
        discovery_terms=("Weyl element order", "order of a Weyl group element"),
        examples=(
            OperationExample(
                name="a2_coxeter_element_order",
                description=(
                    "The product s_0 s_1 in A2 is a Coxeter element of order 3; "
                    "the word indices are applied left to right."
                ),
                input={**_A2, "word": [0, 1]},
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.element.from_word.compute",
        title="Construct a Weyl element from a simple-reflection word",
        description=(
            "Construct the canonical integer action matrix on the ordered simple-root "
            "basis. Equal Weyl elements have the same value regardless of their words."
        ),
        request_type=WeylElementRequest,
        result_type=WeylElement,
        run=_run_weyl_element_from_word,
        tags=("algebra", "root-system", "weyl-group", "exact"),
        discovery_terms=("Weyl element action matrix", "construct Weyl group element"),
        examples=(
            OperationExample(
                name="a2_reflection_product",
                description="Construct s0 s1 on the A2 simple-root lattice.",
                input={**_A2, "word": [0, 1]},
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.element.compose",
        title="Compose two Weyl elements",
        description="Compose canonical Weyl action matrices with an exact common Cartan parent.",
        request_type=WeylElementComposeRequest,
        result_type=WeylElement,
        run=_run_weyl_element_compose,
        tags=("algebra", "root-system", "weyl-group", "exact"),
        discovery_terms=("compose Weyl elements",),
        examples=(
            OperationExample(
                name="compose_a2_identity_elements",
                description="Compose two identity actions on the A2 root lattice.",
                input={
                    "first": {
                        **_A2,
                        "root_action": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["1", "0"], ["0", "1"]],
                        },
                    },
                    "then": {
                        **_A2,
                        "root_action": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["1", "0"], ["0", "1"]],
                        },
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.element.inverse",
        title="Invert a Weyl element",
        description="Return the exact inverse action matrix of a validated Weyl element.",
        request_type=WeylElementInverseRequest,
        result_type=WeylElement,
        run=_run_weyl_element_inverse,
        tags=("algebra", "root-system", "weyl-group", "exact"),
        discovery_terms=("inverse Weyl element",),
        examples=(
            OperationExample(
                name="invert_a2_identity",
                description="Invert the identity action on the A2 root lattice.",
                input={
                    "element": {
                        **_A2,
                        "root_action": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["1", "0"], ["0", "1"]],
                        },
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.element.act_on_weight.compute",
        title="Apply a Weyl element to an exact weight vector",
        description=(
            "Apply a parent-bound finite Weyl element to a weight-lattice "
            "vector in the same ordered Cartan datum. The result is the "
            "canonical weight-lattice value in fundamental-weight coordinates; "
            "the action is induced from the element's exact root-lattice matrix."
        ),
        request_type=WeylElementWeightActionRequest,
        result_type=WeightLatticeVector,
        run=_run_weyl_element_weight_action,
        tags=("algebra", "root-system", "weyl-group", "weight", "action", "exact"),
        discovery_terms=(
            "Weyl element act on weight vector",
            "Weyl group action on fundamental weight coordinates",
            "apply Weyl transformation to weight lattice value",
        ),
        examples=(
            OperationExample(
                name="a2_simple_reflection_on_fundamental_weight",
                description=(
                    "Apply s0 to the first fundamental weight of A2. Since "
                    "alpha0=(2,-1) in fundamental-weight coordinates, the "
                    "image is (-1,1)."
                ),
                input={
                    "element": {
                        "matrix": _A2["matrix"],
                        "root_action": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["-1", "1"], ["0", "1"]],
                        },
                    },
                    "weight": {
                        "datum": {
                            "cartan_matrix": _A2["matrix"],
                            "symmetrizer": [
                                {"num": "1", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                            "root_to_weight": {
                                "domain": "ZZ",
                                "row_count": 2,
                                "column_count": 2,
                                "entries": [["2", "-1"], ["-1", "2"]],
                            },
                            "coroot_to_coweight": {
                                "domain": "ZZ",
                                "row_count": 2,
                                "column_count": 2,
                                "entries": [["2", "-1"], ["-1", "2"]],
                            },
                        },
                        "coordinates": [1, 0],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.element.act_on_root.compute",
        title="Apply a Weyl element to a root-lattice vector",
        description=(
            "Apply a canonical finite Weyl element to a root-lattice vector in "
            "the same ordered Cartan datum. The result is the exact vector in "
            "simple-root coordinates; the input need not itself be a root."
        ),
        request_type=WeylElementRootActionRequest,
        result_type=RootLatticeVector,
        run=_run_weyl_element_root_action,
        tags=(
            "algebra",
            "root-system",
            "weyl-group",
            "root-lattice",
            "action",
            "exact",
        ),
        discovery_terms=(
            "Weyl element act on root vector",
            "Weyl group action on simple-root coordinates",
            "apply Weyl transformation to root-lattice value",
        ),
        examples=(
            OperationExample(
                name="a2_simple_reflection_on_root_vector",
                description=(
                    "Apply s0 to the first simple root of A2. The vector remains "
                    "in the same root lattice and becomes (-1, 0)."
                ),
                input={
                    "element": {
                        "matrix": _A2["matrix"],
                        "root_action": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["-1", "1"], ["0", "1"]],
                        },
                    },
                    "vector": {
                        "datum": _A2_LATTICE_DATUM,
                        "coordinates": [1, 0],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.word.act_on_root_vector.compute",
        title="Apply a Weyl word to a root-lattice vector",
        description=(
            "Apply a bounded sequence of simple reflections, in the declared "
            "left-to-right order, to an integer vector in the Cartan matrix's "
            "simple-root basis. The result remains in that same root-lattice "
            "axis; no root-membership claim is made for the input vector."
        ),
        request_type=WeylVectorActionRequest,
        result_type=WeylVectorActionResult,
        run=_run_weyl_vector_action,
        tags=("algebra", "root-system", "weyl-group", "action", "exact"),
        discovery_terms=(
            "Weyl group action on root lattice",
            "apply Weyl word to vector",
            "root lattice vector action",
        ),
        examples=(
            OperationExample(
                name="a2_word_action",
                description=(
                    "Apply s0 s1 to the A2 simple-root coefficient vector "
                    "(1, 0), with factors applied left to right."
                ),
                input={**_A2, "word": [0, 1], "vector": [1, 0]},
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.weight.orbit.compute",
        title="Compute a complete Weyl orbit of an integral weight",
        description=(
            "Return the complete orbit of an integral weight whose coordinates "
            "are pairings with the ordered simple coroots, equivalently its "
            "fundamental-weight coordinates. The finite crystallographic "
            f"Cartan rank is at most {MAX_RANK}; exact orbit-stabilizer and "
            "invariant-norm bounds are admitted before expansion, and the "
            "orbit is limited to 4096 values."
        ),
        request_type=WeylWeightOrbitRequest,
        result_type=WeylWeightOrbitResult,
        run=_run_weyl_weight_orbit,
        tags=("algebra", "root-system", "weyl-group", "weight", "orbit", "exact"),
        discovery_terms=(
            "Weyl orbit of an integral weight",
            "fundamental weight orbit",
            "weight orbit under Weyl group",
        ),
        examples=(
            OperationExample(
                name="a2_fundamental_weight_orbit",
                description=(
                    "Compute the three weights in the A2 orbit of the first "
                    "fundamental weight (1, 0). Coordinates pair against the "
                    "ordered simple coroots."
                ),
                input={**_A2, "weight": [1, 0]},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.weyl_dimension.compute",
        title="Compute an exact Weyl dimension from a dominant highest weight",
        description=(
            "Compute the integer dimension of the irreducible representation "
            "with a nonnegative integral highest weight in fundamental-weight "
            "coordinates. The result retains the ordered Cartan weight axis and "
            "the exact numerator and denominator pairing for every positive-root "
            "factor, including reducible finite root systems."
        ),
        request_type=WeylDimensionRequest,
        result_type=WeylDimensionResult,
        run=_run_weyl_dimension,
        tags=("algebra", "root-system", "representation", "weyl-dimension", "exact"),
        discovery_terms=(
            "Weyl dimension formula",
            "dimension of irreducible Lie algebra representation",
            "highest weight representation dimension",
        ),
        examples=(
            OperationExample(
                name="g2_seven_dimensional_representation",
                description=(
                    "Compute dimension 7 for the first fundamental G2 weight. "
                    "Coordinates are pairings against the ordered simple coroots."
                ),
                input={
                    "matrix": {
                        "matrix": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["2", "-3"], ["-1", "2"]],
                        },
                        "simple_root_axis": [0, 1],
                    },
                    "highest_weight": [1, 0],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.longest_element.compute",
        title="Compute a reduced word for the longest Weyl-group element",
        description="Search the weak order for the longest element of a "
        "finite Weyl group by greedy ascent, returning a reduced word "
        "whose length equals the positive-root count. The kernel asserts "
        "that maximality before construction.",
        request_type=CartanMatrixRequest,
        result_type=WeylLongestElementResult,
        run=_run_weyl_longest_element,
        tags=("algebra", "weyl-group", "root-system", "exact"),
        discovery_terms=(
            "longest weyl group element",
            "longest reduced word",
            "weak order maximum",
        ),
        examples=(
            OperationExample(
                name="a2_longest_element",
                description="Compute a reduced longest word of length 3 in A2; "
                "the matrix must be a valid finite-type Cartan matrix.",
                input={"matrix": _A2["matrix"]},
            ),
        ),
    ),
    MathTool(
        operation_id="weyl_group.element.descents.compute",
        title="Compute the left and right descent sets of a Weyl-group word",
        description="Find the simple reflections that lower a Weyl-group "
        "word's length when prepended (right descents) or appended (left "
        "descents) by testing the word and its reversal on simple roots.",
        request_type=WeylElementRequest,
        result_type=WeylDescentsResult,
        run=_run_weyl_element_descents,
        tags=("algebra", "weyl-group", "root-system", "exact"),
        discovery_terms=(
            "weyl group descent sets",
            "left right descents",
            "length lowering reflections",
        ),
        examples=(
            OperationExample(
                name="a2_asymmetric_descents",
                description="Compute left {1} and right {0} descents of s_1 s_0 "
                "in A2; each word index must be below the Cartan rank and "
                "the word within the length budget.",
                input={"matrix": _A2["matrix"], "word": [0, 1]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
