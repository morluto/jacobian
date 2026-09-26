"""Koszul complex operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.koszul._models import KoszulComplexRequest
from jacobian.math.koszul.dga_operations import module_koszul_dga
from jacobian.math.koszul.homology_map import koszul_homology_map
from jacobian.math.koszul.module_models import (
    ModuleKoszulChainMap,
    ModuleKoszulComplex,
    ModuleKoszulDGA,
    ModuleKoszulDGARequest,
    ModuleKoszulDifferentialRequest,
    ModuleKoszulDifferentialValue,
    ModuleKoszulDirectSumRequest,
    ModuleKoszulDirectSumValue,
    ModuleKoszulExactnessProfile,
    ModuleKoszulHomology,
    ModuleKoszulHomologyMap,
    ModuleKoszulHomologyMapRequest,
    ModuleKoszulHomologyRequest,
    ModuleKoszulMapRequest,
    ModuleKoszulRequest,
    ModuleKoszulSequencePermutation,
    ModuleKoszulSequencePermutationRequest,
    ModuleKoszulUnitContraction,
    ModuleKoszulUnitContractionRequest,
    ModuleKoszulZeroExtension,
    ModuleKoszulZeroExtensionRequest,
    ModuleQuotientValue,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_append_zero,
    module_koszul_complex,
    module_koszul_differential,
    module_koszul_direct_sum,
    module_koszul_exactness_profile,
    module_koszul_homology,
    module_koszul_map,
    module_koszul_quotient,
    module_koszul_sequence_permute,
    module_koszul_unit_contract,
)
from jacobian.math.koszul.operations import koszul_complex
from jacobian.math.koszul.values import (
    MAX_KOSZUL_DEGREE,
    MAX_KOSZUL_SEQUENCE_LENGTH,
    MAX_KOSZUL_TERMS,
    MAX_KOSZUL_VARIABLES,
    KoszulComplexValue,
)


def _koszul_complex(request: KoszulComplexRequest) -> KoszulComplexValue:
    """Project a wire request into the canonical Koszul construction."""

    return koszul_complex(request.variables, request.sequence)


def _monomial(variables: list[str], exponents: list[int]) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": variables,
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": "1", "den": "1"},
                    "exponents": exponents,
                }
            ]
        },
    }


_XY_RING = ["x", "y"]

_KOSZUL_XY_EXAMPLE = {
    "variables": _XY_RING,
    "sequence": [
        _monomial(_XY_RING, [1, 0]),
        _monomial(_XY_RING, [0, 1]),
    ],
}


_MODULE_EXAMPLE: dict[str, Any] = {
    "algebra": {
        "basis": ["1"],
        "multiplication": [[[{"num": "1", "den": "1"}]]],
        "unit": [{"num": "1", "den": "1"}],
    },
    "module": {
        "algebra": {
            "basis": ["1"],
            "multiplication": [[[{"num": "1", "den": "1"}]]],
            "unit": [{"num": "1", "den": "1"}],
        },
        "basis": ["m"],
        "action": [[[{"num": "1", "den": "1"}]]],
    },
    "sequence": [[{"num": "1", "den": "1"}]],
}
_MODULE_EXAMPLE_COMPLEX = {
    "algebra": _MODULE_EXAMPLE["algebra"],
    "module": _MODULE_EXAMPLE["module"],
    "sequence": _MODULE_EXAMPLE["sequence"],
    "basis_sizes": [1, 1],
    "differentials": [
        {
            "row_count": 1,
            "column_count": 1,
            "entries": [[0, 0, {"num": "1", "den": "1"}]],
        }
    ],
    "square_zero": True,
}

TOOLS: MathTools = (
    MathTool(
        operation_id="homological.koszul.homology_map.compute",
        title="Induce exact maps on finite-module Koszul homology",
        description=(
            "Revalidate a typed module-induced Koszul chain map and compute its "
            "exact matrices on every homology group in the returned canonical "
            "homology bases. Combined chain-basis size and rational coefficient "
            "growth, exact quotient-coordinate work, and aggregate output use a "
            "deliberately small finite envelope; output includes the source and "
            "target homology bases for interpreting coordinates."
        ),
        request_type=ModuleKoszulHomologyMapRequest,
        result_type=ModuleKoszulHomologyMap,
        run=koszul_homology_map,
        tags=("koszul", "homology", "chain-map", "exact"),
        discovery_terms=(
            "induced map on Koszul homology",
            "functoriality of Koszul homology",
            "module map homology matrix",
        ),
        examples=(
            OperationExample(
                name="identity_on_degree_zero_homology",
                description="The identity map of QQ induces the identity on H0 of the empty-sequence complex.",
                input={
                    "chain_map": {
                        "algebra": _MODULE_EXAMPLE["algebra"],
                        "source": _MODULE_EXAMPLE["module"],
                        "target": _MODULE_EXAMPLE["module"],
                        "sequence": [],
                        "module_map": [[{"num": "1", "den": "1"}]],
                        "source_complex": {
                            "algebra": _MODULE_EXAMPLE["algebra"],
                            "module": _MODULE_EXAMPLE["module"],
                            "sequence": [],
                            "basis_sizes": [1],
                            "differentials": [],
                            "square_zero": True,
                        },
                        "target_complex": {
                            "algebra": _MODULE_EXAMPLE["algebra"],
                            "module": _MODULE_EXAMPLE["module"],
                            "sequence": [],
                            "basis_sizes": [1],
                            "differentials": [],
                            "square_zero": True,
                        },
                        "degree_maps": [
                            {
                                "row_count": 1,
                                "column_count": 1,
                                "entries": [[0, 0, {"num": "1", "den": "1"}]],
                            }
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="homological.koszul.module_map.compute",
        title="Induce a map between finite-module Koszul complexes",
        description=(
            "Check an exact QQ-linear map is a homomorphism over the supplied "
            "finite commutative algebra, then return its degreewise induced "
            "chain map between Koszul complexes on the same ordered sequence. "
            "Action checks, wedge axes, chain-map equations, work, and output "
            "are bounded before publication."
        ),
        request_type=ModuleKoszulMapRequest,
        result_type=ModuleKoszulChainMap,
        run=module_koszul_map,
        tags=("koszul", "module", "chain-map", "exact"),
        discovery_terms=(
            "module homomorphism induces map on Koszul complexes",
            "functoriality of Koszul homology",
            "chain map from a module map",
        ),
        examples=(
            OperationExample(
                name="identity_module_map",
                description="The identity of a one-dimensional module induces identity maps in every Koszul degree.",
                input={
                    "algebra": _MODULE_EXAMPLE["algebra"],
                    "source": _MODULE_EXAMPLE["module"],
                    "target": _MODULE_EXAMPLE["module"],
                    "sequence": _MODULE_EXAMPLE["sequence"],
                    "map_matrix": [[{"num": "1", "den": "1"}]],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="homological.koszul.module_direct_sum.compute",
        title="Compute Koszul complexes of a direct-sum module",
        description=(
            "Construct K(f; M direct_sum N) and the two source-bound summand "
            "complexes over one finite commutative algebra. Returns canonical "
            "degreewise inclusions whose images partition every target basis; "
            "both inclusions are checked to commute with each differential. "
            "The aggregate bases, differential contributions and exact matrices "
            "are admitted before complex construction."
        ),
        request_type=ModuleKoszulDirectSumRequest,
        result_type=ModuleKoszulDirectSumValue,
        run=module_koszul_direct_sum,
        tags=("koszul", "module", "direct-sum", "chain-map", "exact"),
        examples=(
            OperationExample(
                name="dual_number_module_split",
                description="Split the Koszul complex of a direct sum of two modules over the dual numbers.",
                input={
                    "algebra": _MODULE_EXAMPLE["algebra"],
                    "left": _MODULE_EXAMPLE["module"],
                    "right": {
                        **_MODULE_EXAMPLE["module"],
                        "basis": ["n"],
                        "action": [[[{"num": "1", "den": "1"}]]],
                    },
                    "sequence": [[{"num": "1", "den": "1"}]],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="homological.koszul.differential.compute",
        title="Compute one degree of a finite-module Koszul differential",
        description=(
            "Compute d_k: M tensor Lambda^k -> M tensor Lambda^(k-1) for one "
            "selected degree of a finite-module Koszul sequence. Returns the "
            "source-bound module/algebra/sequence, canonical increasing wedge "
            "axes, and exact sparse matrix. Only this differential and its "
            "predecessor are materialized; d_(k-1)d_k is replayed exactly. "
            "The current finite-module sequence and matrix admission limits apply."
        ),
        request_type=ModuleKoszulDifferentialRequest,
        result_type=ModuleKoszulDifferentialValue,
        run=module_koszul_differential,
        tags=("koszul", "differential", "exact"),
        examples=(
            OperationExample(
                name="one_dimensional_module_differential",
                description=(
                    "Compute d_1 for the one-dimensional module over QQ with "
                    "the unit as its Koszul sequence element."
                ),
                input={"request": _MODULE_EXAMPLE, "degree": 1},
            ),
        ),
    ),
    MathTool(
        operation_id="koszul.complex.construct.compute",
        title="Construct the exact Koszul complex of a polynomial sequence",
        description=(
            "Construct the complete Koszul complex K(f) = R (x) Lambda(R^c) of "
            "one ordered sequence f = (f_1, ..., f_c) in R = QQ[x_1, ..., x_m] "
            "with the module fixed to the ring itself: per-degree wedge bases "
            "e_{i_1} ^ ... ^ e_{i_k} in canonical increasing order (degree k "
            "has C(c, k) basis elements), the differentials d_k(e_I) = "
            "sum_j (-1)^position f_j e_{I minus j} as exact sparse polynomial "
            "matrices bound to the ordered bases, the exactly replayed "
            "d^2 = 0 identity, and — when the ambient ring is QQ and the wedge "
            "ranks fit the shared envelope — the exact conversion to the based "
            "chain-complex value so downstream homology operations compose "
            "unchanged. Supply 'variables' as the ordered ring axis and "
            f"'sequence' as at most {MAX_KOSZUL_SEQUENCE_LENGTH} canonical "
            f"sparse QQ polynomials on that axis (at most "
            f"{MAX_KOSZUL_VARIABLES} variables, {MAX_KOSZUL_TERMS} terms, and "
            f"degree {MAX_KOSZUL_DEGREE} per variable); the empty sequence "
            "returns the identity complex R in degree 0. Based modules, "
            "DG-algebra structure, homology profiles, and sequence transforms "
            "are deferred."
        ),
        request_type=KoszulComplexRequest,
        result_type=KoszulComplexValue,
        run=_koszul_complex,
        tags=(
            "koszul-complex",
            "commutative-algebra",
            "homological-algebra",
            "chain-complex",
            "exact",
        ),
        discovery_terms=(
            "Koszul complex",
            "Koszul differential",
            "exterior algebra differential",
            "wedge basis chain complex",
            "regular sequence complex",
            "sequence-derived differential",
        ),
        examples=(
            OperationExample(
                name="koszul_complex_of_x_y",
                description=(
                    "Construct K(x, y) over QQ[x, y]: ranks (1, 2, 1) with "
                    "d_1 = [x, y] and d_2(e_12) = x e_2 - y e_1; supply the "
                    "ordered variable axis and the sequence in canonical "
                    "sparse form."
                ),
                input=_KOSZUL_XY_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="homological.koszul.dga.compute",
        title="Construct a finite-algebra Koszul differential graded algebra",
        description=(
            "Construct the complete unital DGA K(f; A) for an explicit finite "
            "commutative rational algebra with a verified two-sided unit. The "
            "result includes its exact module Koszul differential and the sparse "
            "multiplication table induced by algebra multiplication and exterior "
            "wedge signs. Chain basis, identity work, complete product entries, "
            "coefficient growth, and output bytes are admitted before expansion."
        ),
        request_type=ModuleKoszulDGARequest,
        result_type=ModuleKoszulDGA,
        run=module_koszul_dga,
        tags=("koszul", "dga", "differential-graded-algebra", "exact"),
        discovery_terms=(
            "Koszul differential graded algebra",
            "Koszul wedge multiplication",
            "graded Leibniz Koszul complex",
        ),
        examples=(
            OperationExample(
                name="two_generator_koszul_dga_over_qq",
                description=(
                    "Construct K(0,0; QQ), retaining the unit and signed exterior "
                    "product e_0 e_1 = - e_1 e_0."
                ),
                input={
                    "algebra": {
                        "basis": ["1"],
                        "multiplication": [[[{"num": "1", "den": "1"}]]],
                        "unit": [{"num": "1", "den": "1"}],
                    },
                    "sequence": [
                        [{"num": "0", "den": "1"}],
                        [{"num": "0", "den": "1"}],
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="homological.koszul.complex.compute",
        title="Construct a finite-module Koszul complex",
        description="Construct the exact Koszul complex of a finite based module over a finite commutative algebra, retaining algebra/module axes and checking d²=0.",
        request_type=ModuleKoszulRequest,
        result_type=ModuleKoszulComplex,
        run=module_koszul_complex,
        tags=("koszul", "module", "homological-algebra", "exact"),
        examples=(
            OperationExample(
                name="unit_module",
                description="Construct the one-term Koszul complex of the unit in a one-dimensional algebra/module; the module action must respect multiplication.",
                input=_MODULE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="homological.koszul.sequence_permute.compute",
        title="Permute a finite-module Koszul sequence",
        description=(
            "Rebuild the Koszul complex in the requested sequence order and "
            "return explicit signed exterior-power chain isomorphisms in both "
            "directions. The source differential is checked against its retained "
            "algebra, module action, and sequence before transport."
        ),
        request_type=ModuleKoszulSequencePermutationRequest,
        result_type=ModuleKoszulSequencePermutation,
        run=module_koszul_sequence_permute,
        tags=("koszul", "sequence", "permutation", "chain-isomorphism", "exact"),
        examples=(
            OperationExample(
                name="swap_two_unit_entries",
                description=(
                    "Swap two equal unit entries; the degree-two exterior basis "
                    "map still records the minus sign."
                ),
                input={
                    "complex": {
                        "algebra": _MODULE_EXAMPLE["algebra"],
                        "module": _MODULE_EXAMPLE["module"],
                        "sequence": _MODULE_EXAMPLE["sequence"] * 2,
                        "basis_sizes": [1, 2, 1],
                        "differentials": [
                            {
                                "row_count": 1,
                                "column_count": 2,
                                "entries": [
                                    [0, 0, {"num": "1", "den": "1"}],
                                    [0, 1, {"num": "1", "den": "1"}],
                                ],
                            },
                            {
                                "row_count": 2,
                                "column_count": 1,
                                "entries": [
                                    [0, 0, {"num": "-1", "den": "1"}],
                                    [1, 0, {"num": "1", "den": "1"}],
                                ],
                            },
                        ],
                    },
                    "new_to_old": [1, 0],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="homological.koszul.unit_contraction.compute",
        title="Contract a finite-module Koszul complex at a unit entry",
        description=(
            "Return the exact degree-raising homotopy induced by the inverse of "
            "a unit sequence entry. The operation checks dH + Hd = identity "
            "on the retained sequence-derived complex."
        ),
        request_type=ModuleKoszulUnitContractionRequest,
        result_type=ModuleKoszulUnitContraction,
        run=module_koszul_unit_contract,
        tags=("koszul", "unit", "contracting-homotopy", "exact"),
        examples=(
            OperationExample(
                name="one_term_unit",
                description="Contract the one-term Koszul complex of the unit entry.",
                input={"complex": _MODULE_EXAMPLE_COMPLEX, "unit_index": 0},
            ),
        ),
    ),
    MathTool(
        operation_id="homological.koszul.append_zero.compute",
        title="Append a zero entry to a finite-module Koszul sequence",
        description=(
            "Construct K(f_1,...,f_r,0;M) and return the canonical unshifted "
            "and shifted inclusions/projections splitting it as K(f;M) plus "
            "its degree shift. Exact chain and splitting identities are checked."
        ),
        request_type=ModuleKoszulZeroExtensionRequest,
        result_type=ModuleKoszulZeroExtension,
        run=module_koszul_append_zero,
        tags=("koszul", "sequence", "zero-entry", "chain-splitting", "exact"),
        examples=(
            OperationExample(
                name="append_zero_to_one_term_complex",
                description="Split the Koszul complex of (1, 0) into the contractible two-term summand and its shift.",
                input={"complex": _MODULE_EXAMPLE_COMPLEX},
            ),
        ),
    ),
    MathTool(
        operation_id="homological.koszul.homology.compute",
        title="Compute finite-module Koszul homology",
        description=(
            "Compute exact cycle, boundary, and homology bases in every degree "
            "of a retained finite-module Koszul complex. Basis vectors use the "
            "source-bound chain axes; the result also returns their dimensions."
        ),
        request_type=ModuleKoszulHomologyRequest,
        result_type=ModuleKoszulHomology,
        run=lambda request: module_koszul_homology(request.complex),
        tags=("koszul", "homology", "exact"),
        examples=(
            OperationExample(
                name="empty_sequence",
                description="Compute homology of the empty finite-module Koszul complex; the source complex retains its module axis.",
                input={
                    "complex": {
                        "algebra": _MODULE_EXAMPLE["algebra"],
                        "module": _MODULE_EXAMPLE["module"],
                        "sequence": [],
                        "basis_sizes": [1],
                        "differentials": [],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="homological.koszul.exactness_profile.compute",
        title="Compute positive-degree Koszul exactness profile",
        description=(
            "Return every homology dimension and whether this supplied finite "
            "Koszul complex is acyclic above degree zero. If higher homology is "
            "nonzero, include a concrete first-degree class representative in "
            "the retained chain basis. This does not infer a theorem-level "
            "regular-sequence property."
        ),
        request_type=ModuleKoszulHomologyRequest,
        result_type=ModuleKoszulExactnessProfile,
        run=module_koszul_exactness_profile,
        tags=("koszul", "homology", "exactness", "exact"),
        examples=(
            OperationExample(
                name="unit_sequence_is_acyclic_above_zero",
                description=(
                    "The unit sequence on the regular module gives an exact "
                    "positive-degree profile for this finite complex."
                ),
                input={
                    "complex": {
                        "algebra": _MODULE_EXAMPLE["algebra"],
                        "module": _MODULE_EXAMPLE["module"],
                        "sequence": _MODULE_EXAMPLE["sequence"],
                        "basis_sizes": [1, 1],
                        "differentials": [
                            {
                                "row_count": 1,
                                "column_count": 1,
                                "entries": [[0, 0, {"num": "1", "den": "1"}]],
                            }
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="homological.koszul.module_quotient.compute",
        title="Compute the degree-zero Koszul quotient module",
        description=(
            "Construct M/(f_1,...,f_r)M exactly as a based module over the same "
            "finite commutative algebra, together with a canonical relation-space "
            "basis, quotient representatives, and the source projection. The empty "
            "sequence returns M; a unit-generated relation space returns the zero module."
        ),
        request_type=ModuleKoszulRequest,
        result_type=ModuleQuotientValue,
        run=module_koszul_quotient,
        tags=("koszul", "module", "quotient", "exact"),
        examples=(
            OperationExample(
                name="quotient_by_unit",
                description="The unit generates the whole module, so the degree-zero quotient is the zero module.",
                input=_MODULE_EXAMPLE,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
