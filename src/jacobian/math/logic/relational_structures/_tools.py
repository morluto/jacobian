"""Relational homomorphism operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.logic.relational_structures._admission import (
    MAX_EMBEDDING_REFLECTION_CELLS,
    MAX_HOMOMORPHISM_ENUMERATION_MAP_LABELS,
    MAX_POLYMORPHISM_COORDINATE_WORK,
    MAX_POLYMORPHISM_RELATION_COMBINATIONS,
    MAX_SEARCH_CANDIDATES,
    MAX_SEARCH_TUPLE_REPLAYS,
)
from jacobian.math.logic.relational_structures._models import (
    CspAssignmentProfile,
    CspAssignmentRequest,
    CspSolutions,
    EmbeddingSearchRequest,
    EmbeddingSearchResult,
    FiniteCspInstance,
    HomomorphismCheckRequest,
    HomomorphismCheckResult,
    HomomorphismCoreRequest,
    HomomorphismCoreResult,
    HomomorphismCountRequest,
    HomomorphismCountResult,
    HomomorphismEnumerationRequest,
    HomomorphismEnumerationResult,
    HomomorphismSearchRequest,
    HomomorphismSearchResult,
    InducedSubstructureRequest,
    InducedSubstructureResult,
    RelationalPolymorphismCheckResult,
    RelationalPolymorphismRequest,
    RelationalProductRequest,
    RelationalProductResult,
    RelationalQuotient,
    RelationalQuotientRequest,
    RelationalReductRequest,
    RelationalReductResult,
)
from jacobian.math.logic.relational_structures.operations import (
    check_homomorphism,
    check_polymorphism,
    compute_core,
    count_homomorphisms,
    csp_instance_to_source_structure,
    direct_product_structure,
    enumerate_csp_solutions,
    enumerate_homomorphisms,
    induced_substructure,
    profile_csp_assignment,
    quotient_structure,
    reduct_structure,
    search_embedding,
    search_homomorphism,
)
from jacobian.math.logic.relational_structures.values import (
    MAX_RELATIONAL_ARITY,
    MAX_RELATIONAL_CARRIER,
    MAX_RELATIONAL_OPERATION_TABLE_CELLS,
    MAX_RELATIONAL_POLYMORPHISM_ARITY,
    MAX_RELATIONAL_SYMBOLS,
    MAX_RELATIONAL_TABLE_ROWS,
    FiniteRelationalStructure,
)


def _homomorphism_check(
    request: HomomorphismCheckRequest,
) -> HomomorphismCheckResult:
    """Project a wire request into the canonical exhaustive replay."""

    return check_homomorphism(request.source, request.target, request.carrier_map)


def _homomorphism_search(
    request: HomomorphismSearchRequest,
) -> HomomorphismSearchResult:
    """Project a wire request into the canonical exhaustive search."""

    return search_homomorphism(request.source, request.target)


def _homomorphism_count(
    request: HomomorphismCountRequest,
) -> HomomorphismCountResult:
    """Project a wire request into the canonical exhaustive count."""

    return count_homomorphisms(request.source, request.target)


def _homomorphism_enumeration(
    request: HomomorphismEnumerationRequest,
) -> HomomorphismEnumerationResult:
    return enumerate_homomorphisms(request.source, request.target)


def _homomorphism_core(
    request: HomomorphismCoreRequest,
) -> HomomorphismCoreResult:
    """Project a wire request into the canonical core iteration."""

    return compute_core(request.source)


def _embedding_search(
    request: EmbeddingSearchRequest,
) -> EmbeddingSearchResult:
    """Project a wire request into the canonical embedding search."""

    return search_embedding(request.source, request.target)


def _csp_instance_source(
    request: FiniteCspInstance,
) -> FiniteRelationalStructure:
    """Project the canonical CSP conversion through the native kernel."""

    return csp_instance_to_source_structure(request)


def _csp_assignment_profile(
    request: CspAssignmentRequest,
) -> CspAssignmentProfile:
    return profile_csp_assignment(request.instance, request.assignment)


def _csp_solutions(request: FiniteCspInstance) -> CspSolutions:
    return enumerate_csp_solutions(request)


def _induced_substructure(
    request: InducedSubstructureRequest,
) -> InducedSubstructureResult:
    return induced_substructure(request.source, request.inclusion)


def _relational_reduct(request: RelationalReductRequest) -> RelationalReductResult:
    return reduct_structure(request.source, request.symbol_ids)


def _quotient(request: RelationalQuotientRequest) -> RelationalQuotient:
    return quotient_structure(request.source, request.classes)


def _direct_product(request: RelationalProductRequest) -> RelationalProductResult:
    return direct_product_structure(request.left, request.right)


def _polymorphism_check(
    request: RelationalPolymorphismRequest,
) -> RelationalPolymorphismCheckResult:
    return check_polymorphism(request.source, request.arity, request.operation_table)


_DIRECTED_EDGE = {"symbol_id": "E", "arity": 2}

_THREE_CYCLE = {
    "carrier_size": 3,
    "signature": [_DIRECTED_EDGE],
    "relation_tables": [[[0, 1], [1, 2], [2, 0]]],
}

_DIRECTED_TRIANGLE = {
    "carrier_size": 3,
    "signature": [_DIRECTED_EDGE],
    "relation_tables": [
        [
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 2],
            [2, 0],
            [2, 1],
        ]
    ],
}

_TRIANGLE_INTO_CYCLE_EXAMPLE = {
    "source": _THREE_CYCLE,
    "target": _DIRECTED_TRIANGLE,
    "carrier_map": [0, 1, 2],
}

_DIRECTED_EDGE_STRUCTURE = {
    "carrier_size": 2,
    "signature": [_DIRECTED_EDGE],
    "relation_tables": [[[0, 1]]],
}

_EDGE_INTO_CYCLE_EXAMPLE = {
    "source": _DIRECTED_EDGE_STRUCTURE,
    "target": _THREE_CYCLE,
}


TOOLS: MathTools = (
    MathTool(
        operation_id="relational.structure.direct_product.compute",
        title="Form a direct product of finite relational structures",
        description=(
            "Form the direct product of two finite structures with identical "
            "ranked signatures. Pair (i,j) receives canonical label "
            "i*|B|+j; each relation contains exactly the coordinatewise "
            "pairs from the two factor relations. The result retains both "
            "factors and the two projection maps. Carrier, relation-row, and "
            "coordinate-work bounds are checked before Cartesian relation "
            "expansion; the admitted visits bound the product and its "
            "projections."
        ),
        request_type=RelationalProductRequest,
        result_type=RelationalProductResult,
        run=_direct_product,
        tags=("relational-structures", "direct-product", "homomorphism", "exact"),
        discovery_terms=(
            "direct product of finite relational structures",
            "relational structure Cartesian product",
            "product structure projections",
            "power of a finite relational structure",
        ),
        examples=(
            OperationExample(
                name="product_of_two_directed_edges",
                description=(
                    "Form the edge relation by coordinatewise pairing; both "
                    "factors must have identical ordered ranked signatures."
                ),
                input={
                    "left": _DIRECTED_EDGE_STRUCTURE,
                    "right": _DIRECTED_EDGE_STRUCTURE,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="relational.induced_substructure.compute",
        title="Take an induced finite relational substructure",
        description=(
            "Select an ordered subset of one finite relational carrier and "
            "return every restricted relation table, the exact source, and "
            "the inclusion map from canonical induced labels to source labels. "
            "The selected order defines the induced carrier axis; nullary "
            "relations retain their exact truth values. Row transport work "
            "and result shape are admitted before relation expansion."
        ),
        request_type=InducedSubstructureRequest,
        result_type=InducedSubstructureResult,
        run=_induced_substructure,
        tags=("relational-structures", "induced-substructure", "exact"),
        discovery_terms=(
            "induced substructure",
            "finite relational carrier restriction",
            "relational substructure with inclusion map",
        ),
        examples=(
            OperationExample(
                name="ordered_induced_substructure_of_cycle",
                description=(
                    "Select source labels (2, 0), preserving that order; the "
                    "cycle edge from source 2 to source 0 becomes (0, 1)."
                ),
                input={"source": _THREE_CYCLE, "inclusion": [2, 0]},
            ),
        ),
    ),
    MathTool(
        operation_id="relational_structure.reduct.compute",
        title="Take a finite relational structure reduct",
        description=(
            "Retain an explicit subset of a finite relational signature in "
            "source-signature order, preserving the carrier and selected "
            "complete relation tables. The result includes the exact reduct and the map from "
            "reduct symbol positions to source signature positions. Selected "
            "row and coordinate work is admitted before construction."
        ),
        request_type=RelationalReductRequest,
        result_type=RelationalReductResult,
        run=_relational_reduct,
        tags=("relational-structures", "reduct", "signature", "exact"),
        discovery_terms=(
            "finite relational structure reduct",
            "restrict relational signature",
        ),
        examples=(
            OperationExample(
                name="retain_one_relation_from_two_symbol_structure",
                description=(
                    "Keep unary P from a structure whose signature is ordered "
                    "as E, P; the reduct-to-source symbol map is [1]."
                ),
                input={
                    "source": {
                        "carrier_size": 2,
                        "signature": [
                            {"symbol_id": "E", "arity": 2},
                            {"symbol_id": "P", "arity": 1},
                        ],
                        "relation_tables": [[[0, 1]], [[1]]],
                    },
                    "symbol_ids": ["P"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="relational.quotient.compute",
        title="Form a finite relational structure quotient",
        description=(
            "Form the quotient by a supplied carrier partition, accepting it "
            "exactly when every relation table is a union of complete "
            "Cartesian fibers. Partial fibers are rejected; the exact result "
            "retains the signature and canonical surjective quotient map."
        ),
        request_type=RelationalQuotientRequest,
        result_type=RelationalQuotient,
        run=_quotient,
        tags=("relational-structures", "quotient", "exact"),
        examples=(
            OperationExample(
                name="complete_bipartite_relation_quotient",
                description="Collapse the two sides of K2,2 to one relation tuple.",
                input={
                    "source": {
                        "carrier_size": 4,
                        "signature": [{"symbol_id": "E", "arity": 2}],
                        "relation_tables": [[[0, 2], [0, 3], [1, 2], [1, 3]]],
                    },
                    "classes": [7, 7, 9, 9],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="relational.polymorphism.check",
        title="Check a finite relational polymorphism",
        description=(
            "Check one complete caller-supplied operation table f:A^m→A "
            "against every basic relation of the exact finite source structure. "
            "For each relation R and every ordered m-tuple of rows from R, "
            "apply f coordinatewise and require the output row to lie in R. "
            "A failure returns the first exact relation/input/output witness; "
            "a success returns a structure-bound RelationalPolymorphism. "
            "The operation table is distinct from a homomorphism carrier map: "
            "it has |A|^m entries, in lexicographic tuple order. The admitted "
            "limits are arity at most "
            f"{MAX_RELATIONAL_POLYMORPHISM_ARITY}, table size at most "
            f"{MAX_RELATIONAL_OPERATION_TABLE_CELLS}, complete relation products "
            f"at most {MAX_POLYMORPHISM_RELATION_COMBINATIONS} combinations, and "
            f"coordinate work at most {MAX_POLYMORPHISM_COORDINATE_WORK}. "
            "Admission is computed before relation products are enumerated."
        ),
        request_type=RelationalPolymorphismRequest,
        result_type=RelationalPolymorphismCheckResult,
        run=_polymorphism_check,
        tags=("relational-structures", "polymorphism", "csp", "exact"),
        discovery_terms=(
            "finite relational polymorphism",
            "operation preserves relations",
            "CSP polymorphism check",
            "coordinatewise relation preservation",
        ),
        examples=(
            OperationExample(
                name="binary_first_projection_preserves_edge",
                description=(
                    "The binary first projection preserves the singleton "
                    "directed-edge relation."
                ),
                input={
                    "source": {
                        "carrier_size": 2,
                        "signature": [{"symbol_id": "E", "arity": 2}],
                        "relation_tables": [[[0, 1]]],
                    },
                    "arity": 2,
                    "operation_table": [0, 0, 1, 1],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="csp.instance.to_source_structure.compute",
        title="Convert a finite CSP instance to its source structure",
        description=(
            "Return the canonical finite relational structure on the instance "
            "variables whose relation tables are exactly the constraint scopes. "
            "Maps from this structure to the retained template are precisely "
            "the satisfying assignments. Repeated variables in scopes retain "
            "their positions; distinct constraint occurrences remain distinct "
            "on input, while identical symbol/scope rows deduplicate in the "
            "ordinary relation table. Variable count, template carrier, symbols, "
            "arity, constraint occurrences, aggregate scope entries, and tuple "
            "tables are bounded by the finite relational contracts."
        ),
        request_type=FiniteCspInstance,
        result_type=FiniteRelationalStructure,
        run=_csp_instance_source,
        tags=("csp", "relational-structures", "exact"),
        discovery_terms=(
            "finite CSP instance",
            "constraint satisfaction source structure",
            "constraints to relational structure",
            "canonical database",
        ),
        examples=(
            OperationExample(
                name="repeated_variable_scope",
                description=(
                    "The constraint E(x,x) becomes the source relation row "
                    "(0,0), retaining the repeated variable position."
                ),
                input={
                    "template": {
                        "carrier_size": 2,
                        "signature": [{"symbol_id": "E", "arity": 2}],
                        "relation_tables": [[[0, 0], [1, 1]]],
                    },
                    "variable_count": 1,
                    "constraints": [
                        {"constraint_id": "c0", "symbol_id": "E", "scope": [0, 0]}
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="csp.assignment.profile.compute",
        title="Evaluate a finite CSP assignment",
        description=(
            "Evaluate one complete assignment at every named constraint occurrence. "
            "The result retains each assigned target tuple and the first violated "
            "occurrence, if any. A failing assignment is not evidence that the "
            "instance is unsatisfiable; use bounded complete search for that claim."
        ),
        request_type=CspAssignmentRequest,
        result_type=CspAssignmentProfile,
        run=_csp_assignment_profile,
        tags=("csp", "relational-structures", "exact"),
        discovery_terms=(
            "check a CSP assignment",
            "evaluate constraint assignment",
            "CSP solution profile",
            "first violated constraint",
        ),
        examples=(
            OperationExample(
                name="first_failed_constraint",
                description="Return the first named constraint whose assigned tuple is absent.",
                input={
                    "instance": {
                        "template": {
                            "carrier_size": 2,
                            "signature": [{"symbol_id": "E", "arity": 2}],
                            "relation_tables": [[[0, 1]]],
                        },
                        "variable_count": 2,
                        "constraints": [
                            {"constraint_id": "edge", "symbol_id": "E", "scope": [0, 1]}
                        ],
                    },
                    "assignment": [1, 0],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="csp.solutions.enumerate.compute",
        title="Enumerate every solution of a finite CSP instance",
        description=(
            "Return the complete lexicographically ordered family of satisfying "
            "assignments on the instance variable axis, retaining the original "
            "instance and its named constraint occurrences. This is exhaustive: "
            "the carrier-map space and tuple-replay work are admitted before "
            "search, and an over-budget request is rejected rather than truncated. "
            "The assignments agree with homomorphisms from the canonical CSP "
            "source structure into its template, including repeated-variable "
            "scopes, duplicate occurrences, nullary relations, and empty axes."
        ),
        request_type=FiniteCspInstance,
        result_type=CspSolutions,
        run=_csp_solutions,
        tags=("csp", "solutions", "relational-structures", "exact"),
        discovery_terms=(
            "enumerate all CSP solutions",
            "complete satisfying assignments",
            "finite constraint satisfaction solution relation",
            "CSP solutions as homomorphisms",
        ),
        examples=(
            OperationExample(
                name="two_color_one_edge",
                description="Enumerate the two assignments satisfying one disequality constraint.",
                input={
                    "template": {
                        "carrier_size": 2,
                        "signature": [{"symbol_id": "E", "arity": 2}],
                        "relation_tables": [[[0, 1], [1, 0]]],
                    },
                    "variable_count": 2,
                    "constraints": [
                        {"constraint_id": "edge", "symbol_id": "E", "scope": [0, 1]}
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="relational.homomorphism.check",
        title="Check a candidate homomorphism between finite relational structures",
        description=(
            "Decide whether one candidate carrier map h between two finite "
            "relational structures over one shared signature is a "
            "homomorphism by exhaustively replaying the defining invariant: "
            "for every relation symbol R and every tuple t in the complete "
            "source table R^A, the coordinatewise image h(t) must lie in the "
            "complete target table R^B. Returns the closed status "
            "HOMOMORPHISM or NOT_HOMOMORPHISM with the first violating "
            "(symbol, source tuple, image tuple) witness in deterministic "
            "order, plus complete per-symbol transport counts. Supply "
            "'source' and 'target' as structures with canonical carriers "
            f"0..n-1 (at most {MAX_RELATIONAL_CARRIER} elements), a ranked "
            f"signature (at most {MAX_RELATIONAL_SYMBOLS} symbols, arity at "
            f"most {MAX_RELATIONAL_ARITY}), and complete tuple tables (at "
            f"most {MAX_RELATIONAL_TABLE_ROWS} rows each) where an omitted "
            "tuple is exactly NOT in the relation, plus 'carrier_map' as one "
            "target label per source label. A signature mismatch or a "
            "malformed map is typed-rejected; a total map violating one "
            "relation is the mathematical negative. NOT_HOMOMORPHISM concerns "
            "this map only: CSP "
            "instances and polymorphisms are deferred."
        ),
        request_type=HomomorphismCheckRequest,
        result_type=HomomorphismCheckResult,
        run=_homomorphism_check,
        tags=(
            "relational-structures",
            "homomorphism",
            "finite-model-theory",
            "exact",
        ),
        discovery_terms=(
            "relational homomorphism",
            "finite relational structure",
            "homomorphism check",
            "relation preserving map",
            "structure homomorphism witness",
            "constraint satisfaction map check",
        ),
        examples=(
            OperationExample(
                name="directed_three_cycle_into_triangle",
                description=(
                    "The identity carrier map sends a directed 3-cycle into "
                    "the complete directed triangle; every source edge "
                    "transports, so the status is HOMOMORPHISM."
                ),
                input=_TRIANGLE_INTO_CYCLE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="relational.homomorphism.search.compute",
        title="Search two finite relational structures for a homomorphism",
        description=(
            "Decide homomorphism existence between two finite relational "
            "structures over one shared signature by replaying every "
            "carrier map in lexicographic order (source label 0 varying "
            "slowest) through the reused exhaustive check. FOUND retains "
            "the first transporting map with its complete check; EXHAUSTED "
            "retains the receipt that every one of the complete |B|^|A| "
            "carrier maps was examined, a proved negative. Supply 'source' "
            "and 'target' as structures with canonical carriers 0..n-1 "
            f"(at most {MAX_RELATIONAL_CARRIER} elements) over one shared "
            f"ranked signature (at most {MAX_RELATIONAL_SYMBOLS} symbols) "
            "with complete tuple tables where an omitted tuple is exactly "
            "NOT in the relation. The candidate space is admitted up front "
            f"(at most {MAX_SEARCH_CANDIDATES} maps, at most "
            f"{MAX_SEARCH_TUPLE_REPLAYS} joint tuple replays); a larger "
            "space is a typed resource refusal, never a negative."
        ),
        request_type=HomomorphismSearchRequest,
        result_type=HomomorphismSearchResult,
        run=_homomorphism_search,
        tags=(
            "relational-structures",
            "homomorphism",
            "finite-model-theory",
            "csp",
            "exact",
        ),
        discovery_terms=(
            "homomorphism search",
            "homomorphism existence",
            "constraint satisfaction search",
            "first homomorphism",
            "exhaustive map search",
            "no homomorphism proof",
        ),
        examples=(
            OperationExample(
                name="directed_edge_into_three_cycle",
                description=(
                    "The directed edge maps into the directed 3-cycle at "
                    "(0, 1), the first transporting map in lexicographic "
                    "order; both structures share the single binary edge "
                    "symbol, so the status is FOUND."
                ),
                input=_EDGE_INTO_CYCLE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="relational.homomorphism.enumerate.compute",
        title="Enumerate all homomorphisms between finite relational structures",
        description=(
            "Return the complete lexicographically ordered list of all total "
            "carrier maps preserving every relation between two finite "
            "structures over one shared signature. The entire |B|^|A| search "
            "and a conservative retained-map envelope (at most "
            f"{MAX_HOMOMORPHISM_ENUMERATION_MAP_LABELS} carrier map labels) "
            "are admitted "
            "before map enumeration; larger requests receive a typed "
            "resource refusal."
        ),
        request_type=HomomorphismEnumerationRequest,
        result_type=HomomorphismEnumerationResult,
        run=_homomorphism_enumeration,
        tags=("relational-structures", "homomorphism", "finite-model-theory", "exact"),
        discovery_terms=(
            "enumerate all homomorphisms",
            "finite relational homomorphism enumeration",
            "all relation-preserving carrier maps",
        ),
        examples=(
            OperationExample(
                name="all_directed_edges_into_three_cycle",
                description="The two-vertex directed edge has three maps into the directed 3-cycle, one for each cycle edge.",
                input=_EDGE_INTO_CYCLE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="relational.homomorphism.count.compute",
        title="Count every homomorphism between finite relational structures",
        description=(
            "Count every homomorphism between two finite relational "
            "structures over one shared signature by replaying the complete "
            "admitted carrier-map space without early stopping through the "
            "reused exhaustive check. The count is exact and complete, "
            "unlike the first-witness search. Supply 'source' and 'target' "
            "as structures with canonical carriers 0..n-1 over one shared "
            "ranked signature with complete tuple tables where an omitted "
            "tuple is exactly NOT in the relation. The candidate space is "
            f"admitted up front (at most {MAX_SEARCH_CANDIDATES} maps, at "
            f"most {MAX_SEARCH_TUPLE_REPLAYS} joint tuple replays); a larger "
            "space is a typed resource refusal, never a truncated count."
        ),
        request_type=HomomorphismCountRequest,
        result_type=HomomorphismCountResult,
        run=_homomorphism_count,
        tags=(
            "relational-structures",
            "homomorphism",
            "finite-model-theory",
            "csp",
            "exact",
        ),
        discovery_terms=(
            "homomorphism count",
            "number of homomorphisms",
            "constraint satisfaction count",
            "homomorphism enumeration size",
        ),
        examples=(
            OperationExample(
                name="directed_edge_into_three_cycle_count",
                description=(
                    "The directed edge maps into the directed 3-cycle in "
                    "exactly 3 ways, one per cycle edge; both structures "
                    "share the single binary edge symbol, so the count is 3."
                ),
                input=_EDGE_INTO_CYCLE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="relational.core.compute",
        title="Compute the minimal retract of a finite relational structure",
        description=(
            "Compute the core of one finite relational structure by "
            "restricting along the first non-surjective endomorphism in "
            "lexicographic order until no smaller image remains, retaining "
            "the induced core substructure with canonical labels, the "
            "increasing inclusion back into source labels, and the "
            "idempotent retraction. Every level scan is decided by the "
            "reused exhaustive check and the composed retraction is "
            "replayed once. Supply 'source' with canonical carrier 0..n-1 "
            "over a ranked signature with complete tuple tables where an "
            "omitted tuple is exactly NOT in the relation. The summed "
            "level-by-level replay work is admitted up front against the "
            f"{MAX_SEARCH_TUPLE_REPLAYS}-replay envelope; a larger space is "
            "a typed resource refusal, never a partial core."
        ),
        request_type=HomomorphismCoreRequest,
        result_type=HomomorphismCoreResult,
        run=_homomorphism_core,
        tags=(
            "relational-structures",
            "homomorphism",
            "finite-model-theory",
            "csp",
            "exact",
        ),
        discovery_terms=(
            "relational core",
            "minimal retract",
            "retraction",
            "idempotent endomorphism",
            "core computation",
        ),
        examples=(
            OperationExample(
                name="v_shape_core_is_single_edge",
                description=(
                    "The V shape with edges 0→1 and 0→2 retracts onto its "
                    "first edge at (0, 1, 1); the shared binary edge symbol "
                    "keeps the core a single directed edge."
                ),
                input={
                    "source": {
                        "carrier_size": 3,
                        "signature": [_DIRECTED_EDGE],
                        "relation_tables": [[[0, 1], [0, 2]]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="relational.embedding.search.compute",
        title="Search two finite relational structures for an embedding",
        description=(
            "Decide induced-substructure embedding existence between two finite "
            "relational structures over one shared signature by replaying "
            "every distinct-image carrier map in lexicographic order "
            "(source label 0 varying slowest) through the complete "
            "positive-and-negative relation check. FOUND retains the first "
            "embedding with its complete check; EXHAUSTED retains the receipt "
            "that every one of the complete |B|^|A| carrier maps was examined, "
            "a proved negative. The candidate space is admitted up front "
            f"(at most {MAX_SEARCH_CANDIDATES} maps and "
            f"{MAX_SEARCH_TUPLE_REPLAYS} replay units, including at most "
            f"{MAX_EMBEDDING_REFLECTION_CELLS} Cartesian reflection cells per candidate); "
            "a larger space is a typed resource refusal, never a negative."
        ),
        request_type=EmbeddingSearchRequest,
        result_type=EmbeddingSearchResult,
        run=_embedding_search,
        tags=(
            "relational-structures",
            "homomorphism",
            "finite-model-theory",
            "csp",
            "exact",
        ),
        discovery_terms=(
            "relational embedding",
            "injective homomorphism",
            "embedding existence",
            "distinct image map",
            "induced substructure embedding",
        ),
        examples=(
            OperationExample(
                name="bare_pair_embeds_past_constant",
                description=(
                    "Two bare vertices embed into the directed edge plus "
                    "isolate at (0, 2), whose induced image has no edge; this "
                    "skips the constant map that the "
                    "plain search returns; both structures share the single "
                    "binary edge symbol, so the status is FOUND."
                ),
                input={
                    "source": {
                        "carrier_size": 2,
                        "signature": [_DIRECTED_EDGE],
                        "relation_tables": [[]],
                    },
                    "target": {
                        "carrier_size": 3,
                        "signature": [_DIRECTED_EDGE],
                        "relation_tables": [[[0, 1]]],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
