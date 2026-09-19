"""Relational homomorphism operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.logic.relational_structures._admission import (
    MAX_SEARCH_CANDIDATES,
    MAX_SEARCH_TUPLE_REPLAYS,
)
from jacobian.math.logic.relational_structures._models import (
    EmbeddingSearchRequest,
    EmbeddingSearchResult,
    HomomorphismCheckRequest,
    HomomorphismCheckResult,
    HomomorphismCoreRequest,
    HomomorphismCoreResult,
    HomomorphismCountRequest,
    HomomorphismCountResult,
    HomomorphismSearchRequest,
    HomomorphismSearchResult,
)
from jacobian.math.logic.relational_structures.operations import (
    check_homomorphism,
    compute_core,
    count_homomorphisms,
    search_embedding,
    search_homomorphism,
)
from jacobian.math.logic.relational_structures.values import (
    MAX_RELATIONAL_ARITY,
    MAX_RELATIONAL_CARRIER,
    MAX_RELATIONAL_SYMBOLS,
    MAX_RELATIONAL_TABLE_ROWS,
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
            "Decide injective-homomorphism existence between two finite "
            "relational structures over one shared signature by replaying "
            "every distinct-image carrier map in lexicographic order "
            "(source label 0 varying slowest) through the reused "
            "exhaustive check. FOUND retains the first embedding with its "
            "complete check; EXHAUSTED retains the receipt that every one "
            "of the complete |B|^|A| carrier maps was examined, a proved "
            "negative. Supply 'source' and 'target' as structures with "
            "canonical carriers 0..n-1 over one shared ranked signature "
            "with complete tuple tables where an omitted tuple is exactly "
            "NOT in the relation. The candidate space is admitted up front "
            f"(at most {MAX_SEARCH_CANDIDATES} maps, at most "
            f"{MAX_SEARCH_TUPLE_REPLAYS} joint tuple replays); a larger "
            "space is a typed resource refusal, never a negative."
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
                    "isolate at (0, 1), skipping the constant map that the "
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
