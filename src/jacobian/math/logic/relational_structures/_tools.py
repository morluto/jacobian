"""Relational homomorphism operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.logic.relational_structures._models import (
    HomomorphismCheckRequest,
    HomomorphismCheckResult,
)
from jacobian.math.logic.relational_structures.operations import check_homomorphism
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
            "this map only: homomorphism search, embeddings, cores, CSP "
            "instances, and polymorphisms are deferred."
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
)


__all__ = ["TOOLS"]
