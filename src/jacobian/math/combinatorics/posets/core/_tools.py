"""Exact finite-poset operation declarations."""

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.posets.core._closure_tools import CLOSURE_OPERATIONS
from jacobian.math.combinatorics.posets.core._models import (
    AntichainProfileRequest,
    AntichainProfileResult,
    FinitePosetMaterializationResult,
    FinitePosetRequest,
    IncidenceConvolutionRequest,
    IncidenceConvolutionResult,
    LinearExtensionCountResult,
    LinearExtensionRequest,
    MobiusFunctionRequest,
    MobiusFunctionResult,
    PosetClosureRequest,
    PosetClosureResult,
    PosetRequest,
    PosetWidthResult,
    ZetaTransformRequest,
    ZetaTransformResult,
)
from jacobian.math.combinatorics.posets.core.operations import (
    antichain_profile,
    closure,
    incidence_convolution,
    linear_extension_count,
    materialize_finite_poset,
    mobius_function,
    width,
    zeta_transform,
)


def _materialize_request(
    request: FinitePosetRequest,
) -> FinitePosetMaterializationResult:
    return FinitePosetMaterializationResult(
        poset=materialize_finite_poset(
            request.elements,
            request.relation,
            request.interpretation,
            request.reflexive_pairs,
        )
    )


def _width_request(request: PosetRequest) -> PosetWidthResult:
    return width(request.poset)


def _linear_extensions_request(
    request: LinearExtensionRequest,
) -> LinearExtensionCountResult:
    return linear_extension_count(request.poset)


def _mobius_request(request: MobiusFunctionRequest) -> MobiusFunctionResult:
    return mobius_function(request.poset, request.scope, request.intervals)


def _closure_request(request: PosetClosureRequest) -> PosetClosureResult:
    return closure(request.poset, request.subset, request.closure_type)


def _zeta_transform_request(request: ZetaTransformRequest) -> ZetaTransformResult:
    return zeta_transform(request.poset, request.function_values)


def _incidence_convolution_request(
    request: IncidenceConvolutionRequest,
) -> IncidenceConvolutionResult:
    return incidence_convolution(request.poset, request.first, request.second)


def _antichain_profile_request(
    request: AntichainProfileRequest,
) -> AntichainProfileResult:
    return antichain_profile(request.poset)


_DIAMOND: dict[str, Any] = {
    "elements": ["0", "a", "b", "1"],
    "relation": [
        {"lower": "0", "upper": "a"},
        {"lower": "0", "upper": "b"},
        {"lower": "a", "upper": "1"},
        {"lower": "b", "upper": "1"},
    ],
    "interpretation": "COVER_EDGES",
    "reflexive_pairs": "FORBIDDEN",
}

_MATERIALIZED_DIAMOND: dict[str, Any] = {
    "elements": ["0", "1", "a", "b"],
    "strict_order_pairs": [
        {"lower": "0", "upper": "1"},
        {"lower": "0", "upper": "a"},
        {"lower": "0", "upper": "b"},
        {"lower": "a", "upper": "1"},
        {"lower": "b", "upper": "1"},
    ],
    "cover_relations": [
        {"lower": "0", "upper": "a"},
        {"lower": "0", "upper": "b"},
        {"lower": "a", "upper": "1"},
        {"lower": "b", "upper": "1"},
    ],
    "incomparable_pairs": [{"left": "a", "right": "b"}],
    "minimal_elements": ["0"],
    "maximal_elements": ["1"],
    "graded": True,
    "ranks": [
        {"element": "0", "rank": 0},
        {"element": "1", "rank": 2},
        {"element": "a", "rank": 1},
        {"element": "b", "rank": 1},
    ],
    "poset_digest": "sha256:55e795bf7924508b0aa0efe1d0cf32371858ff8b122c890d3eba357dfe2a3374",
}


FINITE_POSET_OPERATIONS: MathTools = (
    MathTool(
        operation_id="poset.finite.compute",
        title="Compute a canonical finite poset",
        description=(
            "Validate exact cover edges or a complete comparable relation and "
            "return canonical closure, Hasse reduction, incomparability, extrema, "
            "and ranks exactly when the poset is graded."
        ),
        request_type=FinitePosetRequest,
        result_type=FinitePosetMaterializationResult,
        run=_materialize_request,
        tags=(
            "poset",
            "partial-order",
            "partially-ordered-set",
            "hasse-diagram",
            "transitive-closure",
            "exact",
        ),
        examples=(
            example(
                "diamond",
                "Materialize the four-element diamond from its cover relation.",
                _DIAMOND,
            ),
            example(
                "three_element_chain",
                "Materialize the chain 0<1<2; the relation must be antisymmetric and acyclic.",
                {
                    "elements": ["0", "1", "2"],
                    "relation": [
                        {"lower": "0", "upper": "1"},
                        {"lower": "0", "upper": "2"},
                        {"lower": "1", "upper": "2"},
                    ],
                    "interpretation": "COMPARABLE_PAIRS",
                    "reflexive_pairs": "FORBIDDEN",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="poset.width.compute",
        title="Compute finite-poset width with dual witnesses",
        description=(
            "Return an exact maximum antichain and a same-size minimum chain "
            "partition, with the bipartite matching intermediate."
        ),
        request_type=PosetRequest,
        result_type=PosetWidthResult,
        run=_width_request,
        tags=(
            "poset",
            "partial-order",
            "partially-ordered-set",
            "width",
            "maximum-antichain",
            "minimum-chain-cover",
            "dilworth",
            "exact",
        ),
        examples=(
            example(
                "materialized_diamond",
                "Compute the width of the canonical four-element diamond.",
                {"poset": _MATERIALIZED_DIAMOND},
            ),
        ),
    ),
    MathTool(
        operation_id="poset.linear_extensions.count",
        title="Count linear extensions of a bounded finite poset",
        description=("Count every linear extension of a bounded finite poset exactly."),
        request_type=LinearExtensionRequest,
        result_type=LinearExtensionCountResult,
        run=_linear_extensions_request,
        tags=(
            "poset",
            "linear-extension",
            "exact-count",
            "order-ideal",
            "dynamic-programming",
        ),
        examples=(
            example(
                "materialized_diamond",
                "Count the linear extensions of the canonical diamond.",
                {"poset": _MATERIALIZED_DIAMOND},
            ),
            example(
                "diamond_complete_mobius_scope",
                "Count the diamond's linear extensions; the poset must have at most 14 elements.",
                {"poset": _MATERIALIZED_DIAMOND},
            ),
        ),
    ),
    MathTool(
        operation_id="poset.mobius_function.compute",
        title="Compute finite-poset Möbius values",
        description=(
            "Return exact incidence-algebra Möbius values for either every "
            "interval or an explicit selected interval scope."
        ),
        request_type=MobiusFunctionRequest,
        result_type=MobiusFunctionResult,
        run=_mobius_request,
        tags=(
            "poset",
            "mobius-function",
            "incidence-algebra",
            "interval",
            "exact",
        ),
        examples=(
            example(
                "materialized_diamond",
                "Compute every Möbius value of the canonical diamond.",
                {"poset": _MATERIALIZED_DIAMOND},
            ),
            example(
                "diamond_selected_interval",
                "Compute the selected Möbius interval [0,1]; selected endpoints must satisfy lower <= upper in the poset.",
                {
                    "poset": _MATERIALIZED_DIAMOND,
                    "scope": "SELECTED_INTERVALS",
                    "intervals": [
                        {"lower": "0", "upper": "1"},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="poset.closure.compute",
        title="Compute ideal or filter closure of a subset",
        description=(
            "Return the lower (ideal) or upper (filter) closure of a given "
            "subset in a finite poset."
        ),
        request_type=PosetClosureRequest,
        result_type=PosetClosureResult,
        run=_closure_request,
        tags=(
            "poset",
            "ideal",
            "filter",
            "closure",
            "exact",
        ),
        examples=(
            example(
                "diamond_lower_closure",
                "Compute the lower closure of {1} in the diamond poset.",
                {
                    "poset": _MATERIALIZED_DIAMOND,
                    "subset": {"elements": ["1"]},
                    "closure_type": "LOWER",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="poset.zeta_transform.compute",
        title="Compute the zeta transform of a function on a poset",
        description=(
            "Apply the incidence-algebra zeta transform to a function "
            "defined on intervals of a finite poset."
        ),
        request_type=ZetaTransformRequest,
        result_type=ZetaTransformResult,
        run=_zeta_transform_request,
        tags=(
            "poset",
            "zeta-transform",
            "incidence-algebra",
            "exact",
        ),
        examples=(
            example(
                "diamond_zeta",
                "Compute the zeta transform of a constant function on the diamond.",
                {
                    "poset": _MATERIALIZED_DIAMOND,
                    "function_values": [
                        {"lower": "0", "upper": "0", "value": 1},
                        {"lower": "a", "upper": "a", "value": 1},
                        {"lower": "b", "upper": "b", "value": 1},
                        {"lower": "1", "upper": "1", "value": 1},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="poset.incidence_convolution.compute",
        title="Convolve two incidence-algebra functions on a poset",
        description=(
            "Compute the incidence-algebra convolution of two functions "
            "defined on intervals of a finite poset."
        ),
        request_type=IncidenceConvolutionRequest,
        result_type=IncidenceConvolutionResult,
        run=_incidence_convolution_request,
        tags=(
            "poset",
            "incidence-algebra",
            "convolution",
            "exact",
        ),
        examples=(
            example(
                "diamond_convolution",
                "Convolve the zeta function with itself on the diamond.",
                {
                    "poset": _MATERIALIZED_DIAMOND,
                    "first": [
                        {"lower": "0", "upper": "0", "value": 1},
                        {"lower": "0", "upper": "a", "value": 1},
                        {"lower": "0", "upper": "b", "value": 1},
                        {"lower": "0", "upper": "1", "value": 1},
                        {"lower": "a", "upper": "a", "value": 1},
                        {"lower": "a", "upper": "1", "value": 1},
                        {"lower": "b", "upper": "b", "value": 1},
                        {"lower": "b", "upper": "1", "value": 1},
                        {"lower": "1", "upper": "1", "value": 1},
                    ],
                    "second": [
                        {"lower": "0", "upper": "0", "value": 1},
                        {"lower": "0", "upper": "a", "value": 1},
                        {"lower": "0", "upper": "b", "value": 1},
                        {"lower": "0", "upper": "1", "value": 1},
                        {"lower": "a", "upper": "a", "value": 1},
                        {"lower": "a", "upper": "1", "value": 1},
                        {"lower": "b", "upper": "b", "value": 1},
                        {"lower": "b", "upper": "1", "value": 1},
                        {"lower": "1", "upper": "1", "value": 1},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="poset.antichain_profile.compute",
        title="Compute the antichain profile of a finite poset",
        description=(
            "Return the maximum antichain size, total antichain count, and "
            "all maximum antichains by enumerating the at most 16,384 subsets "
            "of a poset with at most 14 elements."
        ),
        request_type=AntichainProfileRequest,
        result_type=AntichainProfileResult,
        run=_antichain_profile_request,
        tags=(
            "poset",
            "antichain",
            "profile",
            "exact",
        ),
        examples=(
            example(
                "materialized_diamond",
                "Compute the antichain profile of the canonical diamond.",
                {"poset": _MATERIALIZED_DIAMOND},
            ),
        ),
    ),
)

TOOLS: MathTools = (
    *FINITE_POSET_OPERATIONS,
    *CLOSURE_OPERATIONS,
)

__all__ = ["TOOLS"]
