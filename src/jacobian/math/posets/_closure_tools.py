"""Poset closure, dual, and subposet operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.posets._closure_models import (
    DualPosetRequest,
    DualPosetResult,
    InducedSubposetRequest,
    InducedSubposetResult,
    LowerClosureRequest,
    LowerClosureResult,
    UpperClosureRequest,
    UpperClosureResult,
)
from jacobian.math.posets._closure_operations import (
    dual_poset,
    induced_subposet,
    lower_closure,
    upper_closure,
)

CLOSURE_OPERATIONS: MathTools = (
    MathTool(
        operation_id="poset.lower_closure.compute",
        version="2",
        title="Compute lower closure of a poset subset",
        description=(
            "Compute the lower closure \u2193S = {x : x \u2264 s for some s in S} "
        ),
        request_type=LowerClosureRequest,
        result_type=LowerClosureResult,
        run=lower_closure,
        tags=("poset", "order-ideal", "closure"),
        examples=(
            example(
                "lower_closure_chain",
                "Compute the lower closure of the top element of a 3-chain.",
                {
                    "poset": {
                        "elements": ["a", "b", "c"],
                        "strict_order_pairs": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "a", "upper": "c"},
                            {"lower": "b", "upper": "c"},
                        ],
                        "cover_relations": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "b", "upper": "c"},
                        ],
                        "incomparable_pairs": [],
                        "minimal_elements": ["a"],
                        "maximal_elements": ["c"],
                        "graded": True,
                        "ranks": [
                            {"element": "a", "rank": 0},
                            {"element": "b", "rank": 1},
                            {"element": "c", "rank": 2},
                        ],
                        "poset_digest": "sha256:af53d0ee18b87ccc0b5987c289c88d2139afb880eb6756a44f6bb002ea785a70",
                    },
                    "subset": ["c"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="poset.upper_closure.compute",
        version="2",
        title="Compute upper closure of a poset subset",
        description=(
            "Compute the upper closure \u2191S = {x : s \u2264 x for some s in S} "
        ),
        request_type=UpperClosureRequest,
        result_type=UpperClosureResult,
        run=upper_closure,
        tags=("poset", "order-filter", "closure"),
        examples=(
            example(
                "upper_closure_chain_bottom",
                "Compute the upper closure of the bottom element of a 3-chain.",
                {
                    "poset": {
                        "elements": ["a", "b", "c"],
                        "strict_order_pairs": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "a", "upper": "c"},
                            {"lower": "b", "upper": "c"},
                        ],
                        "cover_relations": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "b", "upper": "c"},
                        ],
                        "incomparable_pairs": [],
                        "minimal_elements": ["a"],
                        "maximal_elements": ["c"],
                        "graded": True,
                        "ranks": [
                            {"element": "a", "rank": 0},
                            {"element": "b", "rank": 1},
                            {"element": "c", "rank": 2},
                        ],
                        "poset_digest": "sha256:af53d0ee18b87ccc0b5987c289c88d2139afb880eb6756a44f6bb002ea785a70",
                    },
                    "subset": ["a"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="poset.dual.compute",
        version="2",
        title="Compute the dual of a finite poset",
        description=(
            "Return the same element domain with the order reversed, plus the "
            "identity transport map."
        ),
        request_type=DualPosetRequest,
        result_type=DualPosetResult,
        run=dual_poset,
        tags=("poset", "dual", "order-reversal"),
        examples=(
            example(
                "dual_chain",
                "Compute the dual of a 3-element chain a < b < c.",
                {
                    "poset": {
                        "elements": ["a", "b", "c"],
                        "strict_order_pairs": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "a", "upper": "c"},
                            {"lower": "b", "upper": "c"},
                        ],
                        "cover_relations": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "b", "upper": "c"},
                        ],
                        "incomparable_pairs": [],
                        "minimal_elements": ["a"],
                        "maximal_elements": ["c"],
                        "graded": True,
                        "ranks": [
                            {"element": "a", "rank": 0},
                            {"element": "b", "rank": 1},
                            {"element": "c", "rank": 2},
                        ],
                        "poset_digest": "sha256:af53d0ee18b87ccc0b5987c289c88d2139afb880eb6756a44f6bb002ea785a70",
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="poset.induced_subposet.compute",
        version="2",
        title="Compute the induced subposet on a subset",
        description=(
            "Return the restricted order, cover relation, and old/new element map "
            "for the subposet induced by the supplied element subset."
        ),
        request_type=InducedSubposetRequest,
        result_type=InducedSubposetResult,
        run=induced_subposet,
        tags=("poset", "subposet", "restriction"),
        examples=(
            example(
                "induced_subposet_chain",
                "Compute the subposet induced on {a, b} of a 3-chain.",
                {
                    "poset": {
                        "elements": ["a", "b", "c"],
                        "strict_order_pairs": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "a", "upper": "c"},
                            {"lower": "b", "upper": "c"},
                        ],
                        "cover_relations": [
                            {"lower": "a", "upper": "b"},
                            {"lower": "b", "upper": "c"},
                        ],
                        "incomparable_pairs": [],
                        "minimal_elements": ["a"],
                        "maximal_elements": ["c"],
                        "graded": True,
                        "ranks": [
                            {"element": "a", "rank": 0},
                            {"element": "b", "rank": 1},
                            {"element": "c", "rank": 2},
                        ],
                        "poset_digest": "sha256:af53d0ee18b87ccc0b5987c289c88d2139afb880eb6756a44f6bb002ea785a70",
                    },
                    "subset": ["a", "b"],
                },
            ),
        ),
    ),
)

__all__ = ["CLOSURE_OPERATIONS"]
