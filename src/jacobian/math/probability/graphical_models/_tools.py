"""Graphical model operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.probability.graphical_models._models import (
    BayesNetConstructRequest,
    BayesNetConstructResult,
    BayesNetJointRequest,
    BayesNetJointResult,
    DSeparationRequest,
    DSeparationResult,
    FactorMarginalizeRequest,
    FactorMarginalizeResult,
    FactorMultiplyRequest,
    FactorMultiplyResult,
    JunctionTreeCalibrateRequest,
    JunctionTreeCalibrateResult,
    VariableEliminationTraceRequest,
    VariableEliminationTraceResult,
)
from jacobian.math.probability.graphical_models.operations import (
    bayes_net_joint,
    construct_bayes_net,
    d_separation,
    factor_marginalize,
    factor_multiply,
    junction_tree_calibrate,
    variable_elimination_trace,
)


def _factor_multiply(request: FactorMultiplyRequest) -> FactorMultiplyResult:
    return FactorMultiplyResult._from_kernel(
        request.left, request.right, factor_multiply(request.left, request.right)
    )


def _factor_marginalize(
    request: FactorMarginalizeRequest,
) -> FactorMarginalizeResult:
    return FactorMarginalizeResult._from_kernel(
        request.factor,
        request.variable,
        factor_marginalize(request.factor, request.variable),
    )


def _d_separation(request: DSeparationRequest) -> DSeparationResult:
    query = request.query
    return DSeparationResult._from_kernel(
        request,
        d_separation(
            query.dag.variable_count,
            query.dag.edges,
            query.set_a,
            query.set_b,
            query.set_c,
        ),
    )


def _bayes_net_construct(request: BayesNetConstructRequest) -> BayesNetConstructResult:
    return BayesNetConstructResult._from_kernel(
        construct_bayes_net(
            request.variable_count,
            request.edges,
            tuple(request.domain_sizes),
            request.tables,
        )
    )


def _bayes_net_joint(request: BayesNetJointRequest) -> BayesNetJointResult:
    return BayesNetJointResult._from_kernel(
        request.network, bayes_net_joint(request.network)
    )


def _elimination_trace(
    request: VariableEliminationTraceRequest,
) -> VariableEliminationTraceResult:
    steps, final = variable_elimination_trace(
        request.factors,
        tuple(request.domain_sizes),
        request.elimination_order,
        request.query_variables,
    )
    return VariableEliminationTraceResult._from_kernel(
        factors=request.factors,
        domain_sizes=tuple(request.domain_sizes),
        elimination_order=request.elimination_order,
        query_variables=request.query_variables,
        steps=steps,
        final=final,
    )


def _junction_calibrate(
    request: JunctionTreeCalibrateRequest,
) -> JunctionTreeCalibrateResult:
    cliques, separators, clique_marginals, separator_marginals, partition = (
        junction_tree_calibrate(request.network, request.elimination_order)
    )
    return JunctionTreeCalibrateResult._from_kernel(
        network=request.network,
        elimination_order=request.elimination_order,
        cliques=cliques,
        separators=separators,
        clique_marginals=clique_marginals,
        separator_marginals=separator_marginals,
        partition=partition,
    )


_FACTOR1 = {
    "variables": [0, 1],
    "domain_sizes": [2, 2, 2],
    "table": [
        {"num": "1", "den": "2"},
        {"num": "1", "den": "2"},
        {"num": "1", "den": "3"},
        {"num": "2", "den": "3"},
    ],
}

_FACTOR_SINGLE = {
    "variables": [0],
    "domain_sizes": [2],
    "table": [{"num": "1", "den": "3"}, {"num": "2", "den": "3"}],
}
_CPT_ROOT = {
    "variable": 0,
    "parents": [],
    "domain_sizes": [2, 2],
    "table": [{"num": "3", "den": "5"}, {"num": "2", "den": "5"}],
}
_CPT_CHILD = {
    "variable": 1,
    "parents": [0],
    "domain_sizes": [2, 2],
    "table": [
        {"num": "1", "den": "4"},
        {"num": "1", "den": "2"},
        {"num": "3", "den": "4"},
        {"num": "1", "den": "2"},
    ],
}
_CHAIN_NET = {
    "variable_count": 2,
    "edges": [[0, 1]],
    "domain_sizes": [2, 2],
    "tables": [_CPT_ROOT, _CPT_CHILD],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graphical_model.factor.multiply",
        title="Multiply two factors",
        description="Compute the product of two factors over the union of their variables "
        "using bounded exact rational arithmetic. Table entries must be nonnegative "
        "canonical rationals. Scalar factors use an empty scope.",
        request_type=FactorMultiplyRequest,
        result_type=FactorMultiplyResult,
        run=_factor_multiply,
        tags=("graphical-model", "factor", "exact"),
        examples=(
            OperationExample(
                name="multiply_two_factors",
                description="Multiply a 2-var factor by another 2-var factor.",
                input={
                    "left": _FACTOR1,
                    "right": {
                        "variables": [1],
                        "domain_sizes": [2, 2, 2],
                        "table": [
                            {"num": "1", "den": "2"},
                            {"num": "1", "den": "2"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graphical_model.factor.marginalize",
        title="Marginalize out a variable from a factor",
        description="Sum out a variable from a factor with nonnegative table entries, "
        "producing a factor over the "
        "remaining variables using exact rational arithmetic.",
        request_type=FactorMarginalizeRequest,
        result_type=FactorMarginalizeResult,
        run=_factor_marginalize,
        tags=("graphical-model", "factor", "marginalization", "exact"),
        examples=(
            OperationExample(
                name="marginalize_var_0",
                description="Marginalize out variable 0 from a single-variable factor.",
                input={
                    "factor": _FACTOR_SINGLE,
                    "variable": 0,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graphical_model.d_separation.compute",
        title="Check d-separation in a Bayesian network",
        description="Check whether two sets of variables are d-separated given a "
        "conditioning set in a bounded directed acyclic graph, using ancestral "
        "restriction and moralization.",
        request_type=DSeparationRequest,
        result_type=DSeparationResult,
        run=_d_separation,
        tags=("graphical-model", "d-separation", "exact"),
        examples=(
            OperationExample(
                name="conditioned_chain",
                description="Decide whether endpoints of a three-node chain are separated by its middle node.",
                input={
                    "query": {
                        "dag": {"variables": [0, 1, 2], "edges": [[0, 1], [1, 2]]},
                        "set_a": [0],
                        "set_b": [2],
                        "set_c": [1],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graphical_model.bayes_net.construct",
        title="Construct a Bayesian network from CPTs",
        description="Bind every CPT exactly to its DAG parent set with row normalization; "
        "the network must be acyclic and each table must cover its parent assignments once.",
        request_type=BayesNetConstructRequest,
        result_type=BayesNetConstructResult,
        run=_bayes_net_construct,
        tags=("graphical-model", "bayes-net", "cpt", "exact"),
        examples=(
            OperationExample(
                name="two_node_chain",
                description="Construct a two-node binary chain; each CPT must bind its DAG parents.",
                input=_CHAIN_NET,
            ),
        ),
    ),
    MathTool(
        operation_id="graphical_model.bayes_net.joint.compute",
        title="Compute a Bayesian-network joint distribution",
        description="Return the induced joint distribution checked to sum to one; "
        "the network must carry one normalized CPT per variable.",
        request_type=BayesNetJointRequest,
        result_type=BayesNetJointResult,
        run=_bayes_net_joint,
        tags=("graphical-model", "bayes-net", "joint", "exact"),
        examples=(
            OperationExample(
                name="two_node_chain_joint",
                description="Joint of a two-node binary chain; the network must be CPT-bound.",
                input={
                    "network": {
                        "variable_count": 2,
                        "edges": [[0, 1]],
                        "domain_sizes": [2, 2],
                        "tables": [_CPT_ROOT, _CPT_CHILD],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graphical_model.variable_elimination.trace.compute",
        title="Trace exact variable elimination",
        description="Eliminate variables in the supplied order returning every intermediate "
        "scope, product/output factor, and fill edge; no order-optimality claim is made.",
        request_type=VariableEliminationTraceRequest,
        result_type=VariableEliminationTraceResult,
        run=_elimination_trace,
        tags=("graphical-model", "variable-elimination", "trace", "exact"),
        examples=(
            OperationExample(
                name="chain_elimination_trace",
                description="Trace elimination of variable 0 from a two-factor chain; the order must cover non-query variables.",
                input={
                    "factors": [
                        {
                            "variables": [0],
                            "domain_sizes": [2, 2],
                            "table": [
                                {"num": "3", "den": "5"},
                                {"num": "2", "den": "5"},
                            ],
                        },
                        {
                            "variables": [0, 1],
                            "domain_sizes": [2, 2],
                            "table": [
                                {"num": "1", "den": "4"},
                                {"num": "1", "den": "2"},
                                {"num": "3", "den": "4"},
                                {"num": "1", "den": "2"},
                            ],
                        },
                    ],
                    "domain_sizes": [2, 2],
                    "elimination_order": [0],
                    "query_variables": [1],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graphical_model.junction_tree.calibrate.compute",
        title="Calibrate junction-tree marginals",
        description="Return consistent clique/separator marginals and the partition value "
        "for a Bayes net and elimination order; marginals marginalize the exact joint.",
        request_type=JunctionTreeCalibrateRequest,
        result_type=JunctionTreeCalibrateResult,
        run=_junction_calibrate,
        tags=("graphical-model", "junction-tree", "calibration", "exact"),
        examples=(
            OperationExample(
                name="chain_junction_calibration",
                description="Calibrate a two-node chain eliminating variable 0; the network must be CPT-bound.",
                input={
                    "network": {
                        "variable_count": 2,
                        "edges": [[0, 1]],
                        "domain_sizes": [2, 2],
                        "tables": [_CPT_ROOT, _CPT_CHILD],
                    },
                    "elimination_order": [0],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
