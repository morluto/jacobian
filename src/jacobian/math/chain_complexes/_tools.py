"""Typed declarations for chain complex operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.chain_complexes._models import (
    HomologyRequest,
    HomologyResult,
    MappingConeRequest,
    MappingConeResult,
)
from jacobian.math.chain_complexes._operations import (
    compute_homology,
    compute_mapping_cone,
)


def chain_complex_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_HOMOLOGY_EXAMPLE: dict[str, Any] = {
    "complex": {
        "prime": 2,
        "min_degree": 0,
        "max_degree": 2,
        "dimensions": [1, 2, 2],
        "differentials": [
            {"prime": 2, "entries": [[1, 1]], "columns": 2},
            {"prime": 2, "entries": [[1, 1], [1, 1]], "columns": 2},
        ],
    },
}


_MAPPING_CONE_EXAMPLE: dict[str, Any] = {
    "source": {
        "prime": 2,
        "min_degree": 0,
        "max_degree": 0,
        "dimensions": [1],
        "differentials": [],
    },
    "target": {
        "prime": 2,
        "min_degree": 0,
        "max_degree": 0,
        "dimensions": [1],
        "differentials": [],
    },
    "chain_map": [{"prime": 2, "entries": [[1]], "columns": 1}],
}


CHAIN_COMPLEX_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    chain_complex_operation(
        "homological_algebra.chain_complex.homology.compute",
        "Compute homology of a chain complex over a prime field",
        "Given a bounded based chain complex over GF(p), compute the exact "
        "homology groups H_n = ker(d_n) / im(d_{n+1}) for every degree. Each "
        "differential is a canonical prime-field matrix (boundary map "
        "C_n -> C_{n-1}) whose rank comes from the shared exact prime-field "
        "linear-algebra kernel, giving Betti numbers dim(C_n) - rank(d_n) - "
        "rank(d_{n+1}). Consecutive differentials must compose to zero.",
        HomologyRequest,
        HomologyResult,
        compute_homology,
        "homological-algebra",
        "chain-complex",
        "homology",
        "exact",
        examples=(
            example(
                "simple_complex",
                "Compute homology of a simple 3-term chain complex over GF(2).",
                _HOMOLOGY_EXAMPLE,
            ),
        ),
    ),
    chain_complex_operation(
        "homological_algebra.chain_complex.mapping_cone.compute",
        "Compute the mapping cone of a chain map",
        "Given chain complexes C (source) and D (target) and a chain map "
        "f: C -> D given as one canonical prime-field matrix per source "
        "degree satisfying the chain-map law, compute the mapping cone "
        "complex Cone(f) with groups Cone(f)_n = C_{n-1} + D_n. The mapping "
        "cone is the fundamental construction in homological algebra for "
        "computing long exact sequences in homology.",
        MappingConeRequest,
        MappingConeResult,
        compute_mapping_cone,
        "homological-algebra",
        "chain-complex",
        "mapping-cone",
        "exact",
        examples=(
            example(
                "identity_chain_map",
                "Compute the mapping cone of the identity chain map; the cone "
                "is acyclic.",
                _MAPPING_CONE_EXAMPLE,
            ),
        ),
    ),
)

TOOLS = CHAIN_COMPLEX_OPERATIONS

__all__ = ["TOOLS"]
