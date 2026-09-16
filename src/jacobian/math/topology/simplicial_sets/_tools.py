"""Public declaration for finite truncated simplicial sets."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets._models import (
    SimplicialSetTablesRequest,
    SimplicialSetTablesResult,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables


def _run_from_tables(request: SimplicialSetTablesRequest) -> SimplicialSetTablesResult:
    return from_tables(request)


# Delta[1] truncated to degrees 0..1: X_0 = {0, 1}, X_1 = {00, 01, 11};
# faces delete one entry and s_0 duplicates the entry.
_DELTA_ONE_PREFIX = {
    "max_degree": 1,
    "sets": [["0", "1"], ["00", "01", "11"]],
    "face_maps": [[[0, 1, 1], [0, 0, 1]]],
    "degeneracy_maps": [[[0, 2]]],
}

TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.from_tables.compute",
        title="Check finite simplicial-set tables against every simplicial identity",
        description=(
            "Exhaust every in-range face/degeneracy identity for complete finite "
            "degree tables X_0..X_N. Return the canonical truncated simplicial "
            "set with its identity count, or the first unequal identity row. "
            "Tables must be complete: every degree carries all its simplices "
            "and every in-range face and degeneracy map as index rows."
        ),
        request_type=SimplicialSetTablesRequest,
        result_type=SimplicialSetTablesResult,
        run=_run_from_tables,
        tags=("topology", "simplicial-set", "face-map", "degeneracy-map", "exact"),
        discovery_terms=(
            "simplicial set",
            "face map",
            "degeneracy map",
            "simplicial identity",
            "truncated simplicial set",
        ),
        examples=(
            OperationExample(
                name="delta_one_prefix",
                description=(
                    "Check the degree-0..1 prefix of the 1-simplex Delta[1]; "
                    "face rows delete one entry and the degeneracy row "
                    "duplicates the entry."
                ),
                input=_DELTA_ONE_PREFIX,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
