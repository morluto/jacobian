"""Catalog declaration for induced cellular-sheaf cohomology maps."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.topology.cellular_sheaves.cohomology_maps import (
    SheafCohomologyMapRequest,
    SheafCohomologyMapResult,
    cohomology_map,
)


def _run(request: SheafCohomologyMapRequest) -> SheafCohomologyMapResult:
    return cohomology_map(request.morphism)


TOOLS: MathTools = (
    MathTool(
        operation_id="cellular_sheaf.morphism.cohomology_map.compute",
        title="Induce the map on cellular sheaf cohomology",
        description=(
            "Re-establish a natural cellular-sheaf morphism, compute the exact "
            "source and target cohomology groups, and return the induced linear "
            "map in their retained cocycle-representative bases."
        ),
        request_type=SheafCohomologyMapRequest,
        result_type=SheafCohomologyMapResult,
        run=_run,
        tags=("topology", "cellular-sheaf", "morphism", "cohomology", "exact"),
        discovery_terms=(
            "induced cellular sheaf cohomology map",
            "sheaf cohomology functoriality",
            "map on cellular sheaf cohomology",
        ),
        examples=(
            OperationExample(
                name="single_vertex_scalar_map",
                description=(
                    "Multiplication by two on the one-dimensional cohomology of "
                    "the constant rank-one sheaf on a single vertex."
                ),
                input={
                    "morphism": {
                        "source": {
                            "complex": {
                                "vertices": ["a"],
                                "maximal_simplices": [["a"]],
                                "faces_by_dimension": [
                                    {"dimension": 0, "faces": [["a"]]}
                                ],
                                "dimension": 0,
                                "f_vector": [1],
                                "closure_size": 1,
                            },
                            "coefficient_field": "QQ",
                            "prime": None,
                            "stalks": [{"simplex": ["a"], "basis": ["x"]}],
                            "cover_restrictions": [],
                            "derived_restrictions": [],
                            "diamonds": 0,
                            "comparable_pairs": 0,
                        },
                        "target": {
                            "complex": {
                                "vertices": ["a"],
                                "maximal_simplices": [["a"]],
                                "faces_by_dimension": [
                                    {"dimension": 0, "faces": [["a"]]}
                                ],
                                "dimension": 0,
                                "f_vector": [1],
                                "closure_size": 1,
                            },
                            "coefficient_field": "QQ",
                            "prime": None,
                            "stalks": [{"simplex": ["a"], "basis": ["x"]}],
                            "cover_restrictions": [],
                            "derived_restrictions": [],
                            "diamonds": 0,
                            "comparable_pairs": 0,
                        },
                        "components": [[["a"], [[{"num": "2", "den": "1"}]]]],
                        "natural": True,
                        "obstruction": None,
                    }
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
