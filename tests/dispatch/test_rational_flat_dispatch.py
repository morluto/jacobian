"""Public dispatch boundary for the motivating rational-flat request."""

from tests.support.rational_flats import seven_coordinate_source_problem

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_math_run_admits_seven_coordinate_source_problem() -> None:
    problem = seven_coordinate_source_problem()
    assert len(problem.candidates.vector_labels) == 105
    assert len(problem.clauses) == 21

    result = invoke_operation(
        "matroid.rational_flat.constrained_orbits.compute",
        {"problem": problem.model_dump(mode="json")},
        Catalog.open(),
    )

    assert result.output["outcome"]["status"] == "COMPLETE_EXACT"
    assert result.output["outcome"]["solution_flat_count"] == 2_940
