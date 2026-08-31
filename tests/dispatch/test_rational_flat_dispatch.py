"""Native and dispatch parity for clause-constrained rational flats."""

from tests.support.rational_flats import seven_coordinate_source_problem

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.combinatorics.matroids.rational_flats import (
    classify_clause_constrained_rational_flats,
)


def test_dispatch_preserves_the_complete_348_rooted_flat_known_answer() -> None:
    problem = seven_coordinate_source_problem()

    native = classify_clause_constrained_rational_flats(problem)
    dispatched = invoke_operation(
        "matroid.rational_flat.constrained_orbits.compute",
        {"problem": problem.model_dump(mode="json")},
        Catalog.open(),
    )

    assert native.outcome.status == "COMPLETE_EXACT"
    assert dispatched.output == native.model_dump(mode="json")
    rooted_incidence = tuple(
        representative.orbit_size
        * len(representative.closed_candidate_indices)
        // problem.candidates.vector_count
        for representative in sorted(
            native.outcome.representatives,
            key=lambda item: item.rank,
        )
    )
    # Issue #2470's bespoke search reported 36 + 293 = 329 rooted flats.  The
    # complete orbit-stabilizer result gives all 36 + 312 = 348 instead.
    assert rooted_incidence == (36, 312)
    assert sum(rooted_incidence) == 348
