"""Public declaration for clause-constrained rational-flat classification."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.matroids.rational_flats._models import (
    ClauseConstrainedRationalFlatClassification,
    ClauseConstrainedRationalFlatRequest,
)
from jacobian.math.combinatorics.matroids.rational_flats.operations import (
    classify_clause_constrained_rational_flats as _classify_native,
)


def classify_clause_constrained_rational_flats(
    request: ClauseConstrainedRationalFlatRequest,
) -> ClauseConstrainedRationalFlatClassification:
    return _classify_native(request.problem)


CLASSIFY_CONSTRAINED_RATIONAL_FLATS_OPERATION = MathTool(
    operation_id="matroid.rational_flat.constrained_orbits.compute",
    title="Classify clause-constrained rational flats up to finite symmetry",
    description=(
        "For a bounded labelled rational row configuration, covering clauses, "
        "forbidden span incidences, an optional rank interval, and compatible "
        "paired coordinate/candidate permutation generators, return every closed "
        "row-matroid flat satisfying the constraints once per symmetry orbit. "
        "Each complete representative retains its closed candidate set, exact "
        "RREF row-space basis, canonical annihilator basis, rank, orbit size, "
        "and stabilizer order. If the deterministic state, work, or exact-output "
        "envelope is exhausted, the result is explicitly INCOMPLETE and makes no "
        "family-"
        "completeness claim."
    ),
    request_type=ClauseConstrainedRationalFlatRequest,
    result_type=ClauseConstrainedRationalFlatClassification,
    run=classify_clause_constrained_rational_flats,
    tags=(
        "matroid",
        "rational-flat",
        "covering-clause",
        "finite-symmetry",
        "orbit-stabilizer",
        "exact",
    ),
    discovery_terms=(
        "clause-constrained rational flats",
        "represented matroid flat orbits",
        "forbidden row-span incidence",
    ),
    examples=(
        example(
            "line_and_plane_containing_one_required_equation",
            "Classify the closed spans containing x+y=0 in a three-row "
            "configuration; candidate and forbidden matrices must retain the "
            "two-coordinate ambient axis and an empty generator tuple means "
            "trivial symmetry.",
            {
                "problem": {
                    "candidates": {
                        "coordinate_axis": ["x", "y"],
                        "vector_labels": ["x=0", "y=0", "x+y=0"],
                        "vectors": {
                            "domain": "QQ",
                            "row_count": 3,
                            "column_count": 2,
                            "entries": [
                                {
                                    "row": 0,
                                    "column": 0,
                                    "value": {"num": "1", "den": "1"},
                                },
                                {
                                    "row": 1,
                                    "column": 1,
                                    "value": {"num": "1", "den": "1"},
                                },
                                {
                                    "row": 2,
                                    "column": 0,
                                    "value": {"num": "1", "den": "1"},
                                },
                                {
                                    "row": 2,
                                    "column": 1,
                                    "value": {"num": "1", "den": "1"},
                                },
                            ],
                        },
                    },
                    "clauses": [[2]],
                    "forbidden_vectors": {
                        "domain": "QQ",
                        "row_count": 0,
                        "column_count": 2,
                        "entries": [],
                    },
                    "rank_interval": {"minimum": 0, "maximum": 2},
                    "symmetry_generators": [],
                }
            },
        ),
    ),
)

TOOLS: MathTools = (CLASSIFY_CONSTRAINED_RATIONAL_FLATS_OPERATION,)

__all__ = ["TOOLS"]
