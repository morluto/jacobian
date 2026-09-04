"""Public declaration for clause-constrained prime-field flat classification."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.matroids.prime_field_flats._models import (
    ClauseConstrainedPrimeFieldFlatClassification,
    ClauseConstrainedPrimeFieldFlatRequest,
)
from jacobian.math.combinatorics.matroids.prime_field_flats.operations import (
    classify_clause_constrained_prime_field_flats as _classify_native,
)


def classify_clause_constrained_prime_field_flats(
    request: ClauseConstrainedPrimeFieldFlatRequest,
) -> ClauseConstrainedPrimeFieldFlatClassification:
    return _classify_native(request.problem)


CLASSIFY_CONSTRAINED_PRIME_FIELD_FLATS_OPERATION = MathTool(
    operation_id="matroid.prime_field_flat.constrained_orbits.compute",
    title="Classify clause-constrained GF(p) flats up to finite symmetry",
    description=(
        "For a bounded labelled prime-field row configuration, covering clauses, "
        "forbidden span incidences, an optional rank interval, and compatible "
        "paired coordinate/candidate permutation generators, return every closed "
        "row-matroid flat satisfying the constraints once per symmetry orbit. "
        "Each complete representative retains its closed candidate set, exact "
        "GF(p) RREF row-space basis, canonical annihilator basis, rank, orbit "
        "size, and stabilizer order. A bounded state, work, or exact-output stop "
        "returns INCOMPLETE and makes no family-completeness claim."
    ),
    request_type=ClauseConstrainedPrimeFieldFlatRequest,
    result_type=ClauseConstrainedPrimeFieldFlatClassification,
    run=classify_clause_constrained_prime_field_flats,
    tags=(
        "matroid",
        "prime-field-flat",
        "covering-clause",
        "finite-symmetry",
        "orbit-stabilizer",
        "exact",
    ),
    discovery_terms=(
        "clause-constrained prime-field flats",
        "GF(p) represented matroid flat orbits",
        "forbidden row-span incidence",
    ),
    examples=(
        OperationExample(
            name="gf3_lines_containing_x_axis",
            description=(
                "Classify GF(3) closed spans in a three-row configuration while "
                "forbidding the all-one row; all matrix rows must use the "
                "declared prime and canonical residues."
            ),
            input={
                "problem": {
                    "candidates": {
                        "prime": 3,
                        "coordinate_axis": ["x", "y", "z"],
                        "vector_labels": ["a", "b", "c"],
                        "vectors": {
                            "prime": 3,
                            "entries": [[1, 2, 0], [1, 0, 2], [0, 1, 1]],
                            "columns": 3,
                        },
                    },
                    "clauses": [[1, 2]],
                    "forbidden_vectors": {
                        "prime": 3,
                        "entries": [[1, 1, 1]],
                        "columns": 3,
                    },
                    "rank_interval": {"minimum": 0, "maximum": 3},
                    "symmetry_generators": [],
                }
            },
        ),
    ),
)

TOOLS: MathTools = (CLASSIFY_CONSTRAINED_PRIME_FIELD_FLATS_OPERATION,)

__all__ = ["TOOLS"]
