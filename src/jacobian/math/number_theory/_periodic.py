"""Declarations for finite periodic congruence unions."""

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceSubset,
    PeriodicCongruenceUnionMeasureResult,
    PeriodicCongruenceUnionProfileRequest,
    PeriodicCongruenceUnionProfileResult,
    PeriodicCongruenceUnionRequest,
    PeriodicCongruenceUnionSource,
)
from jacobian.math.number_theory.operations import (
    periodic_congruence_union_measure,
    periodic_congruence_union_profile,
)


def normalize_periodic_source(
    request: PeriodicCongruenceUnionRequest,
) -> PeriodicCongruenceUnionSource:
    """Project wire rows into the canonical periodic-union source value."""

    merged: dict[int, set[int]] = {}
    for subset in request.subsets:
        modulus = parse_canonical_integer(subset.modulus)
        residues = merged.setdefault(modulus, set())
        residues.update(
            parse_canonical_integer(residue) % modulus for residue in subset.residues
        )
    return PeriodicCongruenceUnionSource(
        subsets=tuple(
            PeriodicCongruenceSubset(
                modulus=format_canonical_integer(modulus),
                residues=tuple(
                    format_canonical_integer(residue) for residue in sorted(residues)
                ),
            )
            for modulus, residues in sorted(merged.items())
        ),
        complement=request.complement,
    )


def compute_periodic_congruence_union_measure(
    request: PeriodicCongruenceUnionRequest,
) -> PeriodicCongruenceUnionMeasureResult:
    """Project a wire request onto the canonical measure operation."""

    return periodic_congruence_union_measure(normalize_periodic_source(request))


def compute_periodic_congruence_union_profile(
    request: PeriodicCongruenceUnionProfileRequest,
) -> PeriodicCongruenceUnionProfileResult:
    """Project a wire request onto the canonical profile operation."""

    return periodic_congruence_union_profile(normalize_periodic_source(request))


PERIODIC_CONGRUENCE_OPERATIONS = (
    MathTool(
        operation_id="congruence.periodic_union.measure.compute",
        title="Measure finite periodic congruence union",
        description=(
            "Compute the exact occupied count and rational density of a normalized "
            "finite union of residue subsets in the least common multiple of their "
            "moduli, optionally complemented, without requiring residue materialization."
        ),
        request_type=PeriodicCongruenceUnionRequest,
        result_type=PeriodicCongruenceUnionMeasureResult,
        run=compute_periodic_congruence_union_measure,
        tags=("number-theory", "combinatorics", "congruence", "periodic", "density"),
        examples=(
            OperationExample(
                name="overlapping_periodic_union_measure",
                description=(
                    "Measure the union of residues {0,1} modulo 4 and {-1,1} "
                    "modulo 6; moduli must be positive decimal strings and residue "
                    "representatives are normalized modulo their modulus."
                ),
                input={
                    "subsets": [
                        {"modulus": "4", "residues": ["0", "1"]},
                        {"modulus": "6", "residues": ["-1", "1"]},
                    ],
                    "complement": False,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="congruence.periodic_union.profile.compute",
        title="Materialize finite periodic congruence union",
        description=(
            "Return every occupied residue in the least common multiple period of "
            "a normalized finite union of residue subsets, together with its exact "
            "count and density, optionally after complementing the union."
        ),
        request_type=PeriodicCongruenceUnionProfileRequest,
        result_type=PeriodicCongruenceUnionProfileResult,
        run=compute_periodic_congruence_union_profile,
        tags=(
            "number-theory",
            "combinatorics",
            "congruence",
            "periodic",
            "enumeration",
        ),
        examples=(
            OperationExample(
                name="complemented_periodic_union_profile",
                description=(
                    "Materialize residues avoiding the even classes modulo 4 and "
                    "the class 1 modulo 3; the lcm period and complete residue output "
                    "must fit the materialization bounds."
                ),
                input={
                    "subsets": [
                        {"modulus": "4", "residues": ["0", "2"]},
                        {"modulus": "3", "residues": ["1"]},
                    ],
                    "complement": True,
                },
            ),
        ),
    ),
)

__all__ = ["PERIODIC_CONGRUENCE_OPERATIONS"]
