"""Public declarations for exact periodic rational fan operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry.periodic_fans._models import (
    MAX_PERIODIC_CELLS,
    MAX_PERIODIC_COORDINATE_DIGITS,
    MAX_PERIODIC_LATTICE_RANK,
    MAX_PERIODIC_OVERLAP_CANDIDATES,
    MAX_PERIODIC_VERTICES,
    PeriodicFanValidationRequest,
    PeriodicFanValidationResult,
)
from jacobian.math.geometry.periodic_fans.operations import validate_periodic_fan

_ENVELOPE_SENTENCE = (
    f"Envelope: lattice rank at most {MAX_PERIODIC_LATTICE_RANK}, at most "
    f"{MAX_PERIODIC_CELLS} maximal cells, {MAX_PERIODIC_VERTICES} vertices, "
    f"{MAX_PERIODIC_OVERLAP_CANDIDATES} overlap candidates, and "
    f"{MAX_PERIODIC_COORDINATE_DIGITS} decimal digits per coordinate."
)

_UNIT_SQUARE_VERTICES = [["0", "0"], ["1", "0"], ["0", "1"], ["1", "1"]]
_UNIT_SQUARE_CELLS = [[0, 1, 3], [0, 2, 3]]


def _unit_square_overlap_candidates() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for first in range(2):
        for second in range(first, 2):
            for first_axis in (-1, 0, 1):
                for second_axis in (-1, 0, 1):
                    candidates.append(
                        {
                            "first_cell": first,
                            "second_cell": second,
                            "translation": [str(first_axis), str(second_axis)],
                        }
                    )
    return candidates


def _run_validate_periodic_fan(
    request: PeriodicFanValidationRequest,
) -> PeriodicFanValidationResult:
    return validate_periodic_fan(request.fan)


PERIODIC_FAN_VALIDATE_OPERATION = MathTool(
    operation_id="periodic_fan.quotient.validate",
    title="Validate a periodic rational fan and return its finite quotient",
    description=(
        "Decide whether a bounded finite presentation defines a rational fan "
        "invariant under a full-rank period lattice: the basis is integral and "
        "full rank with an exact index, every maximal cell is a nondegenerate "
        "simplex in the fundamental parallelotope, every declared overlap is a "
        "common face, every actual overlap is declared, the cells cover the "
        "fundamental parallelotope, and unimodularity claims hold under the "
        "integer Smith normal form. On success return the finite quotient "
        "incidence: quotient cells with orbit members and exact translation "
        "stabilizer sublattices, the face-to-orbit map, quotient face relations, "
        "and the validated local-finiteness and face-to-face properties. No "
        "translated cell is materialized; local finiteness follows from lattice "
        "discreteness and the bounded fundamental region. " + _ENVELOPE_SENTENCE
    ),
    request_type=PeriodicFanValidationRequest,
    result_type=PeriodicFanValidationResult,
    run=_run_validate_periodic_fan,
    tags=("geometry", "fan", "periodic", "lattice", "quotient", "exact"),
    discovery_terms=(
        "periodic rational fan validation",
        "lattice-periodic triangulation quotient",
        "translation-invariant fan quotient",
        "cell stabilizer sublattice",
    ),
    examples=(
        OperationExample(
            name="z2_periodic_unit_square_triangulation",
            description=(
                "Validate the standard Z^2-periodic triangulation of the plane "
                "cut from the unit square by one diagonal; precondition: integer "
                "vertices in the closed fundamental parallelotope, sorted cells, "
                "and a complete candidate list of adjacent translates."
            ),
            input={
                "fan": {
                    "lattice_rank": 2,
                    "period_basis": [
                        [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                        [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                    ],
                    "vertices": _UNIT_SQUARE_VERTICES,
                    "cells": _UNIT_SQUARE_CELLS,
                    "unimodular_cells": [0, 1],
                    "overlap_candidates": _unit_square_overlap_candidates(),
                }
            },
        ),
    ),
)

TOOLS: MathTools = (PERIODIC_FAN_VALIDATE_OPERATION,)

__all__ = ["TOOLS"]
