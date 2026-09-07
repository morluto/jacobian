"""In-process deterministic modular row-HNF producer owned by the lattice domain."""

from __future__ import annotations

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.lattices import hermite_normal_form
from jacobian.math.lattices._models import (
    HermiteNormalFormRequest,
    HermiteNormalFormResult,
)
from jacobian.math.matrices.values import IntegerMatrix


def _matrix(value: Any) -> IntegerMatrix:
    return IntegerMatrix(
        entries=tuple(
            tuple(int(value[row, column]) for column in range(value.ncols()))
            for row in range(value.nrows())
        )
    )


def compute_hermite_normal_form(
    request: HermiteNormalFormRequest,
) -> HermiteNormalFormResult:
    integer_entries = [list(row) for row in request.matrix.entries]
    normal_form, transformation = hermite_normal_form(integer_entries)
    return HermiteNormalFormResult(
        normal_form=_matrix(normal_form),
        transformation=_matrix(transformation),
    )


HERMITE_NORMAL_FORM_OPERATION: MathTool[
    HermiteNormalFormRequest,
    HermiteNormalFormResult,
] = MathTool(
    operation_id="lattice.hermite_normal_form.compute",
    title="Compute an exact row Hermite normal form",
    description=(
        "Compute H and U for one bounded integer matrix, retaining A so that "
        "the serialized result retains the relation H = U A."
    ),
    discovery_terms=(
        "integer row lattice membership canonical basis",
        "integer column lattice via transpose",
        "integer multipliers lattice vector membership witness",
        "unimodular row transformation",
    ),
    request_type=HermiteNormalFormRequest,
    result_type=HermiteNormalFormResult,
    run=compute_hermite_normal_form,
    tags=(
        "lattice",
        "matrix",
        "integer",
        "hermite-normal-form",
    ),
    examples=(
        OperationExample(
            name="unit_matrix",
            description="Compute the row HNF of the one-by-one unit matrix.",
            input={"matrix": {"entries": [["1"]]}},
        ),
    ),
)

__all__ = ["HERMITE_NORMAL_FORM_OPERATION", "compute_hermite_normal_form"]
