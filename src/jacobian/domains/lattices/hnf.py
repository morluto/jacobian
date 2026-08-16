"""In-process Python-FLINT row-HNF producer owned by the lattice domain."""

from __future__ import annotations

from typing import Any

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.contracts.lattices import (
    HermiteNormalFormRequest,
    HermiteNormalFormResult,
)
from jacobian.contracts.matrices import IntegerMatrix
from jacobian.domains._examples import example
from jacobian.math.lattices import hermite_normal_form
from jacobian.math_tools import MathTool


def _matrix(value: Any) -> IntegerMatrix:
    return IntegerMatrix(
        entries=tuple(
            tuple(
                format_canonical_integer(int(value[row, column]))
                for column in range(value.ncols())
            )
            for row in range(value.nrows())
        )
    )


def compute_hermite_normal_form(
    request: HermiteNormalFormRequest,
) -> HermiteNormalFormResult:
    integer_entries = [
        [parse_canonical_integer(value) for value in row]
        for row in request.matrix.entries
    ]
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
    version="1",
    title="Compute an exact row Hermite normal form",
    description=("Compute H and U for one bounded integer matrix with H = U A."),
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
        example(
            "unit_matrix",
            "Compute the row HNF of the one-by-one unit matrix.",
            {"matrix": {"entries": [["1"]]}},
        ),
    ),
)

__all__ = ["HERMITE_NORMAL_FORM_OPERATION", "compute_hermite_normal_form"]
