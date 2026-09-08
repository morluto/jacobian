"""Requests for a single rational exposed-face reduction."""

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer
from jacobian.math.matrices.semidefinite.values import (
    MAX_SEMIDEFINITE_CELLS,
    RationalSemidefiniteSystem,
)

_MAX_SOURCE_DIGITS = 8_000_000


def _raw_component_digits(component: object) -> int:
    if isinstance(component, str):
        return len(component.lstrip("-") or "0")
    if type(component) is int:
        return len(format_canonical_integer(abs(component)))
    return 0


def _raw_rational_digits(value: object) -> int:
    if isinstance(value, dict):
        return _raw_component_digits(value.get("num")) + _raw_component_digits(
            value.get("den")
        )
    if hasattr(value, "num") and hasattr(value, "den"):
        return _raw_component_digits(value.num) + _raw_component_digits(value.den)
    return 0


def _scan_matrix_entries(matrices: object) -> tuple[int, int]:
    if not isinstance(matrices, (list, tuple)):
        return 0, 0
    cells = 0
    digits = 0
    for matrix in matrices:
        entries = matrix.get("entries") if isinstance(matrix, dict) else None
        if entries is None and hasattr(matrix, "entries"):
            entries = matrix.entries
        if not isinstance(entries, (list, tuple)):
            continue
        for row in entries:
            if isinstance(row, (list, tuple)):
                cells += len(row)
                digits += sum(_raw_rational_digits(entry) for entry in row)
            else:
                cells += 1
                digits += _raw_rational_digits(row)
    return cells, digits


def _count_raw_cells_and_digits(data: dict[str, object]) -> tuple[int, int, int]:
    system = data.get("system")
    if not isinstance(system, dict):
        return 0, 0, 0
    order = system.get("order")
    matrices = system.get("matrices")
    cells, digits = _scan_matrix_entries(matrices)
    rhs = system.get("rhs")
    if isinstance(rhs, (list, tuple)):
        digits += sum(_raw_rational_digits(value) for value in rhs)
    multipliers = data.get("multipliers")
    if isinstance(multipliers, (list, tuple)):
        digits += sum(_raw_rational_digits(value) for value in multipliers)
    declared = 0
    if type(order) is int:
        if order < 0:
            raise ValueError(
                "source and reduced matrices exceed the dense cell envelope"
            )
        if isinstance(matrices, (list, tuple)):
            declared = len(matrices) * order * order
    return cells, declared, digits


class SemidefiniteFaceReductionRequest(StrictModel):
    system: RationalSemidefiniteSystem
    multipliers: tuple[CanonicalRational, ...] = Field(
        max_length=8192,
        description="Supplied y with nonzero PSD sum y_i A_i and sum y_i b_i = 0.",
    )

    @model_validator(mode="before")
    @classmethod
    def require_aggregate_cells(cls, data: object) -> object:
        """Reject over-budget dense systems before nested matrix parsing."""

        if isinstance(data, dict):
            cells, declared, digits = _count_raw_cells_and_digits(data)
            if cells > MAX_SEMIDEFINITE_CELLS or declared > MAX_SEMIDEFINITE_CELLS:
                raise ValueError(
                    "source and reduced matrices exceed the dense cell envelope"
                )
            if digits > _MAX_SOURCE_DIGITS:
                raise ValueError(
                    "source rationals exceed the admitted aggregate digit envelope"
                )
        return canonicalize_json_containers(data)
