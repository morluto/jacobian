"""Requests for a single rational exposed-face reduction."""

from collections.abc import Iterable, Mapping, Sequence

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer
from jacobian.math.matrices.semidefinite.values import (
    MAX_SEMIDEFINITE_CELLS,
    RationalSemidefiniteSystem,
)

_MAX_SOURCE_DIGITS = 8_000_000
_MAX_EQUALITIES = 8192


def _raw_component_digits(component: object) -> int:
    if isinstance(component, str):
        return len(component.lstrip("-") or "0")
    if type(component) is int:
        return len(format_canonical_integer(abs(component)))
    return 0


def _raw_rational_digits(value: object) -> int:
    if isinstance(value, Mapping):
        return _raw_component_digits(value.get("num")) + _raw_component_digits(
            value.get("den")
        )
    if hasattr(value, "num") and hasattr(value, "den"):
        return _raw_component_digits(value.num) + _raw_component_digits(value.den)
    return 0


def _materialize_sequence(value: object, *, limit: int) -> list[object] | None:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    if not isinstance(value, Iterable):
        return None
    collected: list[object] = []
    for item in value:
        collected.append(item)
        if len(collected) > limit:
            break
    return collected


def _scan_row(row: object, *, limit: int) -> tuple[object, int, int, bool]:
    """Count cells in one row and materialize a generated row up to ``limit``."""

    if isinstance(row, (str, bytes, bytearray, Mapping)):
        return row, 1, _raw_rational_digits(row), False
    if isinstance(row, Sequence):
        row_length = len(row)
        if row_length > limit:
            return row, row_length, 0, False
        if isinstance(row, (list, tuple)):
            return (
                row,
                row_length,
                sum(_raw_rational_digits(entry) for entry in row),
                False,
            )
    materialized = _materialize_sequence(row, limit=limit)
    if materialized is None:
        return row, 1, _raw_rational_digits(row), False
    return (
        materialized,
        len(materialized),
        sum(_raw_rational_digits(entry) for entry in materialized),
        True,
    )


def _scan_entries(
    entries: object, *, cells: int
) -> tuple[list[object] | None, int, int, bool]:
    row_list = _materialize_sequence(entries, limit=MAX_SEMIDEFINITE_CELLS - cells)
    if row_list is None:
        return None, cells, 0, False
    installed_rows: list[object] = []
    replace_entries = not isinstance(entries, (list, tuple))
    digits = 0
    for row in row_list:
        installed_row, row_cells, row_digits, replaced = _scan_row(
            row, limit=MAX_SEMIDEFINITE_CELLS - cells
        )
        installed_rows.append(installed_row)
        replace_entries = replace_entries or replaced
        cells += row_cells
        digits += row_digits
        if cells > MAX_SEMIDEFINITE_CELLS:
            break
    return installed_rows, cells, digits, replace_entries


def _scan_and_install_scalars(
    container: dict[str, object],
    key: str,
    *,
    digits: int,
) -> int:
    value = container.get(key)
    if isinstance(value, (str, bytes, bytearray, Mapping)) or value is None:
        return digits
    if not isinstance(value, Iterable):
        return digits
    collected: list[object] = []
    extra = 0
    remaining = _MAX_SOURCE_DIGITS - digits
    for item in value:
        collected.append(item)
        extra += _raw_rational_digits(item)
        if extra > remaining or len(collected) > _MAX_EQUALITIES:
            break
    if not isinstance(value, (list, tuple)):
        container[key] = collected
    return digits + extra


def _preflight_raw_payload(
    data: Mapping[str, object],
) -> tuple[dict[str, object], int, int, int]:
    payload = dict(data)
    system = payload.get("system")
    if not isinstance(system, Mapping):
        digits = _scan_and_install_scalars(payload, "multipliers", digits=0)
        return payload, 0, 0, digits
    system = dict(system)
    payload["system"] = system
    matrices_value = system.get("matrices")
    matrix_list = _materialize_sequence(matrices_value, limit=MAX_SEMIDEFINITE_CELLS)
    installed_matrices: list[object] | None = None
    cells = 0
    digits = 0
    if matrix_list is not None:
        installed_matrices = []
        needs_install = not isinstance(matrices_value, (list, tuple))
        if len(matrix_list) > MAX_SEMIDEFINITE_CELLS:
            cells = MAX_SEMIDEFINITE_CELLS + 1
        for matrix in matrix_list:
            if cells > MAX_SEMIDEFINITE_CELLS:
                break
            if not isinstance(matrix, Mapping):
                entries = matrix.entries if hasattr(matrix, "entries") else None
                _rows, cells, row_digits, _replaced = _scan_entries(
                    entries, cells=cells
                )
                digits += row_digits
                installed_matrices.append(matrix)
                continue
            matrix_payload = dict(matrix)
            if not isinstance(matrix, dict):
                needs_install = True
            installed_rows, cells, row_digits, replace_entries = _scan_entries(
                matrix_payload.get("entries"), cells=cells
            )
            digits += row_digits
            if replace_entries and installed_rows is not None:
                matrix_payload["entries"] = installed_rows
                needs_install = True
            installed_matrices.append(matrix_payload)
        if needs_install:
            system["matrices"] = installed_matrices
    digits = _scan_and_install_scalars(system, "rhs", digits=digits)
    digits = _scan_and_install_scalars(payload, "multipliers", digits=digits)
    declared = 0
    order = system.get("order")
    if type(order) is int:
        if order < 0:
            raise ValueError(
                "source and reduced matrices exceed the dense cell envelope"
            )
        declared_matrices = (
            installed_matrices
            if installed_matrices is not None
            else _materialize_sequence(system.get("matrices"), limit=8192)
        )
        if declared_matrices is not None:
            declared = len(declared_matrices) * order * order
    return payload, cells, declared, digits


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

        if isinstance(data, Mapping):
            data, cells, declared, digits = _preflight_raw_payload(data)
            if cells > MAX_SEMIDEFINITE_CELLS or declared > MAX_SEMIDEFINITE_CELLS:
                raise ValueError(
                    "source and reduced matrices exceed the dense cell envelope"
                )
            if digits > _MAX_SOURCE_DIGITS:
                raise ValueError(
                    "source rationals exceed the admitted aggregate digit envelope"
                )
        return canonicalize_json_containers(data)
