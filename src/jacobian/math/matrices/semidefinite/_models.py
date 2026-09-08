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


def _materialize_sequence(value: object, *, limit: int) -> list[object] | None:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    if isinstance(value, Sequence):
        return list(value)
    if isinstance(value, Iterable):
        collected: list[object] = []
        for item in value:
            collected.append(item)
            if len(collected) > limit:
                break
        return collected
    return None


def _install_sequence(
    container: dict[str, object], key: str, *, limit: int
) -> list[object] | None:
    value = container.get(key)
    materialized = _materialize_sequence(value, limit=limit)
    if materialized is not None and not isinstance(value, Sequence):
        container[key] = materialized
    return materialized


def _scan_matrix_entries(matrices: object) -> tuple[int, int]:
    matrix_list = _materialize_sequence(matrices, limit=MAX_SEMIDEFINITE_CELLS)
    if matrix_list is None:
        return 0, 0
    cells = 0
    digits = 0
    for matrix in matrix_list:
        entries = matrix.get("entries") if isinstance(matrix, dict) else None
        if entries is None and hasattr(matrix, "entries"):
            entries = matrix.entries
        row_list = _materialize_sequence(entries, limit=MAX_SEMIDEFINITE_CELLS)
        if row_list is None:
            continue
        for row in row_list:
            if isinstance(row, (list, tuple)):
                cells += len(row)
                digits += sum(_raw_rational_digits(entry) for entry in row)
            else:
                cells += 1
                digits += _raw_rational_digits(row)
            if cells > MAX_SEMIDEFINITE_CELLS:
                return cells, digits
    return cells, digits


def _preflight_raw_payload(
    data: dict[str, object],
) -> tuple[dict[str, object], int, int, int]:
    payload = dict(data)
    system = payload.get("system")
    if not isinstance(system, dict):
        return payload, 0, 0, 0
    system = dict(system)
    payload["system"] = system
    matrices_value = system.get("matrices")
    matrix_list = _materialize_sequence(matrices_value, limit=MAX_SEMIDEFINITE_CELLS)
    installed_matrices: list[object] | None = None
    if matrix_list is not None:
        installed_matrices = []
        needs_install = not isinstance(matrices_value, Sequence)
        for matrix in matrix_list:
            if isinstance(matrix, dict):
                matrix_payload = dict(matrix)
                entries = matrix_payload.get("entries")
                row_list = _install_sequence(
                    matrix_payload, "entries", limit=MAX_SEMIDEFINITE_CELLS
                )
                if row_list is not None and not isinstance(entries, Sequence):
                    needs_install = True
                installed_matrices.append(matrix_payload)
            else:
                installed_matrices.append(matrix)
        if needs_install:
            system["matrices"] = installed_matrices
    cells, digits = _scan_matrix_entries(
        installed_matrices if installed_matrices is not None else system.get("matrices")
    )
    rhs = system.get("rhs")
    if isinstance(rhs, (list, tuple)):
        digits += sum(_raw_rational_digits(value) for value in rhs)
    multipliers = payload.get("multipliers")
    if isinstance(multipliers, (list, tuple)):
        digits += sum(_raw_rational_digits(value) for value in multipliers)
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

        if isinstance(data, dict):
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
