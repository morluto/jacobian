"""Canonical values for lattices of invariant integral bilinear forms."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Self, cast

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import parse_canonical_integer
from jacobian.math._labels import MAX_OPAQUE_LABEL_LENGTH, OpaqueLabel
from jacobian.math.matrices.values import (
    MAX_RATIONAL_MATRIX_ORDER,
    IntegerMatrix,
    RationalMatrix,
)

MAX_ACTION_DIMENSION = MAX_RATIONAL_MATRIX_ORDER
MAX_ACTION_GENERATORS = 65_536
MAX_CONSTRAINT_CELLS = 65_536
MAX_CONSTRAINT_DIGIT_WORK = 500_000_000
MAX_INTEGER_KERNEL_DIGIT_WORK = 500_000_000
MAX_FORM_COEFFICIENT_DIMENSION = MAX_ACTION_DIMENSION**2

FormKind = Literal["BILINEAR", "SYMMETRIC", "ALTERNATING"]
FormCoefficientOrder = Literal[
    "ROW_MAJOR",
    "UPPER_TRIANGULAR_ROW_MAJOR",
    "STRICT_UPPER_TRIANGULAR_ROW_MAJOR",
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"lattice.invariant_form.{reason}", message)


def coefficient_dimension(dimension: int, kind: FormKind) -> int:
    """Return the number of independent coefficients of one form kind."""

    if kind == "BILINEAR":
        return dimension * dimension
    if kind == "SYMMETRIC":
        return dimension * (dimension + 1) // 2
    return dimension * (dimension - 1) // 2


def coefficient_order(kind: FormKind) -> FormCoefficientOrder:
    """Return the canonical coefficient order for one form kind."""

    if kind == "BILINEAR":
        return "ROW_MAJOR"
    if kind == "SYMMETRIC":
        return "UPPER_TRIANGULAR_ROW_MAJOR"
    return "STRICT_UPPER_TRIANGULAR_ROW_MAJOR"


def constraint_coefficient_count(
    dimension: int, generator_count: int, kind: FormKind
) -> int:
    """Return the exact number of congruence coefficients to construct."""

    return (
        generator_count * dimension * dimension * coefficient_dimension(dimension, kind)
    )


def _canonicalize_generator_order(data: dict[str, object]) -> dict[str, object]:
    normalized = dict(data)
    normalized_axis = normalized.get("coordinate_axis")
    if isinstance(normalized_axis, list):
        normalized["coordinate_axis"] = tuple(normalized_axis)
    normalized_generators = normalized.get("generators")
    if isinstance(normalized_generators, (str, bytes, Mapping)):
        return normalized
    try:
        generator_iterator = iter(cast(Iterable[object], normalized_generators))
    except TypeError:
        return normalized
    generator_values: list[object] = []
    for _index in range(MAX_ACTION_GENERATORS + 1):
        try:
            generator_values.append(next(generator_iterator))
        except StopIteration:
            break
    else:
        raise _validation_error(
            "budget_exceeded",
            f"an action has at most {MAX_ACTION_GENERATORS} generators",
        )
    normalized["generators"] = tuple(generator_values)
    labelled_generators: list[tuple[str, object]] = []
    for generator in generator_values:
        label = (
            generator.get("label")
            if isinstance(generator, dict)
            else getattr(generator, "label", None)
        )
        if not isinstance(label, str):
            return normalized
        if len(label) > MAX_OPAQUE_LABEL_LENGTH:
            raise _validation_error(
                "budget_exceeded",
                f"generator labels have at most {MAX_OPAQUE_LABEL_LENGTH} characters",
            )
        labelled_generators.append((label, generator))
    normalized["generators"] = tuple(
        generator
        for _, generator in sorted(labelled_generators, key=lambda item: item[0])
    )
    return normalized


def _canonicalize_coordinate_axis(data: dict[str, object]) -> dict[str, object]:
    """Bound arbitrary coordinate-axis iterables before Pydantic materializes them."""

    normalized = dict(data)
    axis = normalized.get("coordinate_axis")
    if isinstance(axis, (str, bytes, Mapping)) or axis is None:
        return normalized
    if not isinstance(axis, (list, tuple)):
        try:
            axis_iterator = iter(cast(Iterable[object], axis))
        except TypeError:
            return normalized
        axis_values: list[object] = []
        for _index in range(MAX_ACTION_DIMENSION + 1):
            try:
                axis_values.append(next(axis_iterator))
            except StopIteration:
                break
        else:
            raise _validation_error(
                "budget_exceeded",
                f"coordinate_axis has at most {MAX_ACTION_DIMENSION} labels",
            )
        axis = tuple(axis_values)
        normalized["coordinate_axis"] = axis
    if isinstance(axis, (list, tuple)):
        if len(axis) > MAX_ACTION_DIMENSION:
            raise _validation_error(
                "budget_exceeded",
                f"coordinate_axis has at most {MAX_ACTION_DIMENSION} labels",
            )
        for label in axis:
            if not isinstance(label, str):
                raise _validation_error(
                    "invalid_coordinate_label",
                    "coordinate_axis labels must be strings",
                )
            if unicodedata.normalize("NFC", label) != label:
                raise _validation_error(
                    "noncanonical_coordinate_label",
                    "coordinate_axis labels must use NFC Unicode normalization",
                )
        normalized["coordinate_axis"] = canonicalize_json_containers(axis)
    return normalized


def _reject_nested_rational_components(entries: object) -> None:
    """Reject non-string ``num``/``den`` values before recursive parsing.

    A deeply nested dict or list inside a recognized rational component
    would pass the ``num``/``den`` key check but cause a ``RecursionError``
    during ``canonicalize_json_containers``.  Reject it here instead.
    """

    if not isinstance(entries, (list, tuple)):
        return
    for row in entries:
        if not isinstance(row, (list, tuple)):
            continue
        for cell in row:
            if not isinstance(cell, dict):
                continue
            for key in ("num", "den"):
                value = cell.get(key)
                if value is not None and not isinstance(value, str):
                    raise _validation_error(
                        "rational_component",
                        f"rational {key} must be a string, not {type(value).__name__}",
                    )


def _require_raw_action_envelope(data: object) -> object:  # noqa: C901
    """Bound structural action containers before nested rational parsing."""

    if not isinstance(data, dict):
        return canonicalize_json_containers(data)
    axis = data.get("coordinate_axis")
    if isinstance(axis, (list, tuple)) and len(axis) > MAX_ACTION_DIMENSION:
        raise _validation_error(
            "budget_exceeded",
            f"coordinate_axis has at most {MAX_ACTION_DIMENSION} labels",
        )
    generators = data.get("generators")
    if not isinstance(generators, (list, tuple)):
        data = _canonicalize_generator_order(data)
        generators = data.get("generators")
        if not isinstance(generators, (list, tuple)):
            return data
    if len(generators) > MAX_ACTION_GENERATORS:
        raise _validation_error(
            "budget_exceeded",
            f"an action has at most {MAX_ACTION_GENERATORS} generators",
        )
    # Inspect each raw generator matrix against the declared axis dimension
    # before Pydantic canonicalizes every rational cell.  A native caller
    # can reuse one raw matrix object across generators, so bound the total
    # cell count from the raw shapes rather than the first dimension alone.
    # Apply an absolute raw-cell ceiling so a valid-shape but over-large
    # action (e.g. 128-axis x 65536 generators) is rejected before Pydantic
    # parses one billion rational cells.
    if isinstance(axis, (list, tuple)) and axis:
        dimension = len(axis)
        total_cells = 0
        raw_digit_work = 0
        for generator in generators:
            if isinstance(generator, dict):
                raw_matrix = generator.get("matrix")
                if not isinstance(raw_matrix, dict):
                    continue
                raw_entries = raw_matrix.get("entries")
                if not isinstance(raw_entries, (list, tuple)):
                    continue
                _reject_nested_rational_components(raw_entries)
                for row in raw_entries:
                    if isinstance(row, (list, tuple)):
                        total_cells += len(row)
                        for cell in row:
                            if isinstance(cell, dict):
                                for comp in (cell.get("num"), cell.get("den")):
                                    if isinstance(comp, str):
                                        raw_digit_work += len(comp.lstrip("-"))
            elif isinstance(generator, RationalActionGenerator):
                # Already a canonical RationalActionGenerator instance.
                total_cells += len(generator.matrix.entries) * (
                    len(generator.matrix.entries[0]) if generator.matrix.entries else 0
                )
        if dimension > 0 and total_cells > 0:
            max_cells = dimension * dimension * len(generators)
            if total_cells > max_cells:
                raise _validation_error(
                    "budget_exceeded",
                    "generator matrix rows exceed the declared coordinate_axis"
                    " dimension before nested parsing",
                )
            if raw_digit_work > MAX_CONSTRAINT_DIGIT_WORK:
                raise _validation_error(
                    "budget_exceeded",
                    "raw rational digit work exceeds the "
                    f"{MAX_CONSTRAINT_DIGIT_WORK}-digit work bound",
                )
    else:
        # When the axis is missing, empty, or not a sequence, still bound the
        # total raw cell count so Pydantic does not validate a billion cells
        # merely to report the axis error.
        total_cells = 0
        raw_digit_work = 0
        for generator in generators:
            if not isinstance(generator, dict):
                continue
            raw_matrix = generator.get("matrix")
            if not isinstance(raw_matrix, dict):
                continue
            raw_entries = raw_matrix.get("entries")
            if not isinstance(raw_entries, (list, tuple)):
                continue
            for row in raw_entries:
                if isinstance(row, (list, tuple)):
                    total_cells += len(row)
        if total_cells > MAX_CONSTRAINT_CELLS:
            raise _validation_error(
                "budget_exceeded",
                "generator matrix cells exceed the structural bound of "
                f"{MAX_CONSTRAINT_CELLS} coefficients before axis validation",
            )
    return _canonicalize_generator_order(data)


def _raw_action_parts(action: object) -> tuple[object, object]:
    if isinstance(action, dict):
        return action.get("coordinate_axis"), action.get("generators")
    return getattr(action, "coordinate_axis", None), getattr(action, "generators", None)


def _require_raw_request_envelope(data: object) -> object:
    """Reject an over-work request before copying or nested scalar parsing."""

    if not isinstance(data, dict):
        return data
    axis, generators = _raw_action_parts(data.get("action"))
    kind = data.get("kind")
    if kind not in ("BILINEAR", "SYMMETRIC", "ALTERNATING"):
        raise _validation_error(
            "invalid_kind",
            "kind must be BILINEAR, SYMMETRIC, or ALTERNATING",
        )
    if (
        isinstance(axis, (list, tuple))
        and isinstance(generators, (list, tuple))
        and kind in ("BILINEAR", "SYMMETRIC", "ALTERNATING")
    ):
        dimension = len(axis)
        constraint_cells = constraint_coefficient_count(
            dimension, len(generators), kind
        )
        if constraint_cells > MAX_CONSTRAINT_CELLS:
            raise _validation_error(
                "budget_exceeded",
                "the congruence expansion exceeds the structural bound of "
                f"{MAX_CONSTRAINT_CELLS} coefficients",
            )
    return data


class RationalActionGenerator(StrictModel):
    """One labelled rational endomorphism in a common ordered basis."""

    label: OpaqueLabel
    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_canonical_label(self) -> Self:
        if unicodedata.normalize("NFC", self.label) != self.label:
            raise _validation_error(
                "noncanonical_generator_label",
                "generator labels must use NFC Unicode normalization",
            )
        return self


class RationalMatrixAction(StrictModel):
    """A finite labelled family of rational square endomorphisms.

    ``coordinate_axis`` is an ordered lattice basis, and every matrix acts on
    its rational span. The matrices need not preserve the integral lattice or
    be invertible: the invariant-form equation remains a well-defined
    congruence fixed-point condition for rational endomorphisms.
    """

    coordinate_axis: tuple[OpaqueLabel, ...] = Field(
        min_length=1, max_length=MAX_ACTION_DIMENSION
    )
    generators: tuple[RationalActionGenerator, ...] = Field(
        default=(), max_length=MAX_ACTION_GENERATORS
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_envelope(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = _canonicalize_coordinate_axis(data)
            generators = data.get("generators")
            if isinstance(generators, (list, tuple)):
                for generator in generators:
                    label = (
                        generator.get("label")
                        if isinstance(generator, dict)
                        else getattr(generator, "label", None)
                    )
                    if (
                        isinstance(label, str)
                        and unicodedata.normalize("NFC", label) != label
                    ):
                        raise _validation_error(
                            "noncanonical_generator_label",
                            "generator labels must use NFC Unicode normalization",
                        )
        return _require_raw_action_envelope(data)

    @model_validator(mode="after")
    def require_common_axis(self) -> Self:
        if any(
            unicodedata.normalize("NFC", label) != label
            for label in self.coordinate_axis
        ):
            raise _validation_error(
                "noncanonical_coordinate_label",
                "coordinate_axis labels must use NFC Unicode normalization",
            )
        if len(set(self.coordinate_axis)) != len(self.coordinate_axis):
            raise _validation_error(
                "duplicate_coordinate_label",
                "coordinate_axis labels must be pairwise distinct",
            )
        if len({generator.label for generator in self.generators}) != len(
            self.generators
        ):
            raise _validation_error(
                "duplicate_generator_label",
                "generator labels must be pairwise distinct",
            )
        dimension = len(self.coordinate_axis)
        for generator in self.generators:
            entries = generator.matrix.entries
            if len(entries) != dimension or any(
                len(row) != dimension for row in entries
            ):
                raise _validation_error(
                    "generator_shape",
                    "every generator matrix must be square on coordinate_axis",
                )
        return self


class IntegralBilinearForm(StrictModel):
    """One integral form matrix on an explicitly ordered coordinate axis."""

    coordinate_axis: tuple[OpaqueLabel, ...] = Field(
        min_length=1, max_length=MAX_ACTION_DIMENSION
    )
    kind: FormKind
    matrix: IntegerMatrix

    @model_validator(mode="before")
    @classmethod
    def require_canonical_axis_labels(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        unknown = set(data).difference({"coordinate_axis", "kind", "matrix"})
        if unknown:
            raise _validation_error(
                "shape_mismatch",
                "forms contain unknown fields",
            )
        axis = data.get("coordinate_axis")
        normalized = dict(data)
        if isinstance(axis, (str, bytes, Mapping)) or axis is None:
            return normalized
        if not isinstance(axis, (list, tuple)):
            try:
                axis_iterator = iter(cast(Iterable[object], axis))
            except TypeError:
                return normalized
            axis_values: list[object] = []
            for _index in range(MAX_ACTION_DIMENSION + 1):
                try:
                    axis_values.append(next(axis_iterator))
                except StopIteration:
                    break
            else:
                raise _validation_error(
                    "budget_exceeded",
                    f"coordinate_axis has at most {MAX_ACTION_DIMENSION} labels",
                )
            axis = tuple(axis_values)
        if len(axis) > MAX_ACTION_DIMENSION:
            raise _validation_error(
                "budget_exceeded",
                f"coordinate_axis has at most {MAX_ACTION_DIMENSION} labels",
            )
        if isinstance(axis, (list, tuple)):
            for label in axis:
                if not isinstance(label, str):
                    raise _validation_error(
                        "invalid_coordinate_label",
                        "form coordinate_axis labels must be strings",
                    )
                if unicodedata.normalize("NFC", label) != label:
                    raise _validation_error(
                        "noncanonical_coordinate_label",
                        "form coordinate_axis labels must use NFC Unicode normalization",
                    )
            normalized["coordinate_axis"] = canonicalize_json_containers(axis)
        return normalized

    @model_validator(mode="after")
    def require_form_shape(self) -> Self:
        dimension = len(self.coordinate_axis)
        if any(
            unicodedata.normalize("NFC", label) != label
            for label in self.coordinate_axis
        ):
            raise _validation_error(
                "noncanonical_coordinate_label",
                "form coordinate_axis labels must use NFC Unicode normalization",
            )
        if len(set(self.coordinate_axis)) != dimension:
            raise _validation_error(
                "duplicate_coordinate_label",
                "form coordinate_axis labels must be pairwise distinct",
            )
        entries = self.matrix.entries
        if len(entries) != dimension or any(len(row) != dimension for row in entries):
            raise _validation_error(
                "form_shape", "form matrix must be square on coordinate_axis"
            )
        if self.kind == "SYMMETRIC" and any(
            entries[row][column] != entries[column][row]
            for row in range(dimension)
            for column in range(row + 1, dimension)
        ):
            raise _validation_error(
                "form_symmetry", "a SYMMETRIC form matrix must equal its transpose"
            )
        if self.kind == "ALTERNATING" and (
            any(
                parse_canonical_integer(entries[index][index]) != 0
                for index in range(dimension)
            )
            or any(
                parse_canonical_integer(entries[row][column])
                != -parse_canonical_integer(entries[column][row])
                for row in range(dimension)
                for column in range(row + 1, dimension)
            )
        ):
            raise _validation_error(
                "form_alternation",
                "an ALTERNATING form matrix must be skew-symmetric with zero diagonal",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        coordinate_axis: tuple[OpaqueLabel, ...],
        kind: FormKind,
        entries: tuple[tuple[str, ...], ...],
    ) -> Self:
        """Construct a form already established by the owner-local kernel."""

        matrix = IntegerMatrix.model_construct(domain="ZZ", entries=entries)
        return cls.model_construct(
            coordinate_axis=coordinate_axis,
            kind=kind,
            matrix=matrix,
        )


class InvariantBilinearFormLatticeRequest(StrictModel):
    """One rational matrix action and one integral form symmetry class."""

    action: RationalMatrixAction = Field(
        description=(
            "Canonical rational endomorphisms on one labelled lattice axis. "
            "Admission requires generator_count * axis_dimension^2 * "
            f"coefficient_dimension <= {MAX_CONSTRAINT_CELLS:,}, couples that "
            "count to source and intermediate digit heights, and separately "
            "bounds exact elimination, normalization, and output size."
        )
    )
    kind: FormKind = Field(
        description=(
            "Coefficient class: all row-major entries for BILINEAR, the upper "
            "triangle for SYMMETRIC, or the strict upper triangle for ALTERNATING."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_operation_envelope(cls, data: Any) -> Any:
        if isinstance(data, dict):
            action = data.get("action")
            if isinstance(action, dict):
                raw_axis = action.get("coordinate_axis")
                if isinstance(raw_axis, (list, tuple)) and len(
                    raw_axis
                ) > MAX_ACTION_DIMENSION:
                    raise _validation_error(
                        "budget_exceeded",
                        f"coordinate_axis has at most {MAX_ACTION_DIMENSION} labels",
                    )
            if isinstance(action, dict) and isinstance(
                action.get("coordinate_axis"), (list, tuple)
            ):
                axis_labels = action["coordinate_axis"]
                for label in axis_labels:
                    if not isinstance(label, str):
                        raise _validation_error(
                            "invalid_coordinate_label",
                            "coordinate_axis labels must be strings",
                        )
                data = dict(data)
                action = dict(action)
                action["coordinate_axis"] = canonicalize_json_containers(
                    action["coordinate_axis"]
                )
                data["action"] = action
        return _require_raw_request_envelope(data)


class InvariantBilinearFormLattice(StrictModel):
    """The complete lattice of integral forms fixed by a rational action.

    Coefficients are read in the declared canonical order. ``basis_forms`` is
    the row-Hermite basis of the saturated integer kernel of all equations
    ``A^T Q A = Q``. It is empty exactly when the invariant lattice has rank
    zero, while ``coefficient_dimension`` retains the ambient coefficient
    lattice.
    """

    action: RationalMatrixAction
    kind: FormKind
    coefficient_domain: Literal["ZZ"] = "ZZ"
    coefficient_order: FormCoefficientOrder
    coefficient_dimension: int = Field(ge=0, le=MAX_FORM_COEFFICIENT_DIMENSION)
    constraint_rank: int = Field(ge=0, le=MAX_FORM_COEFFICIENT_DIMENSION)
    rank: int = Field(ge=0, le=MAX_FORM_COEFFICIENT_DIMENSION)
    basis_forms: tuple[IntegralBilinearForm, ...] = Field(
        default=(), max_length=MAX_FORM_COEFFICIENT_DIMENSION
    )
    invariance_relation: Literal[
        "GENERATOR_TRANSPOSE_TIMES_FORM_TIMES_GENERATOR_EQUALS_FORM"
    ] = "GENERATOR_TRANSPOSE_TIMES_FORM_TIMES_GENERATOR_EQUALS_FORM"
    basis_normalization: Literal["SATURATED_ROW_HERMITE_NORMAL_FORM"] = (
        "SATURATED_ROW_HERMITE_NORMAL_FORM"
    )

    @model_validator(mode="after")
    def require_source_bound_shape(self) -> Self:
        expected_dimension = coefficient_dimension(
            len(self.action.coordinate_axis), self.kind
        )
        if self.coefficient_dimension != expected_dimension:
            raise _validation_error(
                "coefficient_dimension",
                "coefficient_dimension does not match the action axis and form kind",
            )
        if self.coefficient_order != coefficient_order(self.kind):
            raise _validation_error(
                "coefficient_order",
                "coefficient_order does not match the form kind",
            )
        if self.rank + self.constraint_rank != self.coefficient_dimension:
            raise _validation_error(
                "rank_nullity",
                "rank plus constraint_rank must equal coefficient_dimension",
            )
        if len(self.basis_forms) != self.rank:
            raise _validation_error("basis_rank", "basis_forms length must equal rank")
        if any(
            form.coordinate_axis != self.action.coordinate_axis
            or form.kind != self.kind
            for form in self.basis_forms
        ):
            raise _validation_error(
                "basis_source",
                "every basis form must use the action axis and requested form kind",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        action: RationalMatrixAction,
        kind: FormKind,
        coefficient_dimension: int,
        constraint_rank: int,
        basis_forms: tuple[IntegralBilinearForm, ...],
    ) -> Self:
        """Construct a source-bound result established by the exact kernel."""

        return cls.model_construct(
            action=action,
            kind=kind,
            coefficient_domain="ZZ",
            coefficient_order=coefficient_order(kind),
            coefficient_dimension=coefficient_dimension,
            constraint_rank=constraint_rank,
            rank=len(basis_forms),
            basis_forms=basis_forms,
            invariance_relation=(
                "GENERATOR_TRANSPOSE_TIMES_FORM_TIMES_GENERATOR_EQUALS_FORM"
            ),
            basis_normalization="SATURATED_ROW_HERMITE_NORMAL_FORM",
        )


__all__ = [
    "MAX_ACTION_DIMENSION",
    "MAX_ACTION_GENERATORS",
    "MAX_CONSTRAINT_CELLS",
    "MAX_FORM_COEFFICIENT_DIMENSION",
    "FormCoefficientOrder",
    "FormKind",
    "IntegralBilinearForm",
    "InvariantBilinearFormLattice",
    "InvariantBilinearFormLatticeRequest",
    "RationalActionGenerator",
    "RationalMatrixAction",
]
