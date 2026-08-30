"""Canonical values for lattices of invariant integral bilinear forms."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import parse_canonical_integer
from jacobian.math._labels import OpaqueLabel
from jacobian.math.matrices.values import (
    MAX_RATIONAL_MATRIX_ORDER,
    EmbeddedRealSimpleNumberFieldMatrix,
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
    if not isinstance(normalized_generators, (list, tuple)):
        return normalized
    labelled_generators: list[tuple[str, object]] = []
    for generator in normalized_generators:
        label = (
            generator.get("label")
            if isinstance(generator, dict)
            else getattr(generator, "label", None)
        )
        if not isinstance(label, str):
            return normalized
        labelled_generators.append((label, generator))
    normalized["generators"] = tuple(
        generator
        for _, generator in sorted(labelled_generators, key=lambda item: item[0])
    )
    return normalized


def _require_raw_action_envelope(data: object) -> object:
    """Bound structural action containers before nested rational parsing."""

    if not isinstance(data, dict):
        return data
    axis = data.get("coordinate_axis")
    if isinstance(axis, (list, tuple)) and len(axis) > MAX_ACTION_DIMENSION:
        raise _validation_error(
            "budget_exceeded",
            f"coordinate_axis has at most {MAX_ACTION_DIMENSION} labels",
        )
    generators = data.get("generators")
    if not isinstance(generators, (list, tuple)):
        return _canonicalize_generator_order(data)
    if len(generators) > MAX_ACTION_GENERATORS:
        raise _validation_error(
            "budget_exceeded",
            f"an action has at most {MAX_ACTION_GENERATORS} generators",
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
        field_degree = 1
        if generators:
            first_generator = generators[0]
            first_matrix = (
                first_generator.get("matrix")
                if isinstance(first_generator, dict)
                else getattr(first_generator, "matrix", None)
            )
            if isinstance(first_matrix, dict) and first_matrix.get("domain") == (
                "EMBEDDED_REAL_SIMPLE_NUMBER_FIELD"
            ):
                embedding = first_matrix.get("embedding")
                presentation = (
                    embedding.get("presentation")
                    if isinstance(embedding, dict)
                    else None
                )
                coefficients = (
                    presentation.get("coefficients_descending")
                    if isinstance(presentation, dict)
                    else None
                )
                if isinstance(coefficients, (list, tuple)) and len(coefficients) >= 2:
                    field_degree = len(coefficients) - 1
        constraint_cells = (
            constraint_coefficient_count(dimension, len(generators), kind)
            * field_degree
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
        if isinstance(data, dict) and isinstance(
            data.get("coordinate_axis"), (list, tuple)
        ):
            data = dict(data)
            data["coordinate_axis"] = canonicalize_json_containers(
                data["coordinate_axis"]
            )
        return _require_raw_action_envelope(data)

    @model_validator(mode="after")
    def require_common_axis(self) -> Self:
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


class EmbeddedRealNumberFieldActionGenerator(StrictModel):
    """One labelled real-linear map over a selected simple-field embedding."""

    label: OpaqueLabel
    matrix: EmbeddedRealSimpleNumberFieldMatrix


class EmbeddedRealNumberFieldMatrixAction(StrictModel):
    """A homogeneous family of matrices over one embedded simple number field."""

    coordinate_axis: tuple[OpaqueLabel, ...] = Field(
        min_length=1, max_length=MAX_ACTION_DIMENSION
    )
    generators: tuple[EmbeddedRealNumberFieldActionGenerator, ...] = Field(
        min_length=1,
        max_length=MAX_ACTION_GENERATORS,
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_envelope(cls, data: Any) -> Any:
        return canonicalize_json_containers(_require_raw_action_envelope(data))

    @model_validator(mode="after")
    def require_common_axis_and_embedding(self) -> Self:
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
        embedding = self.generators[0].matrix.embedding
        for generator in self.generators:
            entries = generator.matrix.entries
            if len(entries) != dimension or any(
                len(row) != dimension for row in entries
            ):
                raise _validation_error(
                    "generator_shape",
                    "every generator matrix must be square on coordinate_axis",
                )
            if generator.matrix.embedding != embedding:
                raise _validation_error(
                    "generator_embedding",
                    "every algebraic generator must use one common real embedding",
                )
        return self


MatrixAction = RationalMatrixAction | EmbeddedRealNumberFieldMatrixAction


class IntegralBilinearForm(StrictModel):
    """One integral form matrix on an explicitly ordered coordinate axis."""

    coordinate_axis: tuple[OpaqueLabel, ...] = Field(
        min_length=1, max_length=MAX_ACTION_DIMENSION
    )
    kind: FormKind
    matrix: IntegerMatrix

    @model_validator(mode="before")
    @classmethod
    def require_raw_form_envelope(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        axis = data.get("coordinate_axis")
        if isinstance(axis, (list, tuple)) and len(axis) > MAX_ACTION_DIMENSION:
            raise _validation_error(
                "budget_exceeded",
                f"form coordinate_axis has at most {MAX_ACTION_DIMENSION} labels",
            )
        normalized = dict(data)
        if isinstance(axis, list):
            normalized["coordinate_axis"] = tuple(axis)
        return canonicalize_json_containers(normalized)

    @model_validator(mode="after")
    def require_form_shape(self) -> Self:
        dimension = len(self.coordinate_axis)
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
    """One exact real matrix action and one integral form symmetry class."""

    action: MatrixAction = Field(
        description=(
            "Canonical rational or common-embedding algebraic matrices on one "
            "labelled lattice axis. "
            "Admission requires generator_count * axis_dimension^2 * "
            "coefficient_dimension * field_degree <= "
            f"{MAX_CONSTRAINT_CELLS:,}, couples that count to source and "
            "intermediate digit heights, and separately bounds exact elimination, "
            "normalization, and output size."
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
            if isinstance(action, dict) and isinstance(
                action.get("coordinate_axis"), (list, tuple)
            ):
                data = dict(data)
                action = dict(action)
                action["coordinate_axis"] = canonicalize_json_containers(
                    action["coordinate_axis"]
                )
                data["action"] = action
        return _require_raw_request_envelope(data)


class InvariantBilinearFormLattice(StrictModel):
    """The complete lattice of integral forms fixed by an exact real action.

    Coefficients are read in the declared canonical order. ``basis_forms`` is
    the row-Hermite basis of the saturated integer kernel of all equations
    ``A^T Q A = Q``. It is empty exactly when the invariant lattice has rank
    zero, while ``coefficient_dimension`` retains the ambient coefficient
    lattice.
    """

    action: MatrixAction
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
        action: MatrixAction,
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
    "EmbeddedRealNumberFieldActionGenerator",
    "EmbeddedRealNumberFieldMatrixAction",
    "FormCoefficientOrder",
    "FormKind",
    "IntegralBilinearForm",
    "InvariantBilinearFormLattice",
    "InvariantBilinearFormLatticeRequest",
    "MatrixAction",
    "RationalActionGenerator",
    "RationalMatrixAction",
]
