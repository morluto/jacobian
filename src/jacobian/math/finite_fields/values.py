"""Provider-independent values for exact finite-field linear algebra."""

from __future__ import annotations

from typing import Any, Literal, Self

import rfc8785
from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import sha256_digest
from jacobian.math.graphs.directed._models import DirectedGraph
from jacobian.math.matrices.finite_fields._bounds import (
    MAX_PRIME_FIELD_FLINT_PRIME,
    MAX_PRIME_FIELD_MATRIX_AXIS,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix, rank

_MAX_FIELD_ORDER = 65536
_MIN_MODULUS_COEFFICIENTS = 2
_MAX_MODULUS_COEFFICIENTS = 17
_MAX_VALUE_AXIS_LABELS = 256
_MAX_DERIVATION_WORK = 1_000_000
_MAX_ACTION_GENERATORS = MAX_PRIME_FIELD_MATRIX_AXIS
# The matrix carrier and the fixed-subspace operation both cap the ambient
# homogeneous basis at one matrix axis.  Work and output admission remain
# result-sensitive in the operation owner.
_MAX_HOMOGENEOUS_MONOMIALS = MAX_PRIME_FIELD_MATRIX_AXIS


def _homogeneous_monomial_count(variable_count: int, degree: int) -> int:
    """Count homogeneous monomials, stopping once the result is out of budget."""

    count = 1
    for position in range(1, variable_count):
        count = count * (degree + position) // position
        if count > _MAX_HOMOGENEOUS_MONOMIALS:
            return count
    return count


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


def _digest(payload: dict[str, Any]) -> str:
    return sha256_digest(rfc8785.dumps(payload))


def _encoded_coordinates(value: FiniteFieldElement) -> int:
    return sum(
        coordinate * value.presentation.characteristic**power
        for power, coordinate in enumerate(value.coordinates)
    )


def _validate_presentation_shape(
    characteristic: int,
    modulus_coefficients: tuple[int, ...],
    generator: str,
) -> None:
    if type(characteristic) is not int or characteristic < 2:
        raise _validation_error(
            "finite_field.characteristic_prime_integer",
            "characteristic must be a prime integer",
        )
    if characteristic > _MAX_FIELD_ORDER:
        raise _validation_error(
            "finite_field.characteristic_exceeds_supported_field_order_bound",
            "characteristic exceeds the supported field-order bound",
        )
    if not (
        _MIN_MODULUS_COEFFICIENTS
        <= len(modulus_coefficients)
        <= _MAX_MODULUS_COEFFICIENTS
    ):
        raise _validation_error(
            "finite_field.presentation_modulus_length_bound",
            "finite-field presentation modulus length is outside its bound",
        )
    if modulus_coefficients[-1] != 1:
        raise _validation_error("finite_field.modulus_monic", "modulus must be monic")
    if any(
        type(value) is not int or not 0 <= value < characteristic
        for value in modulus_coefficients
    ):
        raise _validation_error(
            "finite_field.modulus_coefficients_canonical_field_residues",
            "modulus coefficients must be canonical field residues",
        )
    if not generator:
        raise _validation_error(
            "finite_field.generator_nonempty", "generator must be nonempty"
        )


class FiniteFieldPresentation(StrictModel):
    """An exact polynomial presentation with a fixed power-basis encoding."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "characteristic": 2,
                    "modulus_coefficients": [1, 1, 1],
                    "generator": "a",
                }
            ]
        }
    )

    characteristic: int = Field(
        description="Prime p defining the base field GF(p).", examples=[2]
    )
    modulus_coefficients: tuple[int, ...] = Field(
        description=(
            "Constant-to-leading coefficients of a monic irreducible modulus over "
            "GF(characteristic); each coefficient is a canonical residue."
        ),
        examples=[[1, 1, 1]],
    )
    generator: str = Field(
        default="a",
        description="Name of the power-basis generator represented by the modulus.",
        examples=["a"],
    )

    @model_validator(mode="after")
    def validate_presentation(self) -> Self:
        _validate_presentation_shape(
            self.characteristic,
            self.modulus_coefficients,
            self.generator,
        )
        if self.characteristic**self.degree > _MAX_FIELD_ORDER:
            raise _validation_error(
                "finite_field.field_order_exceeds_supported_bound",
                "field order exceeds the supported bound",
            )
        from sympy import Poly, isprime, symbols

        if not isprime(self.characteristic):
            raise _validation_error(
                "finite_field.characteristic_prime_integer",
                "characteristic must be a prime integer",
            )
        variable = symbols("x")
        polynomial = Poly(
            sum(
                coefficient * variable**power
                for power, coefficient in enumerate(self.modulus_coefficients)
            ),
            variable,
            modulus=self.characteristic,
        )
        if not polynomial.is_irreducible:
            raise _validation_error(
                "finite_field.modulus_irreducible_over_prime_field",
                "modulus must be irreducible over the prime field",
            )
        return self

    @property
    def degree(self) -> int:
        return len(self.modulus_coefficients) - 1

    @property
    def order(self) -> int:
        return int(pow(self.characteristic, self.degree))

    @property
    def ordered_basis(self) -> tuple[str, ...]:
        return (
            "1",
            *(
                self.generator if power == 1 else f"{self.generator}^{power}"
                for power in range(1, self.degree)
            ),
        )

    @property
    def digest(self) -> str:
        return _digest(
            {
                "characteristic": self.characteristic,
                "generator": self.generator,
                "modulus_coefficients": list(self.modulus_coefficients),
                "ordered_basis": list(self.ordered_basis),
                "value_type": "finite-field-presentation",
            }
        )


class FiniteFieldElement(StrictModel):
    """Power-basis coordinates bound to one exact field presentation."""

    presentation: FiniteFieldPresentation
    coordinates: tuple[int, ...]

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        if len(self.coordinates) != self.presentation.degree:
            raise _validation_error(
                "finite_field.element_coordinates_match_presentation_degree",
                "element coordinates must match the presentation degree",
            )
        if any(
            type(value) is not int or not 0 <= value < self.presentation.characteristic
            for value in self.coordinates
        ):
            raise _validation_error(
                "finite_field.element_coordinates_canonical_field_residues",
                "element coordinates must be canonical field residues",
            )
        return self

    @property
    def is_zero(self) -> bool:
        return not any(self.coordinates)

    @property
    def is_one(self) -> bool:
        return self.coordinates == (1,) + (0,) * (self.presentation.degree - 1)

    @property
    def digest(self) -> str:
        return _digest(
            {
                "coordinates": list(self.coordinates),
                "presentation": self.presentation.digest,
                "value_type": "finite-field-element",
            }
        )


PaleyTournamentOrientation = Literal["ARC_X_TO_Y_IFF_Y_MINUS_X_IS_NONZERO_SQUARE"]


class PaleyTournamentResult(StrictModel):
    """The directed Paley tournament bound to its exact field presentation."""

    presentation: FiniteFieldPresentation
    graph: DirectedGraph
    orientation: PaleyTournamentOrientation = (
        "ARC_X_TO_Y_IFF_Y_MINUS_X_IS_NONZERO_SQUARE"
    )

    @model_validator(mode="after")
    def bind_graph_to_presentation(self) -> Self:
        if self.presentation.order % 4 != 3:
            raise _validation_error(
                "finite_field.paley_tournament_order_congruent_to_three_mod_four",
                "Paley tournament requires field order congruent to 3 modulo 4",
            )
        if self.graph.vertex_count != self.presentation.order:
            raise _validation_error(
                "finite_field.paley_tournament_vertex_count_matches_field_order",
                "Paley tournament vertex count must equal the field order",
            )
        expected_edges = self.presentation.order * (self.presentation.order - 1) // 2
        if len(self.graph.edges) != expected_edges:
            raise _validation_error(
                "finite_field.paley_tournament_complete_arc_count",
                "Paley tournament must contain one arc per unordered vertex pair",
            )
        if self.graph.edges != tuple(sorted(self.graph.edges)):
            raise _validation_error(
                "finite_field.paley_tournament_lexicographic_arcs",
                "Paley tournament arcs must be lexicographically ordered",
            )
        unordered_pairs = {tuple(sorted(edge)) for edge in self.graph.edges}
        if len(unordered_pairs) != expected_edges:
            raise _validation_error(
                "finite_field.paley_tournament_one_orientation_per_vertex_pair",
                "Paley tournament must orient every unordered vertex pair exactly once",
            )
        return self


class Axis(StrictModel):
    """An ordered semantic axis."""

    name: str
    labels: tuple[str, ...]

    @model_validator(mode="after")
    def validate_axis(self) -> Self:
        if not self.name:
            raise _validation_error(
                "finite_field.axis_name_nonempty", "axis name must be nonempty"
            )
        if not self.labels or any(not label for label in self.labels):
            raise _validation_error(
                "finite_field.axis_labels_nonempty", "axis labels must be nonempty"
            )
        if len(self.labels) > _MAX_VALUE_AXIS_LABELS:
            raise _validation_error(
                "finite_field.axis_exceeds_supported_label_bound",
                "axis exceeds the supported label bound",
            )
        if len(set(self.labels)) != len(self.labels):
            raise _validation_error(
                "finite_field.axis_labels_unique", "axis labels must be unique"
            )
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "labels": list(self.labels),
                "name": self.name,
                "value_type": "axis",
            }
        )


def _raw_field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


class PrimeFieldLinearAction(StrictModel):
    """Explicit generator substitutions on an ordered polynomial-variable axis.

    Matrix column ``j`` gives the coefficients of the image of variable
    ``j``. Invertibility is recognized once by operations that consume this
    structurally canonical source; it is not replayed while decoding results.
    """

    variable_axis: Axis
    generator_matrices: tuple[PrimeFieldMatrix, ...] = Field(
        min_length=1, max_length=_MAX_ACTION_GENERATORS
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_action_envelope(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        axis = data.get("variable_axis")
        labels = _raw_field(axis, "labels")
        matrices = data.get("generator_matrices")
        if not isinstance(labels, (list, tuple)) or not isinstance(
            matrices, (list, tuple)
        ):
            return data
        variable_count = len(labels)
        if variable_count < 1:
            raise _validation_error(
                "finite_field.linear_action_variable_bound",
                "linear action exceeds the variable-count bound",
            )
        if not 1 <= len(matrices) <= _MAX_ACTION_GENERATORS:
            raise _validation_error(
                "finite_field.linear_action_generator_bound",
                "linear action exceeds the generator-count bound",
            )
        for matrix in matrices:
            prime = _raw_field(matrix, "prime")
            columns = _raw_field(matrix, "columns")
            entries = _raw_field(matrix, "entries")
            if type(prime) is int and not 2 <= prime <= MAX_PRIME_FIELD_FLINT_PRIME:
                raise _validation_error(
                    "finite_field.linear_action_prime_bound",
                    "linear action prime exceeds the word-safe backend bound",
                )
            if type(columns) is int and columns != variable_count:
                raise _validation_error(
                    "finite_field.linear_action_matrix_shape",
                    "every action matrix must be square on the variable axis",
                )
            if isinstance(entries, (list, tuple)) and (
                len(entries) != variable_count
                or any(
                    isinstance(row, (list, tuple)) and len(row) != variable_count
                    for row in entries
                )
            ):
                raise _validation_error(
                    "finite_field.linear_action_matrix_shape",
                    "every action matrix must be square on the variable axis",
                )
        return data

    @model_validator(mode="after")
    def require_common_prime_and_axes(self) -> Self:
        variable_count = len(self.variable_axis.labels)
        prime = self.generator_matrices[0].prime
        if any(
            matrix.prime != prime
            or matrix.columns != variable_count
            or len(matrix.entries) != variable_count
            for matrix in self.generator_matrices
        ):
            raise _validation_error(
                "finite_field.linear_action_common_parent",
                "action matrices must be square on one variable axis over one prime",
            )
        signatures = tuple(matrix.entries for matrix in self.generator_matrices)
        if len(set(signatures)) != len(signatures):
            raise _validation_error(
                "finite_field.linear_action_distinct_generators",
                "action generator matrices must be distinct",
            )
        return self

    @property
    def prime(self) -> int:
        return int(self.generator_matrices[0].prime)


class HomogeneousFixedSubspace(StrictModel):
    """One homogeneous simultaneous fixed space in canonical coefficient form."""

    action: PrimeFieldLinearAction
    degree: StrictInt = Field(ge=0)
    monomial_basis: tuple[tuple[StrictInt, ...], ...] = Field(
        min_length=1, max_length=_MAX_HOMOGENEOUS_MONOMIALS
    )
    basis_matrix: PrimeFieldMatrix
    fixed_dimension: StrictInt = Field(ge=0, le=_MAX_HOMOGENEOUS_MONOMIALS)

    @model_validator(mode="before")
    @classmethod
    def require_raw_result_envelope(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        action = data.get("action")
        degree = data.get("degree")
        axis = _raw_field(action, "variable_axis")
        labels = _raw_field(axis, "labels")
        if type(degree) is not int or not isinstance(labels, (list, tuple)):
            return data
        if degree < 0:
            return data
        variable_count = len(labels)
        if variable_count < 1:
            return data
        monomial_count = _homogeneous_monomial_count(variable_count, degree)
        if monomial_count > _MAX_HOMOGENEOUS_MONOMIALS:
            raise _validation_error(
                "finite_field.fixed_subspace_monomial_bound",
                "homogeneous monomial basis exceeds the operation bound",
            )
        raw_basis = data.get("monomial_basis")
        if isinstance(raw_basis, (list, tuple)) and len(raw_basis) != monomial_count:
            raise _validation_error(
                "finite_field.fixed_subspace_monomial_shape",
                "monomial basis length must match the source degree and variable axis",
            )
        matrix = data.get("basis_matrix")
        action_matrices = _raw_field(action, "generator_matrices")
        action_prime = None
        if isinstance(action_matrices, (list, tuple)) and action_matrices:
            action_prime = _raw_field(action_matrices[0], "prime")
        matrix_prime = _raw_field(matrix, "prime")
        columns = _raw_field(matrix, "columns")
        entries = _raw_field(matrix, "entries")
        if (
            type(matrix_prime) is int
            and type(action_prime) is int
            and matrix_prime != action_prime
        ):
            raise _validation_error(
                "finite_field.fixed_subspace_basis_parent",
                "fixed-subspace basis must use the action prime",
            )
        if type(matrix_prime) is int and not (
            2 <= matrix_prime <= MAX_PRIME_FIELD_FLINT_PRIME
        ):
            raise _validation_error(
                "finite_field.fixed_subspace_prime_bound",
                "fixed-subspace basis prime exceeds the word-safe backend bound",
            )
        if type(columns) is int and columns != monomial_count:
            raise _validation_error(
                "finite_field.fixed_subspace_matrix_shape",
                "fixed-subspace basis columns must match the monomial basis",
            )
        if isinstance(entries, (list, tuple)) and len(entries) > monomial_count:
            raise _validation_error(
                "finite_field.fixed_subspace_matrix_shape",
                "fixed-subspace basis rows exceed the ambient dimension",
            )
        return data

    @model_validator(mode="after")
    def require_canonical_shape(self) -> Self:
        variable_count = len(self.action.variable_axis.labels)
        monomial_count = _homogeneous_monomial_count(variable_count, self.degree)
        if len(self.monomial_basis) != monomial_count:
            raise _validation_error(
                "finite_field.fixed_subspace_monomial_shape",
                "monomial basis length must match the source degree and variable axis",
            )
        if any(
            len(exponents) != variable_count
            or any(exponent < 0 for exponent in exponents)
            or sum(exponents) != self.degree
            for exponents in self.monomial_basis
        ):
            raise _validation_error(
                "finite_field.fixed_subspace_monomial_degree",
                "every monomial exponent vector must have the declared homogeneous degree",
            )
        if self.monomial_basis != tuple(sorted(set(self.monomial_basis), reverse=True)):
            raise _validation_error(
                "finite_field.fixed_subspace_monomial_order",
                "monomial basis must use unique descending lexicographic order",
            )
        if (
            self.basis_matrix.prime != self.action.prime
            or self.basis_matrix.columns != monomial_count
            or len(self.basis_matrix.entries) != self.fixed_dimension
        ):
            raise _validation_error(
                "finite_field.fixed_subspace_basis_shape",
                "basis matrix must use the action field and declared dimensions",
            )
        pivots: list[int] = []
        for row in self.basis_matrix.entries:
            pivot = next((index for index, value in enumerate(row) if value), None)
            if pivot is None or row[pivot] != 1:
                raise _validation_error(
                    "finite_field.fixed_subspace_rref_shape",
                    "fixed-subspace basis rows must have normalized pivots",
                )
            pivots.append(pivot)
        if pivots != sorted(set(pivots)) or any(
            row[pivot] != int(row_index == pivot_index)
            for row_index, row in enumerate(self.basis_matrix.entries)
            for pivot_index, pivot in enumerate(pivots)
        ):
            raise _validation_error(
                "finite_field.fixed_subspace_rref_shape",
                "fixed-subspace basis must have canonical reduced pivot columns",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        action: PrimeFieldLinearAction,
        degree: int,
        monomial_basis: tuple[tuple[int, ...], ...],
        basis_matrix: PrimeFieldMatrix,
    ) -> Self:
        return cls.model_construct(
            action=action,
            degree=degree,
            monomial_basis=monomial_basis,
            basis_matrix=basis_matrix,
            fixed_dimension=len(basis_matrix.entries),
        )


class AxisBoundMatrix(StrictModel):
    """An immutable matrix bound to a field presentation and ordered axes."""

    @model_validator(mode="before")
    @classmethod
    def require_legacy_axis_bound(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for field in ("row_axis", "column_axis"):
            axis = data.get(field)
            labels = _raw_field(axis, "labels")
            if (
                isinstance(labels, (list, tuple))
                and len(labels) > _MAX_VALUE_AXIS_LABELS
            ):
                raise _validation_error(
                    "finite_field.axis_exceeds_supported_label_bound",
                    "matrix axes exceed the supported label bound",
                )
        return data

    presentation: FiniteFieldPresentation
    row_axis: Axis
    column_axis: Axis
    entries: tuple[tuple[FiniteFieldElement, ...], ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if len(self.entries) != len(self.row_axis.labels):
            raise _validation_error(
                "finite_field.matrix_rows_match_row_axis",
                "matrix rows must match the row axis",
            )
        if any(len(row) != len(self.column_axis.labels) for row in self.entries):
            raise _validation_error(
                "finite_field.matrix_columns_match_column_axis",
                "matrix columns must match the column axis",
            )
        if any(
            element.presentation != self.presentation
            for row in self.entries
            for element in row
        ):
            raise _validation_error(
                "finite_field.matrix_entries_matrix_field_presentation",
                "matrix entries must use the matrix field presentation",
            )
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "column_axis": self.column_axis.digest,
                "entries": [
                    [list(element.coordinates) for element in row]
                    for row in self.entries
                ],
                "presentation": self.presentation.digest,
                "row_axis": self.row_axis.digest,
                "value_type": "axis-bound-matrix",
            }
        )


class FiniteDimensionalSubspace(StrictModel):
    """An ordered independent matrix basis over the presentation's prime field."""

    presentation: FiniteFieldPresentation
    basis_axis: Axis
    basis: tuple[AxisBoundMatrix, ...]

    @model_validator(mode="before")
    @classmethod
    def require_basis_axis_bound(cls, data: Any) -> Any:
        if isinstance(data, dict):
            axis = data.get("basis_axis")
            labels = _raw_field(axis, "labels")
            if (
                isinstance(labels, (list, tuple))
                and len(labels) > _MAX_VALUE_AXIS_LABELS
            ):
                raise _validation_error(
                    "finite_field.axis_exceeds_supported_label_bound",
                    "subspace basis axes exceed the supported label bound",
                )
        return data

    @model_validator(mode="after")
    def validate_subspace(self) -> Self:
        if len(self.basis) != len(self.basis_axis.labels):
            raise _validation_error(
                "finite_field.subspace_basis_match_basis_axis",
                "subspace basis must match its basis axis",
            )
        if not self.basis:
            raise _validation_error(
                "finite_field.subspace_basis_nonempty",
                "subspace basis must be nonempty",
            )
        first = self.basis[0]
        if any(
            matrix.presentation != self.presentation
            or matrix.row_axis != first.row_axis
            or matrix.column_axis != first.column_axis
            for matrix in self.basis
        ):
            raise _validation_error(
                "finite_field.subspace_matrices_share_parent_axes",
                "subspace matrices must share their parent and axes",
            )
        flattened_dimension = (
            len(first.row_axis.labels)
            * len(first.column_axis.labels)
            * self.presentation.degree
        )
        if flattened_dimension * len(self.basis) > _MAX_VALUE_AXIS_LABELS**2:
            raise _validation_error(
                "finite_field.subspace_rank_matrix_exceeds_supported_bound",
                "subspace rank matrix exceeds its supported bound",
            )
        flattened = tuple(
            tuple(
                coordinate
                for row in matrix.entries
                for element in row
                for coordinate in element.coordinates
            )
            for matrix in self.basis
        )
        coordinate_rows = tuple(zip(*flattened, strict=True))
        basis_matrix = PrimeFieldMatrix(
            prime=self.presentation.characteristic,
            entries=coordinate_rows,
            columns=len(self.basis),
        )
        if rank(basis_matrix) != len(self.basis):
            raise _validation_error(
                "finite_field.subspace_basis_matrices_linearly_independent",
                "subspace basis matrices must be linearly independent",
            )
        return self

    @property
    def row_axis(self) -> Axis:
        return self.basis[0].row_axis

    @property
    def column_axis(self) -> Axis:
        return self.basis[0].column_axis

    @property
    def digest(self) -> str:
        return _digest(
            {
                "basis": [matrix.digest for matrix in self.basis],
                "basis_axis": self.basis_axis.digest,
                "presentation": self.presentation.digest,
                "value_type": "finite-dimensional-subspace",
            }
        )


class ProjectivePoint(StrictModel):
    """A normalized projective point over one field and coordinate axis."""

    presentation: FiniteFieldPresentation
    axis: Axis
    coordinates: tuple[FiniteFieldElement, ...]

    @model_validator(mode="after")
    def validate_point(self) -> Self:
        if len(self.coordinates) != len(self.axis.labels):
            raise _validation_error(
                "finite_field.projective_coordinates_match_axis",
                "projective coordinates must match their axis",
            )
        if any(
            coordinate.presentation != self.presentation
            for coordinate in self.coordinates
        ):
            raise _validation_error(
                "finite_field.projective_coordinates_share_presentation",
                "projective coordinates must share their presentation",
            )
        first_nonzero = next(
            (coordinate for coordinate in self.coordinates if not coordinate.is_zero),
            None,
        )
        if first_nonzero is None:
            raise _validation_error(
                "finite_field.projective_coordinates_zero",
                "projective coordinates cannot all be zero",
            )
        if not first_nonzero.is_one:
            raise _validation_error(
                "finite_field.projective_coordinates_normalized",
                "projective coordinates must be normalized",
            )
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "axis": self.axis.digest,
                "coordinates": [list(value.coordinates) for value in self.coordinates],
                "presentation": self.presentation.digest,
                "value_type": "projective-point",
            }
        )


class ProjectiveLine(StrictModel):
    """The complete ordered projective line for one presentation and axis."""

    presentation: FiniteFieldPresentation
    axis: Axis
    points: tuple[ProjectivePoint, ...]

    @model_validator(mode="after")
    def validate_line(self) -> Self:
        expected = (self.presentation.order ** len(self.axis.labels) - 1) // (
            self.presentation.order - 1
        )
        if expected > _MAX_FIELD_ORDER:
            raise _validation_error(
                "finite_field.projective_line_exceeds_supported_direction_bound",
                "projective line exceeds the supported direction bound",
            )
        if len(self.points) != expected:
            raise _validation_error(
                "finite_field.projective_line_contain_direction",
                "projective line must contain every direction",
            )
        if any(
            point.presentation != self.presentation or point.axis != self.axis
            for point in self.points
        ):
            raise _validation_error(
                "finite_field.projective_line_points_share_parent_axis",
                "projective line points must share their parent and axis",
            )
        if len({point.digest for point in self.points}) != len(self.points):
            raise _validation_error(
                "finite_field.projective_line_repeat_direction",
                "projective line cannot repeat a direction",
            )
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "axis": self.axis.digest,
                "points": [point.digest for point in self.points],
                "presentation": self.presentation.digest,
                "value_type": "projective-line",
            }
        )


class FiniteLinearMap(StrictModel):
    """A matrix-defined linear map with exact source and target axes."""

    source_axis: Axis
    target_axis: Axis
    matrix: PrimeFieldMatrix

    @model_validator(mode="after")
    def validate_linear_map(self) -> Self:
        if self.matrix.columns != len(self.source_axis.labels):
            raise _validation_error(
                "finite_field.linear_map_columns_match_source_axis",
                "linear-map columns must match the source axis",
            )
        if len(self.matrix.entries) != len(self.target_axis.labels):
            raise _validation_error(
                "finite_field.linear_map_rows_match_target_axis",
                "linear-map rows must match the target axis",
            )
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "entries": [list(row) for row in self.matrix.entries],
                "prime": self.matrix.prime,
                "source_axis": self.source_axis.digest,
                "target_axis": self.target_axis.digest,
                "value_type": "finite-linear-map",
            }
        )


class RankResult(StrictModel):
    """The exact rank of a direction-bound finite linear map."""

    subspace: FiniteDimensionalSubspace
    direction: ProjectivePoint
    linear_map: FiniteLinearMap
    rank: int

    @model_validator(mode="after")
    def validate_rank(self) -> Self:
        if self.linear_map.matrix.prime != self.direction.presentation.characteristic:
            raise _validation_error(
                "finite_field.rank_map_direction_s_prime_field",
                "rank map must use the direction's prime field",
            )
        maximum_rank = min(
            len(self.linear_map.matrix.entries),
            self.linear_map.matrix.columns,
        )
        if type(self.rank) is not int or not 0 <= self.rank <= maximum_rank:
            raise _validation_error(
                "finite_field.rank_linear_map_dimensions",
                "rank is outside the linear-map dimensions",
            )
        if self.direction.presentation != self.subspace.presentation:
            raise _validation_error(
                "finite_field.rank_direction_subspace_presentation",
                "rank direction must use the subspace presentation",
            )
        if self.direction.axis != self.subspace.row_axis:
            raise _validation_error(
                "finite_field.rank_direction_subspace_row_axis",
                "rank direction must use the subspace row axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        subspace: FiniteDimensionalSubspace,
        direction: ProjectivePoint,
        linear_map: FiniteLinearMap,
        rank: int,
    ) -> Self:
        return cls.model_construct(
            subspace=subspace,
            direction=direction,
            linear_map=linear_map,
            rank=rank,
        )

    @property
    def digest(self) -> str:
        return _digest(
            {
                "direction": self.direction.digest,
                "linear_map": self.linear_map.digest,
                "rank": self.rank,
                "subspace": self.subspace.digest,
                "value_type": "rank-result",
            }
        )


def _direction_rank_work(
    subspace: FiniteDimensionalSubspace,
    direction_count: int,
) -> int:
    source_dimension = len(subspace.basis)
    target_dimension = len(subspace.column_axis.labels) * subspace.presentation.degree
    restriction = source_dimension * len(subspace.row_axis.labels) * target_dimension
    rank_work = (
        target_dimension * source_dimension * min(target_dimension, source_dimension)
    )
    return direction_count * (restriction + rank_work)


class DirectionRankLedger(StrictModel):
    """An ordered, exact binding from projective directions to rank results."""

    subspace: FiniteDimensionalSubspace
    entries: tuple[RankResult, ...] = Field(min_length=1, max_length=_MAX_FIELD_ORDER)

    @model_validator(mode="after")
    def validate_ledger(self) -> Self:
        first = self.entries[0]
        expected_directions = (
            self.subspace.presentation.order ** len(self.subspace.row_axis.labels) - 1
        ) // (self.subspace.presentation.order - 1)
        if len(self.entries) != expected_directions:
            raise _validation_error(
                "finite_field.direction_rank_ledger_contains_every_projective_direction",
                "direction-rank ledger must contain every projective direction",
            )
        if len({entry.direction.digest for entry in self.entries}) != len(self.entries):
            raise _validation_error(
                "finite_field.direction_rank_ledger_repeat_direction",
                "direction-rank ledger cannot repeat a direction",
            )
        if any(
            entry.subspace != self.subspace
            or entry.direction.presentation != first.direction.presentation
            or entry.direction.axis != first.direction.axis
            or entry.linear_map.source_axis != first.linear_map.source_axis
            or entry.linear_map.target_axis != first.linear_map.target_axis
            or entry.linear_map.matrix.prime != first.linear_map.matrix.prime
            for entry in self.entries
        ):
            raise _validation_error(
                "finite_field.direction_rank_entries_share_bound_semantics",
                "direction-rank entries must share their bound semantics",
            )
        if first.direction.presentation != self.subspace.presentation:
            raise _validation_error(
                "finite_field.ledger_directions_subspace_presentation",
                "ledger directions must use the subspace presentation",
            )
        if first.direction.axis != self.subspace.row_axis:
            raise _validation_error(
                "finite_field.ledger_directions_subspace_row_axis",
                "ledger directions must use the subspace row axis",
            )
        if first.linear_map.source_axis != self.subspace.basis_axis:
            raise _validation_error(
                "finite_field.ledger_maps_subspace_basis_axis",
                "ledger maps must use the subspace basis axis",
            )
        work = _direction_rank_work(self.subspace, len(self.entries))
        if work > _MAX_DERIVATION_WORK:
            raise _validation_error(
                "finite_field.direction_rank_ledger_exceeds_derivation_work_budget",
                "direction-rank ledger exceeds its derivation work budget",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        subspace: FiniteDimensionalSubspace,
        entries: tuple[RankResult, ...],
    ) -> Self:
        return cls.model_construct(subspace=subspace, entries=entries)

    @property
    def digest(self) -> str:
        return _digest(
            {
                "entries": [entry.digest for entry in self.entries],
                "subspace": self.subspace.digest,
                "value_type": "direction-rank-ledger",
            }
        )


class OrbitDistribution(StrictModel):
    """Orbit-size counts derived from one exact direction-rank ledger."""

    ledger: DirectionRankLedger
    counts: tuple[tuple[int, int], ...]

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        if self.counts != _orbit_counts(self.ledger):
            raise _validation_error(
                "finite_field.orbit_counts_do_not_match_direction_rank_ledger",
                "orbit counts do not match the direction-rank ledger",
            )
        return self

    @classmethod
    def from_ledger(cls, ledger: DirectionRankLedger) -> Self:
        return cls(ledger=ledger, counts=_orbit_counts(ledger))

    @property
    def digest(self) -> str:
        return _digest(
            {
                "counts": [list(item) for item in self.counts],
                "ledger": self.ledger.digest,
                "value_type": "orbit-distribution",
            }
        )


class FinitePolynomial(StrictModel):
    """A canonical univariate polynomial over one exact field presentation."""

    presentation: FiniteFieldPresentation
    variable: str
    coefficients: tuple[FiniteFieldElement, ...]

    @model_validator(mode="after")
    def validate_polynomial(self) -> Self:
        if not self.variable:
            raise _validation_error(
                "finite_field.finite_polynomial_variable_nonempty",
                "finite polynomial variable must be nonempty",
            )
        if not self.coefficients:
            raise _validation_error(
                "finite_field.finite_polynomial_constant_coefficient",
                "finite polynomial requires a constant coefficient",
            )
        if len(self.coefficients) > _MAX_FIELD_ORDER:
            raise _validation_error(
                "finite_field.finite_polynomial_exceeds_supported_degree_bound",
                "finite polynomial exceeds the supported degree bound",
            )
        if any(
            coefficient.presentation != self.presentation
            for coefficient in self.coefficients
        ):
            raise _validation_error(
                "finite_field.finite_polynomial_coefficients_share_parent",
                "finite polynomial coefficients must share their parent",
            )
        if len(self.coefficients) > 1 and self.coefficients[-1].is_zero:
            raise _validation_error(
                "finite_field.finite_polynomial_trailing_zero_coefficient",
                "finite polynomial cannot have a trailing zero coefficient",
            )
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "coefficients": [
                    list(value.coordinates) for value in self.coefficients
                ],
                "presentation": self.presentation.digest,
                "value_type": "finite-polynomial",
                "variable": self.variable,
            }
        )


class FinitePolynomialMap(StrictModel):
    """A polynomial self-map of one exactly presented finite field."""

    domain: FiniteFieldPresentation
    codomain: FiniteFieldPresentation
    polynomial: FinitePolynomial

    @model_validator(mode="after")
    def validate_map(self) -> Self:
        if self.polynomial.presentation != self.domain or self.codomain != self.domain:
            raise _validation_error(
                "finite_field.finite_polynomial_map_one_exact_presentation",
                "finite polynomial map must use one exact field presentation",
            )
        return self

    @property
    def digest(self) -> str:
        return _digest(
            {
                "codomain": self.codomain.digest,
                "domain": self.domain.digest,
                "polynomial": self.polynomial.digest,
                "value_type": "finite-polynomial-map",
            }
        )


class FiniteMapTable(StrictModel):
    """A complete ordered evaluation table for one exact finite map."""

    map: FinitePolynomialMap
    entries: tuple[tuple[FiniteFieldElement, FiniteFieldElement], ...] = Field(
        min_length=1,
        max_length=_MAX_FIELD_ORDER,
    )

    @model_validator(mode="after")
    def validate_table(self) -> Self:
        if len(self.entries) != self.map.domain.order:
            raise _validation_error(
                "finite_field.finite_map_table_enumerate_complete_domain",
                "finite map table must enumerate the complete domain",
            )
        if self.map.domain.order > _MAX_FIELD_ORDER:
            raise _validation_error(
                "finite_field.finite_map_table_exceeds_supported_domain_bound",
                "finite map table exceeds the supported domain bound",
            )
        inputs = tuple(source for source, _ in self.entries)
        if any(value.presentation != self.map.domain for value in inputs):
            raise _validation_error(
                "finite_field.finite_map_table_inputs_domain",
                "finite map table inputs must use the exact domain",
            )
        if len({value.digest for value in inputs}) != len(inputs):
            raise _validation_error(
                "finite_field.finite_map_table_repeat_domain_element",
                "finite map table cannot repeat a domain element",
            )
        if tuple(map(_encoded_coordinates, inputs)) != tuple(
            range(self.map.domain.order)
        ):
            raise _validation_error(
                "finite_field.finite_map_table_inputs_canonical_domain_order",
                "finite map table inputs must use canonical domain order",
            )
        if any(value.presentation != self.map.codomain for _, value in self.entries):
            raise _validation_error(
                "finite_field.finite_map_table_outputs_codomain",
                "finite map table outputs must use the exact codomain",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        polynomial_map: FinitePolynomialMap,
        entries: tuple[tuple[FiniteFieldElement, FiniteFieldElement], ...],
    ) -> Self:
        return cls.model_construct(map=polynomial_map, entries=entries)

    @property
    def digest(self) -> str:
        return _digest(
            {
                "entries": [
                    [source.digest, target.digest] for source, target in self.entries
                ],
                "map": self.map.digest,
                "value_type": "finite-map-table",
            }
        )


def _fibers_for_table(
    table: FiniteMapTable,
) -> tuple[tuple[FiniteFieldElement, tuple[FiniteFieldElement, ...]], ...]:
    grouped: dict[str, tuple[FiniteFieldElement, list[FiniteFieldElement]]] = {}
    for source, target in table.entries:
        _, sources = grouped.setdefault(target.digest, (target, []))
        sources.append(source)
    return tuple((target, tuple(sources)) for target, sources in grouped.values())


class FiberPartition(StrictModel):
    """The nonempty fibers of one complete finite map table."""

    table: FiniteMapTable
    fibers: tuple[tuple[FiniteFieldElement, tuple[FiniteFieldElement, ...]], ...]

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        if not self.fibers or any(not sources for _, sources in self.fibers):
            raise _validation_error(
                "finite_field.fiber_partition_nonempty_fibers",
                "fiber partition requires nonempty fibers",
            )
        return self

    @classmethod
    def from_table(cls, table: FiniteMapTable) -> Self:
        return cls.model_construct(table=table, fibers=_fibers_for_table(table))

    @property
    def digest(self) -> str:
        return _digest(
            {
                "fibers": [
                    [target.digest, [source.digest for source in sources]]
                    for target, sources in self.fibers
                ],
                "table": self.table.digest,
                "value_type": "fiber-partition",
            }
        )


class CollisionResult(StrictModel):
    """Whether a complete finite map table has a collision."""

    table: FiniteMapTable
    status: Literal["COLLISION", "INJECTIVE"]
    left: FiniteFieldElement | None = None
    right: FiniteFieldElement | None = None
    image: FiniteFieldElement | None = None

    @model_validator(mode="after")
    def validate_collision(self) -> Self:
        if self.status == "INJECTIVE":
            if any(value is not None for value in (self.left, self.right, self.image)):
                raise _validation_error(
                    "finite_field.injective_table_carry_collision_values",
                    "an injective table cannot carry collision values",
                )
            return self
        if self.left is None or self.right is None or self.image is None:
            raise _validation_error(
                "finite_field.collision_result_both_inputs_image",
                "a collision result requires both inputs and their image",
            )
        if self.left == self.right:
            raise _validation_error(
                "finite_field.collision_inputs_distinct",
                "collision inputs must be distinct",
            )
        if (
            self.left.presentation != self.table.map.domain
            or self.right.presentation != self.table.map.domain
            or self.image.presentation != self.table.map.codomain
        ):
            raise _validation_error(
                "finite_field.collision_result_parent_shape",
                "collision values must use the table's domain and codomain",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        table: FiniteMapTable,
        status: Literal["COLLISION", "INJECTIVE"],
        left: FiniteFieldElement | None = None,
        right: FiniteFieldElement | None = None,
        image: FiniteFieldElement | None = None,
    ) -> Self:
        return cls.model_construct(
            table=table, status=status, left=left, right=right, image=image
        )

    @property
    def digest(self) -> str:
        return _digest(
            {
                "image": self.image.digest if self.image is not None else None,
                "left": self.left.digest if self.left is not None else None,
                "right": self.right.digest if self.right is not None else None,
                "table": self.table.digest,
                "status": self.status,
                "value_type": "finite-map-collision",
            }
        )


class PermutationResult(StrictModel):
    """Whether a complete finite map table is a permutation."""

    table: FiniteMapTable
    status: Literal["PERMUTATION", "NOT_PERMUTATION"]
    inverse_entries: tuple[tuple[FiniteFieldElement, FiniteFieldElement], ...] = ()

    @model_validator(mode="after")
    def validate_permutation(self) -> Self:
        if self.status == "NOT_PERMUTATION":
            if self.inverse_entries:
                raise _validation_error(
                    "finite_field.non_permutation_result_carry_inverse",
                    "a non-permutation result cannot carry an inverse",
                )
            return self
        if len(self.inverse_entries) != len(self.table.entries):
            raise _validation_error(
                "finite_field.permutation_result_inverse_shape",
                "a permutation result must carry one inverse row per table row",
            )
        if any(
            target.presentation != self.table.map.codomain
            or source.presentation != self.table.map.domain
            for target, source in self.inverse_entries
        ):
            raise _validation_error(
                "finite_field.permutation_result_parent_shape",
                "inverse rows must use the table's codomain and domain",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        table: FiniteMapTable,
        status: Literal["PERMUTATION", "NOT_PERMUTATION"],
        inverse_entries: tuple[tuple[FiniteFieldElement, FiniteFieldElement], ...] = (),
    ) -> Self:
        return cls.model_construct(
            table=table, status=status, inverse_entries=inverse_entries
        )

    @property
    def digest(self) -> str:
        return _digest(
            {
                "inverse_entries": [
                    [target.digest, source.digest]
                    for target, source in self.inverse_entries
                ],
                "table": self.table.digest,
                "status": self.status,
                "value_type": "finite-map-permutation",
            }
        )


def _orbit_counts(ledger: DirectionRankLedger) -> tuple[tuple[int, int], ...]:
    first = ledger.entries[0]
    presentation = first.direction.presentation
    expected_directions = (
        presentation.order ** len(first.direction.axis.labels) - 1
    ) // (presentation.order - 1)
    if len(ledger.entries) != expected_directions:
        raise _validation_error(
            "finite_field.orbit_aggregation_projective_direction",
            "orbit aggregation requires every projective direction",
        )
    prime = presentation.characteristic
    target_dimension = len(first.linear_map.target_axis.labels)
    counts: dict[int, int] = {1: expected_directions}
    for entry in ledger.entries:
        orbit_size = prime**entry.rank
        counts[orbit_size] = counts.get(orbit_size, 0) + prime ** (
            target_dimension - entry.rank
        )
    return tuple(sorted(counts.items()))
