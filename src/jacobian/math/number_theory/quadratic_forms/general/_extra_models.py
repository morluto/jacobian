"""Additional exact structural quadratic-form contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import (
    CanonicalRational,
    ExactInteger,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.matrices.values import RationalMatrix, RationalVectorSpaceBasis
from jacobian.math.number_theory.quadratic_forms.general.values import (
    RationalCoordinateVector,
    RationalQuadraticForm,
)

MAX_QUADRATIC_PULLBACK_AXIS = 128
MAX_QUADRATIC_PULLBACK_WORK = 2_000_000
MAX_QUADRATIC_PULLBACK_OUTPUT_ENTRIES = MAX_QUADRATIC_PULLBACK_AXIS**2
MAX_QUADRATIC_DIAGONALIZATION_AXIS = 64
MAX_QUADRATIC_DIAGONALIZATION_WORK = MAX_QUADRATIC_DIAGONALIZATION_AXIS**3
MAX_QUADRATIC_DIAGONALIZATION_INTERMEDIATE_DIGITS = 16_384
MAX_QUADRATIC_DIAGONALIZATION_OUTPUT_DIGITS = 8_192
MAX_QUADRATIC_DIAGONALIZATION_OUTPUT_TOTAL_DIGITS = 2_000_000


class FormRequest(StrictModel):
    form: RationalQuadraticForm


class SignatureResult(StrictModel):
    form: RationalQuadraticForm
    positive_index: int = Field(ge=0)
    negative_index: int = Field(ge=0)
    zero_index: int = Field(ge=0)
    signature: int

    @model_validator(mode="after")
    def shape(self) -> Self:
        if self.positive_index + self.negative_index + self.zero_index != len(
            self.form.axis
        ):
            raise ValueError("signature counts must sum to form dimension")
        if self.signature != self.positive_index - self.negative_index:
            raise ValueError("signature must equal positive minus negative index")
        return self


class RadicalResult(StrictModel):
    form: RationalQuadraticForm
    rank: int = Field(ge=0)
    radical: RationalVectorSpaceBasis

    @model_validator(mode="after")
    def shape(self) -> Self:
        if self.radical.ambient_dimension != len(
            self.form.axis
        ) or self.rank + self.radical.vectors.__len__() != len(self.form.axis):
            raise ValueError("rank and radical must span the form dimension")
        return self


class PullbackRequest(StrictModel):
    form: RationalQuadraticForm
    matrix: RationalMatrix
    target_axis: tuple[OpaqueLabel, ...]

    @model_validator(mode="after")
    def shape(self) -> Self:
        if self.matrix.row_count != len(
            self.form.axis
        ) or self.matrix.column_count != len(self.target_axis):
            raise ValueError(
                "pullback matrix dimensions must map target axis to source axis"
            )
        if any(not label or label != label.strip() for label in self.target_axis):
            raise ValueError("target axis labels must be nonempty and trimmed")
        if len(set(self.target_axis)) != len(self.target_axis):
            raise ValueError("target axis labels must be unique")
        return self


class PullbackResult(StrictModel):
    source_form: RationalQuadraticForm
    matrix: RationalMatrix
    form: RationalQuadraticForm
    source_axis: tuple[OpaqueLabel, ...] = Field(max_length=MAX_QUADRATIC_PULLBACK_AXIS)
    target_axis: tuple[OpaqueLabel, ...] = Field(max_length=MAX_QUADRATIC_PULLBACK_AXIS)

    @model_validator(mode="after")
    def shape(self) -> Self:
        if len(self.source_form.axis) > MAX_QUADRATIC_PULLBACK_AXIS:
            raise ValueError("pullback source axis exceeds the owner envelope")
        if self.source_axis != self.source_form.axis:
            raise ValueError("pullback source axis must match the source form")
        if self.target_axis != self.form.axis:
            raise ValueError("pullback target axis must match the result form")
        if self.matrix.row_count != len(self.source_axis):
            raise ValueError("pullback matrix rows must match the source axis")
        if self.matrix.column_count != len(self.target_axis):
            raise ValueError("pullback matrix columns must match the target axis")
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_form: RationalQuadraticForm,
        matrix: RationalMatrix,
        form: RationalQuadraticForm,
        source_axis: tuple[OpaqueLabel, ...],
        target_axis: tuple[OpaqueLabel, ...],
    ) -> Self:
        return cls.model_construct(
            source_form=source_form,
            matrix=matrix,
            form=form,
            source_axis=source_axis,
            target_axis=target_axis,
        )


class DiagonalizationResult(StrictModel):
    form: RationalQuadraticForm
    diagonal: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_QUADRATIC_DIAGONALIZATION_AXIS
    )
    change: RationalMatrix
    source_axis: tuple[OpaqueLabel, ...] = Field(
        max_length=MAX_QUADRATIC_DIAGONALIZATION_AXIS
    )
    basis_axis: tuple[OpaqueLabel, ...] = Field(
        max_length=MAX_QUADRATIC_DIAGONALIZATION_AXIS
    )

    @model_validator(mode="after")
    def shape(self) -> Self:
        n = len(self.form.axis)
        if (
            len(self.diagonal) != n
            or self.change.row_count != n
            or self.change.column_count != n
            or self.source_axis != self.form.axis
            or len(self.basis_axis) != n
            or len(set(self.basis_axis)) != n
        ):
            raise ValueError("diagonalization axes and dimensions must match")
        output_digits = 0
        for value in self.diagonal:
            require_bounded_rational(
                value,
                max_digits=MAX_QUADRATIC_DIAGONALIZATION_OUTPUT_DIGITS,
                label="quadratic-form diagonalization coefficient",
            )
            output_digits += len(str(abs(value.num))) + len(str(abs(value.den)))
        for row in self.change.entries:
            for value in row:
                require_bounded_rational(
                    value,
                    max_digits=MAX_QUADRATIC_DIAGONALIZATION_OUTPUT_DIGITS,
                    label="quadratic-form diagonalization change coefficient",
                )
                output_digits += len(str(abs(value.num))) + len(str(abs(value.den)))
        if output_digits > MAX_QUADRATIC_DIAGONALIZATION_OUTPUT_TOTAL_DIGITS:
            raise ValueError("diagonalization result exceeds its aggregate digit bound")
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        form: RationalQuadraticForm,
        diagonal: tuple[CanonicalRational, ...],
        change: RationalMatrix,
        source_axis: tuple[OpaqueLabel, ...],
        basis_axis: tuple[OpaqueLabel, ...],
    ) -> Self:
        return cls.model_construct(
            form=form,
            diagonal=diagonal,
            change=change,
            source_axis=source_axis,
            basis_axis=basis_axis,
        )


class ModularProfileRequest(StrictModel):
    form: RationalQuadraticForm
    modulus: int = Field(ge=1, le=64)


class ModularProfileResult(StrictModel):
    form: RationalQuadraticForm
    modulus: int
    histogram: tuple[int, ...]
    total: int

    @model_validator(mode="after")
    def shape(self) -> Self:
        if len(self.histogram) != self.modulus or self.total != self.modulus ** len(
            self.form.axis
        ):
            raise ValueError("modular histogram total mismatch")
        return self


MAX_QUADRATIC_GAUSS_MODULUS = 64
MAX_QUADRATIC_GAUSS_STATES = 2_000_000
MAX_QUADRATIC_GAUSS_WORK = 2_000_000
MAX_QUADRATIC_GAUSS_SUPPORT_TERMS = 4_096
MAX_QUADRATIC_GAUSS_OUTPUT_DIGITS = 1_000_000


class FiniteGaussSumRequest(StrictModel):
    """The sum of exp(2*pi*i*Q(x)/m) over the complete residue module.

    Admission bounds the residue domain, the polynomial support (which the
    result retains and each enumerated state evaluates), their product as
    kernel work, and the retained source plus canonical output as aggregate
    decimal digits, all before any enumeration runs.
    """

    form: RationalQuadraticForm
    modulus: int = Field(ge=1, le=MAX_QUADRATIC_GAUSS_MODULUS)


class FiniteGaussSumResult(StrictModel):
    """Exact Gauss sum and the complete value histogram determining it."""

    form: RationalQuadraticForm
    modulus: int = Field(ge=1, le=MAX_QUADRATIC_GAUSS_MODULUS)
    additive_character: Literal["EXP_2PI_I_Q_OVER_MODULUS_V1"] = (
        "EXP_2PI_I_Q_OVER_MODULUS_V1"
    )
    histogram: tuple[int, ...]
    total: int = Field(ge=1)
    value: RationalCyclotomicElement

    @model_validator(mode="after")
    def exact_shape(self) -> Self:
        if (
            len(self.histogram) != self.modulus
            or self.total != self.modulus ** len(self.form.axis)
            or sum(self.histogram) != self.total
            or self.value.field != RationalCyclotomicField(order=self.modulus)
        ):
            raise ValueError("finite Gauss sum must match its complete modular profile")
        if any(
            count < 0 or count > MAX_QUADRATIC_GAUSS_STATES for count in self.histogram
        ):
            raise ValueError("finite Gauss histogram exceeds its admitted state count")
        if any(
            max(len(str(abs(coordinate.num))), len(str(abs(coordinate.den))))
            > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
            for coordinate in self.value.coefficients_ascending
        ):
            raise ValueError("finite Gauss coordinates exceed the cyclotomic bound")
        return self


MAX_THETA_PREFIX_CUTOFF = 512
MAX_THETA_SELECTED_INDEX = 1_000_000_000
MAX_THETA_SELECTED_INDICES = 128
MAX_THETA_PREFIX_DIMENSION = 7
MAX_THETA_PREFIX_VECTORS = 100_000
MAX_THETA_PREFIX_WORK = 2_000_000
MAX_THETA_PREFIX_OUTPUT_DIGITS = 64_000
MAX_THETA_REPRESENTATION_VECTOR_COUNT = 100_000
MAX_THETA_REPRESENTATION_COORDINATE_ABS = 49_999


class ThetaRepresentingVectorsRequest(StrictModel):
    """Complete vector fibers at selected values of a positive-definite form."""

    form: RationalQuadraticForm
    indices: tuple[Annotated[int, Field(ge=0, le=MAX_THETA_SELECTED_INDEX)], ...] = (
        Field(
            min_length=1,
            max_length=MAX_THETA_SELECTED_INDICES,
            description=(
                f"Strictly increasing, distinct indices in [0, {MAX_THETA_SELECTED_INDEX}]."
            ),
        )
    )

    @model_validator(mode="after")
    def require_canonical_indices(self) -> Self:
        if any(index < 0 or index > MAX_THETA_SELECTED_INDEX for index in self.indices):
            raise ValueError(
                f"representation indices must lie in [0, {MAX_THETA_SELECTED_INDEX}]"
            )
        if tuple(sorted(set(self.indices))) != self.indices:
            raise ValueError("representation indices must be strictly increasing")
        return self

    @property
    def cutoff(self) -> int:
        return self.indices[-1]


class ThetaRepresentingVectorsRow(StrictModel):
    index: int = Field(ge=0, le=MAX_THETA_SELECTED_INDEX)
    vectors: tuple[RationalCoordinateVector, ...] = Field(
        max_length=MAX_THETA_REPRESENTATION_VECTOR_COUNT,
        description=(
            "Every integer coordinate tuple in the result form's ordered axis, "
            "in strictly increasing lexicographic order."
        ),
    )


class ThetaRepresentingVectorsResult(StrictModel):
    """All selected vectors, with coordinates ordered by ``form.axis``."""

    form: RationalQuadraticForm
    rows: tuple[ThetaRepresentingVectorsRow, ...] = Field(
        min_length=1, max_length=MAX_THETA_SELECTED_INDICES
    )

    @model_validator(mode="after")
    def require_complete_canonical_table(self) -> Self:
        previous_index = -1
        vector_count = 0
        for row in self.rows:
            if row.index <= previous_index:
                raise ValueError("representation rows must have increasing indices")
            previous_index = row.index
            previous_vector: tuple[int, ...] | None = None
            for vector in row.vectors:
                if vector.axis != self.form.axis:
                    raise ValueError("representation vector axis must match form axis")
                if any(value.den != 1 for value in vector.coordinates):
                    raise ValueError("representation coordinates must be integers")
                coordinates = tuple(value.num for value in vector.coordinates)
                if previous_vector is not None and coordinates <= previous_vector:
                    raise ValueError(
                        "representation vectors must be unique and ordered"
                    )
                previous_vector = coordinates
                for coordinate in coordinates:
                    if abs(coordinate) > MAX_THETA_REPRESENTATION_COORDINATE_ABS:
                        raise ValueError("representation coordinate exceeds its bound")
            vector_count += len(row.vectors)
            if vector_count > MAX_THETA_REPRESENTATION_VECTOR_COUNT:
                raise ValueError("representation table exceeds its vector-count bound")
        return self


class ThetaSeriesPrefixRequest(StrictModel):
    """Exact coefficients of a positive-definite integral form through q^N."""

    form: RationalQuadraticForm
    cutoff: int = Field(ge=0, le=MAX_THETA_PREFIX_CUTOFF)


class ThetaSeriesPrefixResult(StrictModel):
    """The source-bound tuple (r_Q(0), ..., r_Q(cutoff))."""

    form: RationalQuadraticForm
    cutoff: int = Field(ge=0, le=MAX_THETA_PREFIX_CUTOFF)
    coefficients: tuple[int, ...]

    @model_validator(mode="after")
    def require_prefix_shape(self) -> Self:
        if len(self.coefficients) != self.cutoff + 1:
            raise ValueError("theta coefficients must cover q^0 through q^cutoff")
        if any(value < 0 for value in self.coefficients):
            raise ValueError("theta coefficients must be nonnegative")
        return self


class ThetaSelectedCoefficientsRequest(StrictModel):
    """Selected exact representation numbers of a positive-definite form."""

    form: RationalQuadraticForm
    indices: tuple[Annotated[int, Field(ge=0, le=MAX_THETA_SELECTED_INDEX)], ...] = (
        Field(
            min_length=1,
            max_length=MAX_THETA_SELECTED_INDICES,
            description=(
                f"Strictly increasing distinct indices in [0, {MAX_THETA_SELECTED_INDEX}]."
            ),
        )
    )

    @model_validator(mode="after")
    def require_canonical_indices(self) -> Self:
        if any(index < 0 or index > MAX_THETA_SELECTED_INDEX for index in self.indices):
            raise ValueError(
                f"theta indices must lie in [0, {MAX_THETA_SELECTED_INDEX}]"
            )
        if tuple(sorted(set(self.indices))) != self.indices:
            raise ValueError("theta indices must be strictly increasing")
        return self

    @property
    def cutoff(self) -> int:
        return self.indices[-1]


class ThetaSelectedCoefficient(StrictModel):
    index: int = Field(ge=0, le=MAX_THETA_SELECTED_INDEX)
    coefficient: int = Field(ge=0)


class ThetaSelectedCoefficientsResult(StrictModel):
    """Exact source-bound coefficients at the requested increasing indices."""

    form: RationalQuadraticForm
    coefficients: tuple[ThetaSelectedCoefficient, ...] = Field(
        min_length=1, max_length=MAX_THETA_SELECTED_INDICES
    )

    @model_validator(mode="after")
    def require_increasing_indices(self) -> Self:
        indices = tuple(row.index for row in self.coefficients)
        if tuple(sorted(set(indices))) != indices:
            raise ValueError(
                "selected theta result indices must be strictly increasing"
            )
        if any(row.coefficient > MAX_THETA_PREFIX_VECTORS for row in self.coefficients):
            raise ValueError(
                "a selected theta coefficient exceeds the admitted vector count"
            )
        return self


MAX_QUADRATIC_BOX_RADIUS = 64
MAX_QUADRATIC_BOX_VECTORS = 25_000
MAX_QUADRATIC_BOX_PROFILE_ROWS = 25_000
MAX_QUADRATIC_BOX_OUTPUT_DIGITS = 1_000_000


class FiniteBoxProfileRequest(StrictModel):
    """Complete value histogram on the symmetric integer box [-radius,radius]^n."""

    form: RationalQuadraticForm
    radius: int = Field(ge=0, le=MAX_QUADRATIC_BOX_RADIUS)


class FiniteBoxProfileRow(StrictModel):
    value: ExactInteger
    representation_count: int = Field(ge=1)


class FiniteBoxProfileResult(StrictModel):
    """Exact source-bound counts of integral vectors by the value of an integral form."""

    form: RationalQuadraticForm
    radius: int = Field(ge=0, le=MAX_QUADRATIC_BOX_RADIUS)
    coordinate_bounds: tuple[tuple[int, int], ...]
    vector_count: int = Field(ge=1, le=MAX_QUADRATIC_BOX_VECTORS)
    rows: tuple[FiniteBoxProfileRow, ...] = Field(
        max_length=MAX_QUADRATIC_BOX_PROFILE_ROWS
    )
    minimum_value: ExactInteger
    maximum_value: ExactInteger

    @model_validator(mode="after")
    def complete_profile_shape(self) -> Self:
        if len(self.coordinate_bounds) != len(self.form.axis) or any(
            bound != (-self.radius, self.radius) for bound in self.coordinate_bounds
        ):
            raise ValueError("coordinate bounds must match the source form axis")
        side_length = 2 * self.radius + 1
        expected_vectors = 1
        for _ in self.form.axis:
            expected_vectors *= side_length
            if expected_vectors > MAX_QUADRATIC_BOX_VECTORS:
                break
        if self.vector_count != expected_vectors:
            raise ValueError("vector count must cover the declared integer box")
        if sum(row.representation_count for row in self.rows) != self.vector_count:
            raise ValueError("profile counts must cover the complete vector domain")
        values = tuple(row.value for row in self.rows)
        if not values or values != tuple(sorted(set(values))):
            raise ValueError("finite-box profile values must be strictly increasing")
        if (self.minimum_value, self.maximum_value) != (values[0], values[-1]):
            raise ValueError("finite-box extrema must match the complete profile")
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        form: RationalQuadraticForm,
        radius: int,
        coordinate_bounds: tuple[tuple[int, int], ...],
        vector_count: int,
        rows: tuple[FiniteBoxProfileRow, ...],
        minimum_value: int,
        maximum_value: int,
    ) -> Self:
        """Construct the profile after its complete histogram was established."""
        return cls.model_construct(
            form=form,
            radius=radius,
            coordinate_bounds=coordinate_bounds,
            vector_count=vector_count,
            rows=rows,
            minimum_value=minimum_value,
            maximum_value=maximum_value,
        )


__all__ = [
    "DiagonalizationResult",
    "FiniteBoxProfileRequest",
    "FiniteBoxProfileResult",
    "FiniteBoxProfileRow",
    "FormRequest",
    "ModularProfileRequest",
    "ModularProfileResult",
    "PullbackRequest",
    "PullbackResult",
    "RadicalResult",
    "SignatureResult",
    "ThetaSeriesPrefixRequest",
    "ThetaSeriesPrefixResult",
]
