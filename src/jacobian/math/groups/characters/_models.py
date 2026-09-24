"""Typed wire contracts for finite class-function operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._cyclotomic import euler_phi

MAX_CLASS_COUNT = 128
MAX_GROUP_ORDER = 1_000_000
MAX_CYCLOTOMIC_ORDER = 60
MAX_VALUE_COEFFICIENT_DIGITS = 512
MAX_INNER_PRODUCT_WORK = 250_000
MAX_CHARACTER_TABLE_CELLS = 100_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"groups.characters.{reason}", message)


class CyclotomicValue(StrictModel):
    """One exact element of Q(zeta_order) in the distinguished power basis.

    ``coefficients`` are ascending powers of a primitive ``order``-th root of
    unity, reduced modulo the order's cyclotomic polynomial; equality is
    coefficient-wise.  Order one is the rational subfield with the single
    coefficient equal to the rational value.
    """

    order: int = Field(ge=1, le=MAX_CYCLOTOMIC_ORDER)
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_CYCLOTOMIC_ORDER,
        description="Ascending-power coefficients of length phi(order).",
    )

    @model_validator(mode="after")
    def require_cyclotomic_degree(self) -> Self:
        if len(self.coefficients) != euler_phi(self.order):
            raise _validation_error(
                "cyclotomic_degree",
                "value coefficients must have length phi(order)",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, order: int, coefficients: tuple[CanonicalRational, ...]
    ) -> CyclotomicValue:
        return cls.model_construct(order=order, coefficients=coefficients)


class ConjugacyClassPartition(GroupConjugacyClassesResult):
    """A canonical conjugacy partition retaining its concrete group parent."""

    @classmethod
    def _from_group_result(cls, value: GroupConjugacyClassesResult) -> Self:
        return cls.model_construct(source=value.source, classes=value.classes)


class ClassAxis(StrictModel):
    """The shared conjugacy-class axis of one finite group.

    ``class_sizes`` are the exact conjugacy-class cardinalities in the
    declared order and ``group_order`` is their sum.  ``cyclotomic_order`` is
    the common root-of-unity order declared for values on this axis.
    """

    class_sizes: tuple[int, ...] = Field(min_length=1, max_length=MAX_CLASS_COUNT)
    group_order: int = Field(ge=1, le=MAX_GROUP_ORDER)
    cyclotomic_order: int = Field(ge=1, le=MAX_CYCLOTOMIC_ORDER)
    # Legacy synthetic axes remain useful for class-function arithmetic.  A
    # concrete group-bound axis carries both fields, making equal-sized axes
    # from different groups impossible to confuse.
    group: PermutationGroup | None = None
    class_representatives: tuple[tuple[int, ...], ...] | None = None

    @model_validator(mode="after")
    def require_complete_partition(self) -> Self:
        if any(size < 1 or size > self.group_order for size in self.class_sizes):
            raise _validation_error(
                "class_size",
                "every class size must be a positive part of the group order",
            )
        if sum(self.class_sizes) != self.group_order:
            raise _validation_error(
                "group_order",
                "group_order must equal the sum of the class sizes",
            )
        if (self.group is None) != (self.class_representatives is None):
            raise _validation_error(
                "group_parent",
                "a concrete class axis must carry both its group and representatives",
            )
        if self.group is not None:
            representatives = self.class_representatives
            if representatives is None:
                raise _validation_error(
                    "group_parent",
                    "a concrete class axis requires class representatives",
                )
            if len(representatives) != len(self.class_sizes):
                raise _validation_error(
                    "representative_count", "one representative is required per class"
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        class_sizes: tuple[int, ...],
        cyclotomic_order: int,
        group: PermutationGroup | None = None,
        class_representatives: tuple[tuple[int, ...], ...] | None = None,
    ) -> ClassAxis:
        return cls.model_construct(
            class_sizes=class_sizes,
            group_order=sum(class_sizes),
            cyclotomic_order=cyclotomic_order,
            group=group,
            class_representatives=class_representatives,
        )


class FiniteClassFunction(StrictModel):
    """One complete exact class function on a single conjugacy-class axis.

    ``values`` contains exactly one exact cyclotomic value per class in
    ``axis.class_sizes`` order, all sharing the axis's cyclotomic order.  A
    same-length vector for another axis is structurally invalid.
    """

    axis: ClassAxis
    values: tuple[CyclotomicValue, ...] = Field(
        min_length=1, max_length=MAX_CLASS_COUNT
    )

    @model_validator(mode="after")
    def require_axis_bound_values(self) -> Self:
        if len(self.values) != len(self.axis.class_sizes):
            raise _validation_error(
                "value_count",
                "a class function needs exactly one value per class",
            )
        if any(value.order != self.axis.cyclotomic_order for value in self.values):
            raise _validation_error(
                "value_cyclotomic_order",
                "every value must share the class axis cyclotomic order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, axis: ClassAxis, values: tuple[CyclotomicValue, ...]
    ) -> FiniteClassFunction:
        return cls.model_construct(axis=axis, values=values)


class ClassFunctionInnerProductRequest(StrictModel):
    """Two class functions on one shared conjugacy-class axis."""

    phi: FiniteClassFunction = Field(
        description="Left class function on the shared class axis."
    )
    psi: FiniteClassFunction = Field(
        description=(
            "Right class function on the same class axis; it must carry the "
            "identical class_sizes, group_order, and cyclotomic_order."
        )
    )


class ClassFunctionPointwiseProductRequest(StrictModel):
    """Two class functions on one shared exact cyclotomic class axis."""

    phi: FiniteClassFunction = Field(description="Left class function.")
    psi: FiniteClassFunction = Field(
        description="Right class function on the same axis."
    )


class ClassFunctionAddRequest(StrictModel):
    """Two class functions on one shared exact cyclotomic class axis."""

    phi: FiniteClassFunction = Field(description="Left class function.")
    psi: FiniteClassFunction = Field(
        description="Right class function on the same axis."
    )


class ClassFunctionConjugateRequest(StrictModel):
    """One exact class function to conjugate coefficientwise."""

    function: FiniteClassFunction = Field(
        description="The exact class function on its retained class axis."
    )


class ClassFunctionScaleRequest(StrictModel):
    """One cyclotomic scalar acting on a class function over the same field."""

    scalar: CyclotomicValue = Field(
        description="Exact scalar in the cyclotomic field named by the class axis."
    )
    function: FiniteClassFunction = Field(description="Exact class function to scale.")


class ClassContribution(StrictModel):
    """One exact per-class term of the Hermitian inner product."""

    class_index: int = Field(ge=0)
    class_size: int = Field(ge=1)
    phi_value: CyclotomicValue
    psi_value: CyclotomicValue
    conjugate_psi_value: CyclotomicValue
    weighted_product: CyclotomicValue


class ClassFunctionInnerProductResult(StrictModel):
    """The exact Hermitian inner product with its complete contribution table."""

    axis: ClassAxis
    phi: FiniteClassFunction
    psi: FiniteClassFunction
    contributions: tuple[ClassContribution, ...] = Field(min_length=1)
    inner_product: CyclotomicValue

    @model_validator(mode="after")
    def require_bound_contributions(self) -> Self:
        if self.phi.axis != self.axis or self.psi.axis != self.axis:
            raise _validation_error(
                "result_axis",
                "result inputs must carry the declared shared class axis",
            )
        if len(self.contributions) != len(self.axis.class_sizes):
            raise _validation_error(
                "contribution_count",
                "the contribution table must cover every class",
            )
        if tuple(row.class_index for row in self.contributions) != tuple(
            range(len(self.axis.class_sizes))
        ):
            raise _validation_error(
                "contribution_order",
                "contributions must cover class indices in order",
            )
        if self.inner_product.order != self.axis.cyclotomic_order:
            raise _validation_error(
                "inner_product_order",
                "the inner product must share the axis cyclotomic order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        axis: ClassAxis,
        phi: FiniteClassFunction,
        psi: FiniteClassFunction,
        contributions: tuple[ClassContribution, ...],
        inner_product: CyclotomicValue,
    ) -> ClassFunctionInnerProductResult:
        return cls.model_construct(
            axis=axis,
            phi=phi,
            psi=psi,
            contributions=contributions,
            inner_product=inner_product,
        )


class ClassFunctionRestrictionRequest(StrictModel):
    """Restrict a source class function to an explicitly embedded subgroup."""

    class_function: FiniteClassFunction = Field(
        description="Class function on an axis carrying its concrete source group."
    )
    subgroup: PermutationGroup = Field(
        description=(
            "Subgroup generators on the same permutation domain, each of which "
            "must belong to the source group."
        )
    )


class ClassFunctionRestrictionResult(StrictModel):
    """Exact restriction with the target-to-source conjugacy-class map."""

    source_class_function: FiniteClassFunction
    subgroup_partition: ConjugacyClassPartition
    target_class_to_source_class: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_CLASS_COUNT
    )
    restricted: FiniteClassFunction

    @model_validator(mode="after")
    def require_restriction_axes(self) -> Self:
        source = self.source_class_function.axis.group
        if source is None:
            raise _validation_error(
                "restriction_parent", "the source class function must retain its group"
            )
        if self.restricted.axis.group != self.subgroup_partition.source:
            raise _validation_error(
                "restriction_axis", "restricted values must use the subgroup axis"
            )
        if len(self.target_class_to_source_class) != len(
            self.subgroup_partition.classes
        ) or len(self.restricted.values) != len(self.subgroup_partition.classes):
            raise _validation_error(
                "restriction_shape",
                "one source class and value are required per target class",
            )
        if any(
            not 0 <= index < len(self.source_class_function.values)
            for index in self.target_class_to_source_class
        ):
            raise _validation_error(
                "restriction_index", "source class index is outside the source axis"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_class_function: FiniteClassFunction,
        subgroup_partition: ConjugacyClassPartition,
        target_class_to_source_class: tuple[int, ...],
        restricted: FiniteClassFunction,
    ) -> Self:
        return cls.model_construct(
            source_class_function=source_class_function,
            subgroup_partition=subgroup_partition,
            target_class_to_source_class=target_class_to_source_class,
            restricted=restricted,
        )


class ClassFunctionInductionRequest(StrictModel):
    """Induce a class function along an explicit same-domain subgroup inclusion."""

    class_function: FiniteClassFunction = Field(
        description=(
            "Class function on a concrete subgroup; its axis group supplies the "
            "subgroup generators and canonical class coordinates."
        )
    )
    parent_group: PermutationGroup = Field(
        description=(
            "Concrete parent permutation group on the same domain; every source "
            "subgroup generator must belong to it."
        )
    )


class ClassFunctionInductionResult(StrictModel):
    """Exact induced class function and the subgroup-class inclusion map."""

    source_class_function: FiniteClassFunction
    subgroup_partition: ConjugacyClassPartition
    parent_partition: ConjugacyClassPartition
    subgroup_class_to_parent_class: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_CLASS_COUNT
    )
    induced: FiniteClassFunction

    @model_validator(mode="after")
    def require_induction_axes(self) -> Self:
        subgroup = self.source_class_function.axis.group
        if subgroup is None or subgroup != self.subgroup_partition.source:
            raise _validation_error(
                "induction_subgroup",
                "source class function must be bound to the retained subgroup",
            )
        if self.induced.axis.group != self.parent_partition.source:
            raise _validation_error(
                "induction_parent", "induced values must use the retained parent axis"
            )
        if self.subgroup_partition.source.degree != self.parent_partition.source.degree:
            raise _validation_error(
                "induction_degree",
                "subgroup and parent partitions must share a permutation domain",
            )
        if (
            self.source_class_function.axis.cyclotomic_order
            != self.induced.axis.cyclotomic_order
        ):
            raise _validation_error(
                "induction_field", "induction must preserve the cyclotomic field"
            )
        if len(self.subgroup_class_to_parent_class) != len(
            self.subgroup_partition.classes
        ) or len(self.source_class_function.values) != len(
            self.subgroup_partition.classes
        ):
            raise _validation_error(
                "induction_subgroup_shape",
                "one source value and parent-class image are required per subgroup class",
            )
        if len(self.induced.values) != len(self.parent_partition.classes):
            raise _validation_error(
                "induction_parent_shape",
                "induced values must cover every parent conjugacy class",
            )
        if any(
            not 0 <= index < len(self.parent_partition.classes)
            for index in self.subgroup_class_to_parent_class
        ):
            raise _validation_error(
                "induction_class_index", "parent class index is outside the partition"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_class_function: FiniteClassFunction,
        subgroup_partition: ConjugacyClassPartition,
        parent_partition: ConjugacyClassPartition,
        subgroup_class_to_parent_class: tuple[int, ...],
        induced: FiniteClassFunction,
    ) -> Self:
        return cls.model_construct(
            source_class_function=source_class_function,
            subgroup_partition=subgroup_partition,
            parent_partition=parent_partition,
            subgroup_class_to_parent_class=subgroup_class_to_parent_class,
            induced=induced,
        )


class CharacterRow(StrictModel):
    """One irreducible character row on a retained class partition."""

    label: str = Field(min_length=1, max_length=64)
    degree: int = Field(ge=1, le=MAX_GROUP_ORDER)
    values: tuple[CyclotomicValue, ...] = Field(
        min_length=1, max_length=MAX_CLASS_COUNT
    )


class CharacterTableRequest(StrictModel):
    """Construct a complete table for a supported concrete finite group."""

    partition: GroupConjugacyClassesResult = Field(
        description=(
            "Complete conjugacy-class partition returned by "
            "group.conjugacy_classes.compute; its source group is retained. "
            "The complete-table envelope admits at most "
            f"{MAX_CHARACTER_TABLE_CELLS:,} exact coefficient cells."
        )
    )


class CyclicCharacterRestrictionRequest(StrictModel):
    """Restrict one supported cyclic-group irreducible to its unique C_d."""

    partition: GroupConjugacyClassesResult
    row_index: int = Field(ge=0, le=MAX_CLASS_COUNT - 1, strict=True)
    subgroup_order: int = Field(ge=1, le=MAX_CYCLOTOMIC_ORDER, strict=True)


class CyclicCharacterRestrictionResult(StrictModel):
    """Restricted class function with its subgroup embedding into the source."""

    source_table: CharacterTableResult
    target_partition: ConjugacyClassPartition
    row_index: int = Field(ge=0, le=MAX_CLASS_COUNT - 1)
    source_class_indices: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_CLASS_COUNT
    )
    restricted_character: FiniteClassFunction

    @model_validator(mode="after")
    def require_inclusion_and_values(self) -> Self:
        source_classes = self.source_table.partition.classes
        target_classes = self.target_partition.classes
        if (
            self.target_partition.source.degree
            != self.source_table.partition.source.degree
        ):
            raise _validation_error(
                "restriction_parent", "subgroup action degree must match source"
            )
        if len(self.source_class_indices) != len(target_classes):
            raise _validation_error(
                "restriction_map_shape",
                "one source class index is required per target class",
            )
        if not 0 <= self.row_index < len(self.source_table.rows):
            raise _validation_error(
                "restriction_row", "row index is outside the source table"
            )
        source_row = self.source_table.rows[self.row_index]
        expected_axis = ClassAxis._from_kernel(
            class_sizes=tuple(len(cls) for cls in target_classes),
            cyclotomic_order=self.source_table.axis.cyclotomic_order,
            group=self.target_partition.source,
            class_representatives=tuple(cls[0] for cls in target_classes),
        )
        if self.restricted_character.axis != expected_axis:
            raise _validation_error(
                "restriction_axis", "restricted function must use the target class axis"
            )
        for target_index, target_class in enumerate(target_classes):
            source_index = self.source_class_indices[target_index]
            if not 0 <= source_index < len(source_classes):
                raise _validation_error(
                    "restriction_map_index",
                    "source class index is outside the source partition",
                )
            if not set(target_class).issubset(source_classes[source_index]):
                raise _validation_error(
                    "restriction_inclusion",
                    "target class must map into its declared source class",
                )
            if (
                self.restricted_character.values[target_index]
                != source_row.values[source_index]
            ):
                raise _validation_error(
                    "restriction_value",
                    "restricted value must be pulled back along the class map",
                )
        return self


class ClassPowerMapRequest(StrictModel):
    """Power-map exponent on one source-bound finite permutation group."""

    partition: GroupConjugacyClassesResult = Field(
        description="Complete canonical conjugacy partition of the source group."
    )
    exponent: int = Field(ge=1, le=1_000_000)


class ClassPowerMapResult(StrictModel):
    """Image class index for each class under ``g -> g**exponent``."""

    partition: ConjugacyClassPartition
    exponent: int = Field(ge=1, le=1_000_000)
    image_class_indices: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_CLASS_COUNT
    )

    @model_validator(mode="after")
    def require_class_axis_shape(self) -> Self:
        if len(self.image_class_indices) != len(self.partition.classes):
            raise _validation_error(
                "power_map_shape", "one image class is required per source class"
            )
        if any(
            not 0 <= index < len(self.partition.classes)
            for index in self.image_class_indices
        ):
            raise _validation_error(
                "power_map_index", "image class index is outside the partition"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        partition: ConjugacyClassPartition,
        exponent: int,
        image_class_indices: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            partition=partition,
            exponent=exponent,
            image_class_indices=image_class_indices,
        )


class CharacterTableResult(StrictModel):
    """Complete irreducible character table bound to one group partition."""

    partition: ConjugacyClassPartition
    # The axis is retained independently of the source partition so every
    # exact cell has an explicit class ordering and ambient cyclotomic field.
    axis: ClassAxis
    rows: tuple[CharacterRow, ...] = Field(min_length=1, max_length=MAX_CLASS_COUNT)
    degree_square_sum: int = Field(ge=1)

    @model_validator(mode="after")
    def require_table_shape(self) -> Self:
        classes = self.partition.classes
        if (
            self.axis.class_sizes != tuple(len(cls) for cls in classes)
            or self.axis.group_order != sum(len(cls) for cls in classes)
            or self.axis.group != self.partition.source
            or self.axis.class_representatives != tuple(cls[0] for cls in classes)
        ):
            raise _validation_error(
                "table_axis", "character-table axis must bind to the retained partition"
            )
        if len(self.rows) != len(classes):
            raise _validation_error(
                "table_row_count", "a complete character table needs one row per class"
            )
        if len({row.label for row in self.rows}) != len(self.rows):
            raise _validation_error(
                "table_row_labels", "character-table row labels must be unique"
            )
        if any(len(row.values) != len(classes) for row in self.rows):
            raise _validation_error(
                "table_shape", "every character row must cover every class"
            )
        if any(
            not isinstance(row, CharacterRow)
            or not isinstance(row.values, tuple)
            or any(
                not isinstance(value, CyclotomicValue)
                or value.order != self.axis.cyclotomic_order
                for value in row.values
            )
            for row in self.rows
        ):
            raise _validation_error(
                "table_cyclotomic_axis",
                "every character-table value must use the declared cyclotomic order",
            )
        if self.degree_square_sum != sum(row.degree * row.degree for row in self.rows):
            raise _validation_error(
                "degree_square_sum", "degree squares must sum to the row degrees"
            )
        if self.degree_square_sum != sum(len(cls) for cls in classes):
            raise _validation_error(
                "degree_square_sum",
                "degree squares must sum to the concrete group order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        partition: ConjugacyClassPartition,
        rows: tuple[CharacterRow, ...],
        axis: ClassAxis,
    ) -> Self:
        return cls.model_construct(
            partition=partition,
            axis=axis,
            rows=rows,
            degree_square_sum=sum(row.degree * row.degree for row in rows),
        )


class CharacterTensorDecompositionRequest(StrictModel):
    """Decompose the tensor product of two rows in a supported canonical table."""

    partition: GroupConjugacyClassesResult
    left_row_index: int = Field(ge=0, le=MAX_CLASS_COUNT - 1, strict=True)
    right_row_index: int = Field(ge=0, le=MAX_CLASS_COUNT - 1, strict=True)


class CharacterTensorDecompositionResult(StrictModel):
    """Tensor-product class function and multiplicities in the complete basis."""

    table: CharacterTableResult
    left_row_index: int = Field(ge=0, le=MAX_CLASS_COUNT - 1, strict=True)
    right_row_index: int = Field(ge=0, le=MAX_CLASS_COUNT - 1, strict=True)
    tensor_product: FiniteClassFunction
    multiplicities: tuple[int, ...] = Field(min_length=1, max_length=MAX_CLASS_COUNT)

    @model_validator(mode="after")
    def require_basis_bound_decomposition(self) -> Self:
        row_count = len(self.table.rows)
        if self.left_row_index >= row_count or self.right_row_index >= row_count:
            raise _validation_error(
                "tensor_row_index", "tensor-product row index is outside the table"
            )
        if self.tensor_product.axis != self.table.axis:
            raise _validation_error(
                "tensor_axis", "tensor-product values must use the table class axis"
            )
        if len(self.multiplicities) != row_count or any(
            multiplicity < 0 for multiplicity in self.multiplicities
        ):
            raise _validation_error(
                "tensor_multiplicities",
                "one nonnegative multiplicity is required per table row",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        table: CharacterTableResult,
        left_row_index: int,
        right_row_index: int,
        tensor_product: FiniteClassFunction,
        multiplicities: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            table=table,
            left_row_index=left_row_index,
            right_row_index=right_row_index,
            tensor_product=tensor_product,
            multiplicities=multiplicities,
        )


class FrobeniusSchurIndicatorRequest(StrictModel):
    """Ordinary second indicator of one row in a complete exact table."""

    table: CharacterTableResult = Field(
        description="Complete exact ordinary character table retaining its group."
    )
    row_index: int = Field(ge=0, le=MAX_CLASS_COUNT - 1, strict=True)


class FrobeniusSchurIndicatorResult(StrictModel):
    """Second indicator with its complete table and irreducible row retained."""

    source_table: CharacterTableResult
    row_index: int = Field(ge=0, le=MAX_CLASS_COUNT - 1)
    indicator: int = Field(ge=-1, le=1, strict=True)

    @model_validator(mode="after")
    def require_irreducible_row(self) -> Self:
        if self.row_index >= len(self.source_table.rows):
            raise _validation_error("indicator_row", "row index is outside the table")
        return self


__all__ = [
    "MAX_CHARACTER_TABLE_CELLS",
    "MAX_CLASS_COUNT",
    "MAX_CYCLOTOMIC_ORDER",
    "MAX_GROUP_ORDER",
    "MAX_INNER_PRODUCT_WORK",
    "MAX_VALUE_COEFFICIENT_DIGITS",
    "CharacterRow",
    "CharacterTableRequest",
    "CharacterTableResult",
    "CharacterTensorDecompositionRequest",
    "CharacterTensorDecompositionResult",
    "ClassAxis",
    "ClassContribution",
    "ClassFunctionConjugateRequest",
    "ClassFunctionInductionRequest",
    "ClassFunctionInductionResult",
    "ClassFunctionInnerProductRequest",
    "ClassFunctionInnerProductResult",
    "ClassFunctionPointwiseProductRequest",
    "ClassFunctionRestrictionRequest",
    "ClassFunctionRestrictionResult",
    "ClassPowerMapRequest",
    "ClassPowerMapResult",
    "ConjugacyClassPartition",
    "CyclicCharacterRestrictionRequest",
    "CyclicCharacterRestrictionResult",
    "CyclotomicValue",
    "FiniteClassFunction",
    "FrobeniusSchurIndicatorRequest",
    "FrobeniusSchurIndicatorResult",
]
