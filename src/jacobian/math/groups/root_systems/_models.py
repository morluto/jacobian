"""Typed wire contracts for root system operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.matrices.values import IntegerMatrix

MAX_RANK = 8
MAX_REFLECTION_COORDINATE = ((1 << 53) - 1) // (1 + 3 * MAX_RANK)
MAX_REFLECTION_REPRESENTABLE = (1 << 53) - 1
MAX_POSITIVE_ROOTS = 120
MAX_ROOT_COORDINATE = 6
MAX_COXETER_NUMBER = 30
MAX_WEYL_WORD_LENGTH = 1024
# E8 is the largest finite crystallographic Weyl group at the admitted rank.
MAX_WEYL_GROUP_ORDER = 696_729_600


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by root-system contracts."""

    return PydanticCustomError(f"root_system.{reason}", message)


class CartanMatrix(StrictModel):
    """A canonical ZZ matrix on an ordered simple-root axis.

    Finite-type recognition belongs to root-system operation admission; this
    carrier only establishes the square dimensions and preserves the matrix
    context through serialization.
    """

    matrix: IntegerMatrix
    simple_root_axis: tuple[int, ...] = Field(min_length=1, max_length=MAX_RANK)

    @model_validator(mode="before")
    @classmethod
    def accept_matrix_rows_for_native_projection(cls, data: object) -> object:
        if isinstance(data, (list, tuple)):
            rows = tuple(tuple(value for value in row) for row in data)
            n = len(rows)
            return canonicalize_json_containers(
                {
                    "matrix": IntegerMatrix(
                        row_count=n,
                        column_count=len(rows[0]) if rows else 0,
                        entries=rows,
                    ),
                    "simple_root_axis": tuple(range(n)),
                }
            )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_square_simple_root_axis(self) -> Self:
        n = self.matrix.row_count
        if self.matrix.column_count != n or self.simple_root_axis != tuple(range(n)):
            raise _validation_error(
                "cartan_matrix_shape",
                "Cartan matrix must be square on its simple-root axis",
            )
        return self

    @property
    def entries(self) -> tuple[tuple[int, ...], ...]:
        return self.matrix.entries

    def __len__(self) -> int:
        return self.matrix.row_count

    def __getitem__(self, index: int) -> tuple[int, ...]:
        return self.entries[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CartanMatrix):
            return self.matrix == other.matrix
        if isinstance(other, (list, tuple)):
            return self.entries == tuple(
                tuple(int(value) for value in row) for row in other
            )
        return NotImplemented


class CartanMatrixRequest(StrictModel):
    """A bounded finite-type Cartan matrix."""

    matrix: CartanMatrix = Field(
        description=(
            "Finite-type generalized Cartan matrix of rank 1 through "
            f"{MAX_RANK}: square, diagonal entries 2, non-positive "
            "off-diagonal entries with paired products in {0, 1, 2, 3}, "
            "and positive-definite symmetrization."
        )
    )


CartanType = Literal["A", "B", "C", "D", "E", "F", "G"]

VALID_CARTAN_TYPE_DESCRIPTION = (
    "Finite Dynkin type: A_n (n >= 1), B_n (n >= 2), C_n (n >= 2), "
    "D_n (n >= 4), E_6, E_7, E_8, F_4, G_2."
)


class CartanTypeRequest(StrictModel):
    """A bounded finite Dynkin type and rank."""

    cartan_type: CartanType = Field(
        description=f"Simply-laced or multiply-laced {VALID_CARTAN_TYPE_DESCRIPTION}",
    )
    rank: int = Field(
        ge=1,
        le=MAX_RANK,
        description=(
            "Rank of the root system, from 1 through "
            f"{MAX_RANK}. Only the finite-type pairs A_n (n >= 1), "
            "B_n/C_n (n >= 2), D_n (n >= 4), E_6/E_7/E_8, F_4, and G_2 "
            "are admitted."
        ),
    )


class CartanTypeResult(StrictModel):
    """The Cartan matrix built from a finite Dynkin type and rank."""

    cartan_type: CartanType
    rank: int = Field(ge=1, le=MAX_RANK)
    matrix: CartanMatrix

    @model_validator(mode="after")
    def require_type_result_shape(self) -> Self:
        if self.rank != len(self.matrix):
            raise _validation_error(
                "cartan_type_rank", "rank must equal the Cartan-matrix rank"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        cartan_type: CartanType,
        rank: int,
        matrix: CartanMatrix,
    ) -> Self:
        return cls.model_construct(
            cartan_type=cartan_type,
            rank=rank,
            matrix=matrix,
        )


class PositiveRootsResult(CartanMatrixRequest):
    """The positive roots of a root system."""

    rank: int
    positive_roots: tuple[tuple[int, ...], ...]
    num_positive_roots: int

    @model_validator(mode="after")
    def require_root_shape(self) -> Self:
        if self.rank != len(self.matrix):
            raise _validation_error(
                "positive_roots_rank", "rank must equal the Cartan-matrix rank"
            )
        if self.num_positive_roots != len(self.positive_roots):
            raise _validation_error(
                "positive_roots_count",
                "num_positive_roots must equal the number of positive roots",
            )
        if len(self.positive_roots) > MAX_POSITIVE_ROOTS or any(
            len(root) != self.rank
            or any(
                coordinate < 0 or coordinate > MAX_ROOT_COORDINATE
                for coordinate in root
            )
            or not any(root)
            for root in self.positive_roots
        ):
            raise _validation_error(
                "positive_roots_shape",
                "positive roots must be nonzero bounded vectors on the Cartan axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matrix: CartanMatrix,
        positive_roots: tuple[tuple[int, ...], ...],
    ) -> Self:
        return cls.model_construct(
            matrix=matrix,
            rank=len(matrix),
            positive_roots=positive_roots,
            num_positive_roots=len(positive_roots),
        )


class RootComponentData(StrictModel):
    simple_root_indices: tuple[int, ...]
    positive_roots: tuple[tuple[int, ...], ...]
    highest_root: tuple[int, ...]
    marks: tuple[int, ...]
    coxeter_number: int


class RootSystemDataResult(StrictModel):
    """Complete root system data from a Cartan matrix."""

    rank: int
    cartan_matrix: CartanMatrix
    positive_roots: tuple[tuple[int, ...], ...]
    negative_roots: tuple[tuple[int, ...], ...]
    simple_roots: tuple[tuple[int, ...], ...]
    num_positive_roots: int
    components: tuple[RootComponentData, ...]

    @model_validator(mode="after")
    def require_root_data_shape(self) -> Self:
        rank = len(self.cartan_matrix)
        if self.rank != rank or self.num_positive_roots != len(self.positive_roots):
            raise _validation_error(
                "root_data_count",
                "root data must agree with its declared rank and count",
            )
        if any(
            len(root) != rank
            or any(
                coordinate < 0 or coordinate > MAX_ROOT_COORDINATE
                for coordinate in root
            )
            or not any(root)
            for root in self.positive_roots
        ) or any(
            len(root) != rank
            or any(
                coordinate < -MAX_ROOT_COORDINATE or coordinate > 0
                for coordinate in root
            )
            or not any(root)
            for root in self.negative_roots
        ):
            raise _validation_error(
                "root_data_root_shape",
                "roots must be nonzero bounded vectors on the Cartan axis",
            )
        if self.simple_roots != tuple(
            tuple(int(i == j) for j in range(rank)) for i in range(rank)
        ):
            raise _validation_error(
                "simple_roots", "simple roots must be the canonical Cartan basis"
            )
        indices = [
            index
            for component in self.components
            for index in component.simple_root_indices
        ]
        if tuple(sorted(indices)) != tuple(range(rank)) or any(
            not component.positive_roots
            or tuple(sorted(set(component.simple_root_indices)))
            != component.simple_root_indices
            or any(not 0 <= index < rank for index in component.simple_root_indices)
            or any(
                len(root) != rank
                or any(
                    coordinate < 0 or coordinate > MAX_ROOT_COORDINATE
                    for coordinate in root
                )
                or not any(root)
                for root in component.positive_roots
            )
            or len(component.highest_root) != rank
            or any(
                coordinate < 0 or coordinate > MAX_ROOT_COORDINATE
                for coordinate in component.highest_root
            )
            or len(component.marks) != len(component.simple_root_indices)
            or any(mark < 0 or mark > MAX_ROOT_COORDINATE for mark in component.marks)
            or component.coxeter_number < 2
            or component.coxeter_number > MAX_COXETER_NUMBER
            for component in self.components
        ):
            raise _validation_error(
                "component_data_shape",
                "components must be a nonempty partition with bounded root data",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matrix: CartanMatrix,
        *,
        positive_roots: tuple[tuple[int, ...], ...],
        negative_roots: tuple[tuple[int, ...], ...],
        simple_roots: tuple[tuple[int, ...], ...],
        components: tuple[RootComponentData, ...],
    ) -> Self:
        return cls.model_construct(
            rank=len(matrix),
            cartan_matrix=matrix,
            positive_roots=positive_roots,
            negative_roots=negative_roots,
            simple_roots=simple_roots,
            num_positive_roots=len(positive_roots),
            components=components,
        )


class SimpleReflectionRequest(StrictModel):
    """One bounded simple reflection on a finite-type root lattice."""

    matrix: CartanMatrix = Field(
        min_length=1,
        max_length=MAX_RANK,
        description=(
            "Finite-type generalized Cartan matrix of rank 1 through "
            f"{MAX_RANK}; it must meet the same Cartan conditions as "
            "``CartanMatrixRequest.matrix``."
        ),
    )
    vector: tuple[
        Annotated[
            int,
            Field(ge=-MAX_REFLECTION_REPRESENTABLE, le=MAX_REFLECTION_REPRESENTABLE),
        ],
        ...,
    ] = Field(
        min_length=1,
        description=(
            "Root-lattice coordinates in the matrix's simple-root basis; "
            "length must equal the Cartan-matrix rank, and each coordinate "
            f"must lie in [-{MAX_REFLECTION_REPRESENTABLE}, "
            f"{MAX_REFLECTION_REPRESENTABLE}]. Inputs within "
            f"[-{MAX_REFLECTION_COORDINATE}, {MAX_REFLECTION_COORDINATE}] "
            "are guaranteed a representable image; larger representable "
            "inputs are admitted when their exact reflection also fits."
        ),
    )
    simple_index: int = Field(
        ge=0,
        description=(
            "Zero-based simple-root index; it must be smaller than the "
            "Cartan-matrix rank."
        ),
    )


class SimpleReflectionResult(StrictModel):
    """Result of applying a simple reflection to a vector."""

    matrix: CartanMatrix
    vector: tuple[int, ...]
    simple_index: int
    reflected_vector: tuple[int, ...]

    @model_validator(mode="after")
    def require_reflection_shape(self) -> Self:
        if len(self.reflected_vector) != len(self.vector) or any(
            abs(coordinate) > MAX_REFLECTION_REPRESENTABLE
            for coordinate in self.reflected_vector
        ):
            raise _validation_error(
                "reflected_vector_shape",
                "reflected_vector must fit the bounded root-lattice axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matrix: CartanMatrix,
        vector: tuple[int, ...],
        simple_index: int,
        reflected_vector: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            matrix=matrix,
            vector=vector,
            simple_index=simple_index,
            reflected_vector=reflected_vector,
        )


class WeylGroupOrderResult(StrictModel):
    """The exact order of the Weyl group generated by one Cartan matrix."""

    matrix: CartanMatrix
    group_order: int = Field(ge=1, le=MAX_WEYL_GROUP_ORDER)

    @classmethod
    def _from_kernel(cls, matrix: CartanMatrix, group_order: int) -> Self:
        return cls.model_construct(
            matrix=matrix,
            group_order=group_order,
        )


class WeylElementRequest(StrictModel):
    """One bounded word in the simple reflections of a finite Weyl group."""

    matrix: CartanMatrix = Field(
        description=(
            "Finite-type generalized Cartan matrix of rank 1 through "
            f"{MAX_RANK}; it must meet the same Cartan conditions as "
            "``CartanMatrixRequest.matrix``."
        ),
    )
    word: tuple[Annotated[int, Field(ge=0)], ...] = Field(
        max_length=MAX_WEYL_WORD_LENGTH,
        description=(
            "Simple-reflection indices applied left to right, zero-based; "
            "each index must be below the Cartan-matrix rank and the word "
            f"holds at most {MAX_WEYL_WORD_LENGTH} factors."
        ),
    )


class WeylElementLengthResult(StrictModel):
    """The length, reducedness, and inversion set of a Weyl-group word."""

    matrix: CartanMatrix
    word: tuple[int, ...]
    length: int = Field(ge=0, le=MAX_POSITIVE_ROOTS)
    is_reduced: bool
    inversions: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def require_length_shape(self) -> Self:
        rank = len(self.matrix)
        if (
            self.length != len(self.inversions)
            or self.is_reduced != (self.length == len(self.word))
            or self.inversions != tuple(sorted(set(self.inversions)))
            or any(
                len(root) != rank
                or any(
                    coordinate < 0 or coordinate > MAX_ROOT_COORDINATE
                    for coordinate in root
                )
                or not any(root)
                for root in self.inversions
            )
        ):
            raise _validation_error(
                "weyl_length_shape",
                "length must count distinct sorted positive-root inversions "
                "and reducedness must compare it with the word length",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matrix: CartanMatrix,
        word: tuple[int, ...],
        length: int,
        is_reduced: bool,
        inversions: tuple[tuple[int, ...], ...],
    ) -> Self:
        return cls.model_construct(
            matrix=matrix,
            word=word,
            length=length,
            is_reduced=is_reduced,
            inversions=inversions,
        )


class WeylLongestElementResult(StrictModel):
    """A reduced word for the longest Weyl-group element.

    The longest element is the unique element sending every positive
    root negative, so its length equals the positive-root count and
    the word is reduced.
    """

    matrix: CartanMatrix
    word: tuple[int, ...]
    length: int = Field(ge=1, le=MAX_POSITIVE_ROOTS)
    num_positive_roots: int = Field(ge=1, le=MAX_POSITIVE_ROOTS)

    @model_validator(mode="after")
    def require_longest_shape(self) -> Self:
        if (
            self.length != self.num_positive_roots
            or self.length != len(self.word)
            or any(
                type(index) is not int or index < 0 or index >= len(self.matrix)
                for index in self.word
            )
        ):
            raise _validation_error(
                "weyl_longest_shape",
                "a longest word is reduced and as long as the positive-root "
                "count, with indices below the Cartan rank",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matrix: CartanMatrix,
        word: tuple[int, ...],
        length: int,
        num_positive_roots: int,
    ) -> Self:
        return cls.model_construct(
            matrix=matrix,
            word=word,
            length=length,
            num_positive_roots=num_positive_roots,
        )


class WeylDescentsResult(StrictModel):
    """The left and right descent sets of a Weyl-group word."""

    matrix: CartanMatrix
    word: tuple[int, ...]
    left_descents: tuple[int, ...]
    right_descents: tuple[int, ...]

    @model_validator(mode="after")
    def require_descents_shape(self) -> Self:
        rank = len(self.matrix)
        for descents in (self.left_descents, self.right_descents):
            if descents != tuple(sorted(set(descents))) or any(
                type(index) is not int or index < 0 or index >= rank
                for index in descents
            ):
                raise _validation_error(
                    "weyl_descents_shape",
                    "descent sets must be sorted index sets below the Cartan rank",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matrix: CartanMatrix,
        word: tuple[int, ...],
        left_descents: tuple[int, ...],
        right_descents: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            matrix=matrix,
            word=word,
            left_descents=left_descents,
            right_descents=right_descents,
        )


class SimpleReflectionsResult(StrictModel):
    """One exact simple-reflection matrix per simple root and lattice."""

    matrix: CartanMatrix
    rank: int = Field(ge=1, le=MAX_RANK)
    root_matrices: tuple[IntegerMatrix, ...]
    coroot_matrices: tuple[IntegerMatrix, ...]
    weight_matrices: tuple[IntegerMatrix, ...]
    coweight_matrices: tuple[IntegerMatrix, ...]
    involution_verified: bool

    @model_validator(mode="after")
    def require_reflection_family_shape(self) -> Self:
        families = (
            self.root_matrices,
            self.coroot_matrices,
            self.weight_matrices,
            self.coweight_matrices,
        )
        if any(len(family) != self.rank for family in families) or any(
            reflection.row_count != self.rank or reflection.column_count != self.rank
            for family in families
            for reflection in family
        ):
            raise _validation_error(
                "reflection_family_shape",
                "each lattice must carry one square reflection matrix per simple root",
            )
        if self.involution_verified is not True:
            raise _validation_error(
                "reflection_involution",
                "reflection families must square to the identity",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matrix: CartanMatrix,
        *,
        root_matrices: tuple[IntegerMatrix, ...],
        coroot_matrices: tuple[IntegerMatrix, ...],
        weight_matrices: tuple[IntegerMatrix, ...],
        coweight_matrices: tuple[IntegerMatrix, ...],
    ) -> Self:
        return cls.model_construct(
            matrix=matrix,
            rank=len(matrix),
            root_matrices=root_matrices,
            coroot_matrices=coroot_matrices,
            weight_matrices=weight_matrices,
            coweight_matrices=coweight_matrices,
            involution_verified=True,
        )
