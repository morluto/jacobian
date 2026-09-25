"""Typed wire contracts for root system operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.combinatorics.posets.core._models import FinitePoset
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.polynomials._models import IntegerPolynomial

MAX_RANK = 8
MAX_REFLECTION_COORDINATE = ((1 << 53) - 1) // (1 + 3 * MAX_RANK)
MAX_REFLECTION_REPRESENTABLE = (1 << 53) - 1
MAX_POSITIVE_ROOTS = 120
MAX_ROOT_COORDINATE = 6
MAX_COXETER_NUMBER = 30
MAX_WEYL_WORD_LENGTH = 1024
# E8 is the largest finite crystallographic Weyl group at the admitted rank.
MAX_WEYL_GROUP_ORDER = 696_729_600
# Lagrange bounds every element order by the largest admitted Weyl-group order.
MAX_WEYL_ELEMENT_ORDER = MAX_WEYL_GROUP_ORDER
MAX_WEIGHT_ORBIT_SIZE = 4096
# Each admitted orbit value holds rank coordinates of at most 18 decimal digits.
MAX_WEIGHT_ORBIT_OUTPUT_DIGITS = 1_000_000
MAX_ROOT_POSET_ROOTS = 64
MAX_HIGHEST_WEIGHT_BITS = 64
MAX_WEYL_DIMENSION_BITS = 10_000
MAX_LATTICE_COORDINATE_BITS = 128
MAX_LATTICE_OUTPUT_COORDINATE_BITS = MAX_LATTICE_COORDINATE_BITS + 5
MAX_LATTICE_VECTOR_OUTPUT_CELLS = 8_192


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by root-system contracts."""

    return PydanticCustomError(f"root_system.{reason}", message)


def _cartan_components(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Return the bounded connected components of a Cartan matrix's diagram."""
    remaining = set(range(len(matrix)))
    components: list[tuple[int, ...]] = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component = {first}
        frontier = [first]
        while frontier:
            current = frontier.pop()
            adjacent = {
                index
                for index in remaining
                if matrix[current][index] != 0 or matrix[index][current] != 0
            }
            remaining.difference_update(adjacent)
            component.update(adjacent)
            frontier.extend(adjacent)
        components.append(tuple(sorted(component)))
    return tuple(components)


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


class FiniteCartanDatum(StrictModel):
    """Finite Cartan data with explicit root/coroot and weight bases."""

    cartan_matrix: CartanMatrix
    symmetrizer: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_RANK
    )
    root_to_weight: IntegerMatrix
    coroot_to_coweight: IntegerMatrix

    @model_validator(mode="after")
    def require_basis_shapes(self) -> Self:
        rank = len(self.cartan_matrix)
        if len(self.symmetrizer) != rank:
            raise _validation_error(
                "datum_symmetrizer", "symmetrizer must match Cartan rank"
            )
        for name in ("root_to_weight", "coroot_to_coweight"):
            value = getattr(self, name)
            if value.row_count != rank or value.column_count != rank:
                raise _validation_error(
                    "datum_basis", f"{name} must be a square Cartan-rank map"
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        cartan_matrix: CartanMatrix,
        symmetrizer: tuple[CanonicalRational, ...],
        root_to_weight: IntegerMatrix,
        coroot_to_coweight: IntegerMatrix,
    ) -> Self:
        return cls.model_construct(
            cartan_matrix=cartan_matrix,
            symmetrizer=symmetrizer,
            root_to_weight=root_to_weight,
            coroot_to_coweight=coroot_to_coweight,
        )


class CartanDatumRequest(StrictModel):
    """Request a complete finite Cartan datum from a canonical matrix."""

    matrix: CartanMatrix = Field(
        description="A finite-type Cartan matrix with its simple-root axis."
    )


class LatticeVectorCreateRequest(StrictModel):
    """Create one exact vector in a named basis of a finite Cartan datum."""

    matrix: CartanMatrix
    coordinates: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)


class _FiniteCartanLatticeVector(StrictModel):
    """Shared structure for vectors in four distinct datum-owned lattices."""

    datum: FiniteCartanDatum
    coordinates: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)

    @model_validator(mode="after")
    def require_bounded_datum_axis(self) -> Self:
        if len(self.coordinates) != len(self.datum.cartan_matrix) or any(
            abs(coordinate).bit_length() > MAX_LATTICE_OUTPUT_COORDINATE_BITS
            for coordinate in self.coordinates
        ):
            raise _validation_error(
                "lattice_vector_axis",
                "lattice coordinates must be bounded and match the datum rank",
            )
        return self


class RootLatticeVector(_FiniteCartanLatticeVector):
    """An integral vector in the datum's ordered simple-root basis."""


class CorootLatticeVector(_FiniteCartanLatticeVector):
    """An integral vector in the datum's ordered simple-coroot basis."""


class RootToWeightLatticeRequest(StrictModel):
    """Map a root-lattice vector through the canonical inclusion Q -> P."""

    vector: RootLatticeVector


class CorootToCoweightLatticeRequest(StrictModel):
    """Map a coroot-lattice vector through the canonical inclusion Q^vee -> P^vee."""

    vector: CorootLatticeVector


class WeightLatticeVector(_FiniteCartanLatticeVector):
    """An integral vector in the ordered fundamental-weight basis."""


class CoweightLatticeVector(_FiniteCartanLatticeVector):
    """An integral vector in the ordered fundamental-coweight basis."""


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


class PositiveRootProfileEntry(StrictModel):
    """Height and simple-root support of one positive root."""

    root_coefficients: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)
    height: StrictInt = Field(ge=1, le=MAX_RANK * MAX_ROOT_COORDINATE)
    support_simple_root_indices: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_RANK
    )
    component_index: StrictInt = Field(ge=0, le=MAX_RANK - 1)


class PositiveRootComponentProfile(StrictModel):
    """Root-family indices and highest-root selection for one component."""

    simple_root_indices: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_RANK
    )
    positive_root_indices: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_POSITIVE_ROOTS
    )
    highest_root_index: StrictInt = Field(ge=0, le=MAX_POSITIVE_ROOTS - 1)


class PositiveRootProfileResult(StrictModel):
    """Complete positive-root height/support profile on a Cartan datum."""

    datum: FiniteCartanDatum
    positive_roots: tuple[PositiveRootProfileEntry, ...] = Field(
        min_length=1, max_length=MAX_POSITIVE_ROOTS
    )
    components: tuple[PositiveRootComponentProfile, ...] = Field(
        min_length=1, max_length=MAX_RANK
    )

    @model_validator(mode="after")
    def require_profile_axes(self) -> Self:
        rank = len(self.datum.cartan_matrix)
        roots = tuple(entry.root_coefficients for entry in self.positive_roots)
        if roots != tuple(sorted(set(roots))):
            raise _validation_error(
                "root_profile_order",
                "positive-root profiles must use canonical lexicographic order",
            )
        if any(
            len(entry.root_coefficients) != rank
            or any(
                value < 0 or value > MAX_ROOT_COORDINATE
                for value in entry.root_coefficients
            )
            or not any(entry.root_coefficients)
            or not 1 <= entry.height <= MAX_RANK * MAX_ROOT_COORDINATE
            or tuple(sorted(set(entry.support_simple_root_indices)))
            != entry.support_simple_root_indices
            or any(not 0 <= index < rank for index in entry.support_simple_root_indices)
            or entry.component_index >= len(self.components)
            for entry in self.positive_roots
        ):
            raise _validation_error(
                "root_profile_shape",
                "root profiles must match the datum's ordered simple-root axis",
            )
        factors = tuple(component.simple_root_indices for component in self.components)
        if (
            tuple(sorted(index for factor in factors for index in factor))
            != tuple(range(rank))
            or any(tuple(sorted(set(factor))) != factor for factor in factors)
            or factors != tuple(sorted(factors, key=lambda factor: factor[0]))
        ):
            raise _validation_error(
                "root_profile_components",
                "component factors must partition the simple-root axis",
            )
        root_indices = tuple(
            index
            for component in self.components
            for index in component.positive_root_indices
        )
        if tuple(sorted(root_indices)) != tuple(range(len(self.positive_roots))) or any(
            component.highest_root_index not in component.positive_root_indices
            or tuple(sorted(set(component.positive_root_indices)))
            != component.positive_root_indices
            for component in self.components
        ):
            raise _validation_error(
                "root_profile_component_roots",
                "component root indices must partition the root axis",
            )
        root_component = {
            root_index: component_index
            for component_index, component in enumerate(self.components)
            for root_index in component.positive_root_indices
        }
        if any(
            entry.component_index != root_component[root_index]
            for root_index, entry in enumerate(self.positive_roots)
        ):
            raise _validation_error(
                "root_profile_component_axis",
                "each root profile must name the factor owning its root index",
            )
        return self


class WeylExponentComponent(StrictModel):
    """Exponents of one irreducible factor, retaining its simple-root axis."""

    simple_root_indices: tuple[
        Annotated[StrictInt, Field(ge=0, le=MAX_RANK - 1)], ...
    ] = Field(min_length=1, max_length=MAX_RANK)
    exponents: tuple[
        Annotated[StrictInt, Field(ge=1, le=MAX_RANK * MAX_ROOT_COORDINATE)], ...
    ] = Field(min_length=1, max_length=MAX_RANK)

    @model_validator(mode="after")
    def require_exponent_axis(self) -> Self:
        if (
            tuple(sorted(set(self.simple_root_indices))) != self.simple_root_indices
            or len(self.exponents) != len(self.simple_root_indices)
            or tuple(sorted(self.exponents)) != self.exponents
            or self.exponents[0] < 1
        ):
            raise _validation_error(
                "weyl_exponent_component",
                "Weyl exponents must be nondecreasing and match the component rank",
            )
        return self


class WeylExponentsResult(StrictModel):
    """Weyl exponents, partitioned across the Cartan datum's components."""

    datum: FiniteCartanDatum
    components: tuple[WeylExponentComponent, ...] = Field(
        min_length=1, max_length=MAX_RANK
    )

    @model_validator(mode="after")
    def require_component_partition(self) -> Self:
        matrix = self.datum.cartan_matrix.entries
        remaining = set(range(len(matrix)))
        expected_parts = []
        while remaining:
            stack = [min(remaining)]
            part = set()
            while stack:
                index = stack.pop()
                if index in part:
                    continue
                part.add(index)
                remaining.discard(index)
                stack.extend(
                    neighbor for neighbor in remaining if matrix[index][neighbor] != 0
                )
            expected_parts.append(tuple(sorted(part)))
        expected = tuple(expected_parts)
        if (
            tuple(component.simple_root_indices for component in self.components)
            != expected
        ):
            raise _validation_error(
                "weyl_exponent_components",
                "Weyl exponent factors must preserve the datum component partition",
            )
        return self


class WeylPoincarePolynomialResult(StrictModel):
    """Length-generating polynomial of the Weyl group of a Cartan datum."""

    matrix: CartanMatrix
    polynomial: IntegerPolynomial

    @model_validator(mode="after")
    def require_poincare_polynomial_shape(self) -> Self:
        coefficients = self.polynomial.coefficients
        if (
            len(coefficients) > MAX_POSITIVE_ROOTS + 1
            or any(coefficient <= 0 for coefficient in coefficients)
            or coefficients[0] != 1
            or coefficients[-1] != 1
            or coefficients != tuple(reversed(coefficients))
        ):
            raise _validation_error(
                "weyl_poincare_coefficients",
                "Weyl Poincare-polynomial coefficients must be positive, palindromic, and have unit endpoints",
            )
        return self


class WeylParabolicRequest(StrictModel):
    """A standard parabolic subgroup indexed by simple reflections."""

    matrix: CartanMatrix
    simple_root_indices: tuple[
        Annotated[StrictInt, Field(ge=0, le=MAX_RANK - 1)], ...
    ] = Field(max_length=MAX_RANK)

    @model_validator(mode="after")
    def require_canonical_subset(self) -> Self:
        rank = len(self.matrix)
        if tuple(
            sorted(set(self.simple_root_indices))
        ) != self.simple_root_indices or any(
            index >= rank for index in self.simple_root_indices
        ):
            raise _validation_error(
                "parabolic_simple_indices",
                "parabolic simple-root indices must be a strictly increasing subset of the Cartan axis",
            )
        return self


class WeylParabolicResult(StrictModel):
    """A standard parabolic subgroup embedded in its parent Weyl group."""

    matrix: CartanMatrix
    simple_root_indices: tuple[
        Annotated[StrictInt, Field(ge=0, le=MAX_RANK - 1)], ...
    ] = Field(max_length=MAX_RANK)
    parabolic_cartan_matrix: tuple[tuple[StrictInt, ...], ...] = Field(
        max_length=MAX_RANK
    )
    group_order: StrictInt = Field(ge=1, le=MAX_WEYL_GROUP_ORDER)

    @model_validator(mode="after")
    def require_embedded_subdatum(self) -> Self:
        parent = self.matrix.entries
        indices = self.simple_root_indices
        expected = tuple(tuple(parent[i][j] for j in indices) for i in indices)
        if (
            tuple(sorted(set(indices))) != indices
            or any(index >= len(parent) for index in indices)
            or self.parabolic_cartan_matrix != expected
        ):
            raise _validation_error(
                "parabolic_embedding",
                "parabolic Cartan data must be the principal subdatum on the selected simple-root indices",
            )
        return self


class RootCorootPair(StrictModel):
    """One positive root with its coroot in the matching simple bases."""

    root_coefficients: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)
    coroot_coefficients: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_RANK
    )
    squared_length: CanonicalRational

    @model_validator(mode="after")
    def require_positive_matched_axes(self) -> Self:
        if (
            len(self.root_coefficients) != len(self.coroot_coefficients)
            or not any(self.root_coefficients)
            or not any(self.coroot_coefficients)
            or any(value < 0 for value in self.root_coefficients)
            or any(value < 0 for value in self.coroot_coefficients)
            or self.squared_length.as_fraction() <= 0
        ):
            raise _validation_error(
                "coroot_pair_shape",
                "root and coroot coefficients must be positive-axis vectors with positive squared length",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        root_coefficients: tuple[int, ...],
        coroot_coefficients: tuple[int, ...],
        squared_length: CanonicalRational,
    ) -> Self:
        return cls.model_construct(
            root_coefficients=root_coefficients,
            coroot_coefficients=coroot_coefficients,
            squared_length=squared_length,
        )


class PositiveCorootsResult(StrictModel):
    """Complete positive root-to-coroot data in one finite Cartan datum."""

    datum: FiniteCartanDatum
    positive_root_coroot_pairs: tuple[RootCorootPair, ...] = Field(
        min_length=1, max_length=MAX_POSITIVE_ROOTS
    )

    @model_validator(mode="after")
    def require_complete_axis_shape(self) -> Self:
        rank = len(self.datum.cartan_matrix)
        if any(
            len(pair.root_coefficients) != rank or len(pair.coroot_coefficients) != rank
            for pair in self.positive_root_coroot_pairs
        ):
            raise _validation_error(
                "coroot_table_axis",
                "every root and coroot must use the datum's simple-root rank",
            )
        roots = tuple(
            pair.root_coefficients for pair in self.positive_root_coroot_pairs
        )
        if roots != tuple(sorted(set(roots))):
            raise _validation_error(
                "coroot_table_order",
                "root-coroot pairs must be ordered by distinct canonical root coordinates",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        datum: FiniteCartanDatum,
        pairs: tuple[RootCorootPair, ...],
    ) -> Self:
        return cls.model_construct(datum=datum, positive_root_coroot_pairs=pairs)


class RootLengthClass(StrictModel):
    """Positive roots in one component having the same squared length."""

    squared_length: CanonicalRational
    squared_length_ratio_to_short: CanonicalRational
    roots: tuple[tuple[StrictInt, ...], ...] = Field(
        min_length=1, max_length=MAX_POSITIVE_ROOTS
    )

    @model_validator(mode="after")
    def require_length_class(self) -> Self:
        if (
            self.squared_length.as_fraction() <= 0
            or self.squared_length_ratio_to_short.as_fraction() < 1
            or self.roots != tuple(sorted(set(self.roots)))
            or any(not root or any(value < 0 for value in root) for root in self.roots)
        ):
            raise _validation_error(
                "root_length_class",
                "length classes require ordered positive roots and positive squared lengths",
            )
        return self


class RootLengthComponentProfile(StrictModel):
    """Root-length classes relative to the local normalization of a factor."""

    simple_root_indices: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_RANK
    )
    length_classes: tuple[RootLengthClass, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def require_class_order(self) -> Self:
        lengths = tuple(
            item.squared_length.as_fraction() for item in self.length_classes
        )
        short_length = lengths[0]
        if lengths != tuple(sorted(set(lengths))) or any(
            item.squared_length_ratio_to_short.as_fraction()
            != item.squared_length.as_fraction() / short_length
            for item in self.length_classes
        ):
            raise _validation_error(
                "root_length_component",
                "length classes must be ordered from short to long and normalized at one",
            )
        return self


class RootLengthProfileResult(StrictModel):
    """Exact squared root lengths grouped separately in every irreducible factor."""

    datum: FiniteCartanDatum
    positive_roots: tuple[tuple[StrictInt, ...], ...] = Field(
        min_length=1, max_length=MAX_POSITIVE_ROOTS
    )
    components: tuple[RootLengthComponentProfile, ...] = Field(
        min_length=1, max_length=MAX_RANK
    )

    @model_validator(mode="after")
    def require_root_axis(self) -> Self:
        rank = len(self.datum.cartan_matrix)
        if self.positive_roots != tuple(sorted(set(self.positive_roots))) or any(
            len(root) != rank
            or any(value < 0 or value > MAX_ROOT_COORDINATE for value in root)
            or not any(root)
            for root in self.positive_roots
        ):
            raise _validation_error(
                "root_length_axis", "positive roots must use the canonical datum axis"
            )
        simple_indices = tuple(
            index
            for component in self.components
            for index in component.simple_root_indices
        )
        roots = tuple(
            root
            for component in self.components
            for group in component.length_classes
            for root in group.roots
        )
        factors = tuple(component.simple_root_indices for component in self.components)
        cartan = self.datum.cartan_matrix.entries
        root_factor_valid = all(
            {index for index, coefficient in enumerate(root) if coefficient}
            <= set(component.simple_root_indices)
            for component in self.components
            for group in component.length_classes
            for root in group.roots
        )
        factor_partition_valid = factors == tuple(
            tuple(sorted(component)) for component in _cartan_components(cartan)
        )
        if (
            tuple(sorted(simple_indices)) != tuple(range(rank))
            or len(set(simple_indices)) != rank
            or tuple(sorted(roots)) != self.positive_roots
            or not root_factor_valid
            or not factor_partition_valid
        ):
            raise _validation_error(
                "root_length_partition",
                "components and length classes must partition the datum roots",
            )
        return self


class RootToCorootRequest(CartanMatrixRequest):
    """Convert one positive root in the supplied finite Cartan datum."""

    root_coefficients: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_RANK,
        description=(
            "Positive root coordinates in the Cartan datum's ordered "
            "simple-root basis; each coordinate is bounded by "
            f"{MAX_ROOT_COORDINATE}."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_positive_axis(self) -> Self:
        if (
            len(self.root_coefficients) != len(self.matrix)
            or not any(self.root_coefficients)
            or any(
                coefficient < 0 or coefficient > MAX_ROOT_COORDINATE
                for coefficient in self.root_coefficients
            )
        ):
            raise _validation_error(
                "root_coroot_input_shape",
                "input must be a nonzero bounded vector on the datum's positive simple-root axis",
            )
        return self


class RootToCorootResult(StrictModel):
    """One datum-bound positive root, coroot, and exact squared length."""

    datum: FiniteCartanDatum
    pair: RootCorootPair

    @model_validator(mode="after")
    def require_datum_axis(self) -> Self:
        rank = len(self.datum.cartan_matrix)
        if (
            len(self.pair.root_coefficients) != rank
            or len(self.pair.coroot_coefficients) != rank
        ):
            raise _validation_error(
                "root_coroot_axis",
                "root and coroot coordinates must match the retained Cartan datum",
            )
        return self

    @classmethod
    def _from_kernel(cls, datum: FiniteCartanDatum, pair: RootCorootPair) -> Self:
        return cls.model_construct(datum=datum, pair=pair)


class RootPosetResult(StrictModel):
    """Root poset together with the Cartan context and root-coordinate axis."""

    datum: FiniteCartanDatum
    positive_roots: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_ROOT_POSET_ROOTS
    )
    poset: FinitePoset

    @model_validator(mode="after")
    def require_root_poset_axis(self) -> Self:
        rank = len(self.datum.cartan_matrix)
        roots = self.positive_roots
        labels = tuple(f"root_{index:02d}" for index in range(len(roots)))
        if (
            roots != tuple(sorted(set(roots)))
            or any(
                len(root) != rank
                or any(
                    type(value) is not int or not 0 <= value <= MAX_ROOT_COORDINATE
                    for value in root
                )
                or not any(root)
                for root in roots
            )
            or self.poset.elements != labels
        ):
            raise _validation_error(
                "root_poset_axis",
                "root coordinates and poset labels must share one canonical axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        datum: FiniteCartanDatum,
        positive_roots: tuple[tuple[int, ...], ...],
        poset: FinitePoset,
    ) -> Self:
        return cls.model_construct(
            datum=datum, positive_roots=positive_roots, poset=poset
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


class WeylElement(StrictModel):
    """Canonical Weyl transformation on the Cartan datum's root lattice.

    ``root_action`` is the element itself: its columns are images of the
    ordered simple-root basis. Different words for the same element therefore
    have the same value.
    """

    matrix: CartanMatrix
    root_action: IntegerMatrix

    @model_validator(mode="after")
    def require_action_shape(self) -> Self:
        rank = len(self.matrix)
        if self.root_action.row_count != rank or self.root_action.column_count != rank:
            raise _validation_error(
                "weyl_element_shape",
                "root action must be square on the Cartan root axis",
            )
        return self


class WeylElementComposeRequest(StrictModel):
    """Compose two Weyl elements of the same ordered Cartan parent."""

    first: WeylElement
    then: WeylElement


class WeylElementInverseRequest(StrictModel):
    """Invert one Weyl element."""

    element: WeylElement


class WeylVectorActionRequest(WeylElementRequest):
    """Apply a bounded Weyl word to a vector in the simple-root basis."""

    vector: tuple[
        Annotated[
            int,
            Field(ge=-MAX_REFLECTION_REPRESENTABLE, le=MAX_REFLECTION_REPRESENTABLE),
        ],
        ...,
    ] = Field(min_length=1, max_length=MAX_RANK)


class WeylVectorActionResult(StrictModel):
    """The exact image of a root-lattice vector under a supplied Weyl word."""

    matrix: CartanMatrix
    word: tuple[int, ...]
    vector: tuple[int, ...]
    image: tuple[int, ...]

    @model_validator(mode="after")
    def require_action_shape(self) -> Self:
        rank = len(self.matrix)
        if (
            len(self.vector) != rank
            or len(self.image) != rank
            or len(self.word) > MAX_WEYL_WORD_LENGTH
            or any(
                type(index) is not int or not 0 <= index < rank for index in self.word
            )
            or any(
                type(coordinate) is not int
                or abs(coordinate) > MAX_REFLECTION_REPRESENTABLE
                for coordinate in (*self.vector, *self.image)
            )
        ):
            raise _validation_error(
                "weyl_vector_action_shape",
                "the word and vectors must use the bounded Cartan root axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matrix: CartanMatrix,
        word: tuple[int, ...],
        vector: tuple[int, ...],
        image: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(matrix=matrix, word=word, vector=vector, image=image)


class WeylWeightOrbitRequest(CartanMatrixRequest):
    """An integral weight given in the fundamental-weight basis."""

    weight: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)


class WeylWeightOrbitResult(StrictModel):
    """The complete Weyl orbit of an integral weight in fundamental coordinates."""

    matrix: CartanMatrix
    weight: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)
    orbit: tuple[tuple[StrictInt, ...], ...] = Field(
        min_length=1, max_length=MAX_WEIGHT_ORBIT_SIZE
    )

    @model_validator(mode="after")
    def require_canonical_orbit(self) -> Self:
        rank = len(self.matrix)
        if (
            len(self.weight) != rank
            or len(self.orbit) > MAX_WEIGHT_ORBIT_SIZE
            or any(len(value) != rank for value in self.orbit)
            or self.orbit != tuple(sorted(set(self.orbit)))
            or self.weight not in self.orbit
            or any(
                abs(coordinate) > MAX_REFLECTION_REPRESENTABLE
                for value in (*self.orbit, self.weight)
                for coordinate in value
            )
        ):
            raise _validation_error(
                "weight_orbit_shape",
                "the complete orbit must be sorted, distinct, bounded, and use the Cartan rank",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matrix: CartanMatrix,
        weight: tuple[int, ...],
        orbit: tuple[tuple[int, ...], ...],
    ) -> Self:
        return cls.model_construct(matrix=matrix, weight=weight, orbit=orbit)


class WeylDominantRepresentativeRequest(CartanMatrixRequest):
    """An integral weight in fundamental-weight coordinates."""

    weight: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)


class WeylDominantRepresentativeResult(StrictModel):
    """The dominant orbit representative and a Weyl element mapping to it."""

    matrix: CartanMatrix
    weight: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)
    dominant_weight: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)
    element: WeylElement

    @model_validator(mode="after")
    def require_dominant_representative_shape(self) -> Self:
        rank = len(self.matrix)
        if (
            len(self.weight) != rank
            or len(self.dominant_weight) != rank
            or self.element.matrix != self.matrix
            or any(
                abs(value) > MAX_REFLECTION_REPRESENTABLE
                for value in (*self.weight, *self.dominant_weight)
            )
            or any(value < 0 for value in self.dominant_weight)
        ):
            raise _validation_error(
                "dominant_representative_shape",
                "the source, dominant weight, and Weyl element must share the Cartan axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matrix: CartanMatrix,
        weight: tuple[int, ...],
        dominant_weight: tuple[int, ...],
        element: WeylElement,
    ) -> Self:
        return cls.model_construct(
            matrix=matrix,
            weight=weight,
            dominant_weight=dominant_weight,
            element=element,
        )


class WeylDimensionRequest(CartanMatrixRequest):
    """An integral dominant weight in fundamental-weight coordinates."""

    highest_weight: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)

    @model_validator(mode="after")
    def require_dominant_weight_axis(self) -> Self:
        if len(self.highest_weight) != len(self.matrix) or any(
            value < 0 for value in self.highest_weight
        ):
            raise _validation_error(
                "invalid_dominant_weight",
                "highest weight must be nonnegative and match the Cartan rank",
            )
        return self


class WeylDimensionFactor(StrictModel):
    """One positive-root numerator and denominator in the Weyl product."""

    positive_root: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)
    positive_coroot: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)
    numerator_pairing: StrictInt = Field(gt=0)
    denominator_pairing: StrictInt = Field(gt=0)


class WeylDimensionResult(StrictModel):
    """Exact irreducible dimension bound to its root datum and weight axis."""

    matrix: CartanMatrix
    weight_axis: tuple[int, ...] = Field(min_length=1, max_length=MAX_RANK)
    highest_weight: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)
    dimension: StrictInt = Field(ge=1)
    positive_root_factors: tuple[WeylDimensionFactor, ...] = Field(
        min_length=1, max_length=MAX_POSITIVE_ROOTS
    )

    @model_validator(mode="after")
    def require_axis_and_profile_shape(self) -> Self:
        rank = len(self.matrix)
        if (
            self.weight_axis != tuple(range(rank))
            or len(self.highest_weight) != rank
            or any(value < 0 for value in self.highest_weight)
            or any(
                len(factor.positive_root) != rank or len(factor.positive_coroot) != rank
                for factor in self.positive_root_factors
            )
        ):
            raise _validation_error(
                "weyl_dimension_shape",
                "the dominant weight and every root factor must use the Cartan weight axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matrix: CartanMatrix,
        highest_weight: tuple[int, ...],
        dimension: int,
        positive_root_factors: tuple[WeylDimensionFactor, ...],
    ) -> Self:
        return cls.model_construct(
            matrix=matrix,
            weight_axis=tuple(range(len(matrix))),
            highest_weight=highest_weight,
            dimension=dimension,
            positive_root_factors=positive_root_factors,
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


class WeylElementOrderResult(StrictModel):
    """Exact finite order of a Weyl element specified by a word."""

    matrix: CartanMatrix
    word: tuple[StrictInt, ...] = Field(max_length=MAX_WEYL_WORD_LENGTH)
    order: StrictInt = Field(ge=1, le=MAX_WEYL_ELEMENT_ORDER)

    @model_validator(mode="after")
    def require_word_axis(self) -> Self:
        if any(index < 0 or index >= len(self.matrix) for index in self.word):
            raise _validation_error(
                "weyl_element_order_word",
                "every simple-reflection index must be below the Cartan rank",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matrix: CartanMatrix,
        word: tuple[int, ...],
        order: int,
    ) -> Self:
        return cls.model_construct(matrix=matrix, word=word, order=order)


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
