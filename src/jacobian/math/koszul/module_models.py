"""Finite commutative algebra/module carriers for module Koszul complexes."""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_MODULE_ALGEBRA_DIMENSION = 6
MAX_MODULE_DIMENSION = 8
MAX_MODULE_SEQUENCE_LENGTH = 6
MAX_KOSZUL_DGA_PRODUCT_ENTRIES = 200_000


def _err(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"koszul.module.{reason}", message)


class FiniteCommutativeAlgebra(StrictModel):
    basis: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_MODULE_ALGEBRA_DIMENSION
    )
    multiplication: tuple[tuple[tuple[CanonicalRational, ...], ...], ...]
    unit: tuple[CanonicalRational, ...] | None = None

    @model_validator(mode="after")
    def shape(self) -> Self:
        n = len(self.basis)
        if len(set(self.basis)) != n or len(self.multiplication) != n:
            raise _err(
                "algebra_shape",
                "algebra basis and multiplication dimensions must agree",
            )
        if any(
            len(row) != n or any(len(cell) != n for cell in row)
            for row in self.multiplication
        ):
            raise _err("algebra_shape", "multiplication must be an n by n by n tensor")
        if self.unit is not None and len(self.unit) != n:
            raise _err("unit_shape", "unit coordinates must match the algebra basis")
        return self


class BasedFiniteModule(StrictModel):
    algebra: FiniteCommutativeAlgebra
    # The zero module is needed for quotients by the unit ideal.
    basis: tuple[str, ...] = Field(max_length=MAX_MODULE_DIMENSION)
    # action[a][target][source] is multiplication by algebra basis a
    action: tuple[tuple[tuple[CanonicalRational, ...], ...], ...]

    @model_validator(mode="after")
    def shape(self) -> Self:
        n, m = len(self.algebra.basis), len(self.basis)
        if len(set(self.basis)) != m or len(self.action) != n:
            raise _err(
                "module_shape",
                "module action must have one matrix per algebra basis element",
            )
        if any(
            len(matrix) != m or any(len(row) != m for row in matrix)
            for matrix in self.action
        ):
            raise _err(
                "module_shape",
                "module action matrices must be square on the module basis",
            )
        return self


class ModuleKoszulRequest(StrictModel):
    algebra: FiniteCommutativeAlgebra
    module: BasedFiniteModule
    sequence: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_MODULE_SEQUENCE_LENGTH
    )

    @model_validator(mode="after")
    def binding(self) -> Self:
        if self.module.algebra != self.algebra:
            raise _err(
                "parent_mismatch", "module must be based over the supplied algebra"
            )
        if any(len(element) != len(self.algebra.basis) for element in self.sequence):
            raise _err(
                "sequence_shape", "sequence coordinates must use the algebra basis"
            )
        return self


class ModuleKoszulMapRequest(StrictModel):
    """An exact algebra-linear map between modules on the same sequence."""

    algebra: FiniteCommutativeAlgebra
    source: BasedFiniteModule
    target: BasedFiniteModule
    sequence: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_MODULE_SEQUENCE_LENGTH
    )
    # Rows are target coordinates, columns are source coordinates.
    map_matrix: tuple[tuple[CanonicalRational, ...], ...]

    @model_validator(mode="after")
    def map_axes(self) -> Self:
        rows, columns = len(self.target.basis), len(self.source.basis)
        if (
            self.source.algebra != self.algebra
            or self.target.algebra != self.algebra
            or len(self.map_matrix) != rows
            or any(len(row) != columns for row in self.map_matrix)
            or any(len(element) != len(self.algebra.basis) for element in self.sequence)
        ):
            raise _err(
                "map_axes",
                "module map, algebra, and sequence must use their declared axes",
            )
        return self


class ModuleChainMapMatrix(StrictModel):
    """A sparse exact matrix on one pair of module-wedge axes."""

    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    entries: tuple[tuple[int, int, CanonicalRational], ...] = ()

    @model_validator(mode="after")
    def entry_shape(self) -> Self:
        keys = tuple((row, column) for row, column, _ in self.entries)
        if (
            keys != tuple(sorted(set(keys)))
            or any(
                row < 0
                or row >= self.row_count
                or column < 0
                or column >= self.column_count
                for row, column, _ in self.entries
            )
            or any(value.num == 0 for _, _, value in self.entries)
        ):
            raise _err(
                "chain_map_entries", "chain-map entries must be sorted and in range"
            )
        return self


class ModuleKoszulChainMap(StrictModel):
    """The Koszul chain map induced by a supplied module homomorphism.

    Decoding checks only axes. A future consumer that relies on this map must
    check module-linearity and the chain-map relation against the retained
    complexes; the producer establishes both before returning it.
    """

    algebra: FiniteCommutativeAlgebra
    source: BasedFiniteModule
    target: BasedFiniteModule
    sequence: tuple[tuple[CanonicalRational, ...], ...]
    module_map: tuple[tuple[CanonicalRational, ...], ...]
    source_complex: ModuleKoszulComplex
    target_complex: ModuleKoszulComplex
    degree_maps: tuple[ModuleChainMapMatrix, ...]

    @model_validator(mode="after")
    def map_axes(self) -> Self:
        if (
            self.source.algebra != self.algebra
            or self.target.algebra != self.algebra
            or self.source_complex.algebra != self.algebra
            or self.target_complex.algebra != self.algebra
            or self.source_complex.module != self.source
            or self.target_complex.module != self.target
            or self.source_complex.sequence != self.sequence
            or self.target_complex.sequence != self.sequence
            or len(self.module_map) != len(self.target.basis)
            or any(len(row) != len(self.source.basis) for row in self.module_map)
            or len(self.degree_maps) != len(self.sequence) + 1
        ):
            raise _err(
                "chain_map_binding", "chain map must retain compatible source axes"
            )
        if any(len(element) != len(self.algebra.basis) for element in self.sequence):
            raise _err(
                "chain_map_axes", "sequence coordinates must use the algebra basis"
            )
        for degree, matrix in enumerate(self.degree_maps):
            wedge_size = comb(len(self.sequence), degree)
            expected_source = len(self.source.basis) * wedge_size
            expected_target = len(self.target.basis) * wedge_size
            if (
                matrix.row_count != expected_target
                or matrix.column_count != expected_source
                or self.source_complex.basis_sizes[degree] != expected_source
                or self.target_complex.basis_sizes[degree] != expected_target
            ):
                raise _err(
                    "chain_map_axes",
                    "degree maps must match canonical module-wedge axes",
                )
        return self


class ModuleKoszulDGARequest(StrictModel):
    """Construct the unital Koszul DGA of a finite commutative algebra."""

    algebra: FiniteCommutativeAlgebra
    sequence: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_MODULE_SEQUENCE_LENGTH
    )

    @model_validator(mode="after")
    def sequence_axes(self) -> Self:
        if any(len(element) != len(self.algebra.basis) for element in self.sequence):
            raise _err(
                "sequence_shape", "sequence coordinates must use the algebra basis"
            )
        return self


class ModuleDifferential(StrictModel):
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    entries: tuple[tuple[int, int, CanonicalRational], ...] = ()

    @model_validator(mode="after")
    def entry_shape(self) -> Self:
        keys = tuple((row, column) for row, column, _ in self.entries)
        if keys != tuple(sorted(set(keys))):
            raise _err(
                "differential_entries", "differential entries must be unique and sorted"
            )
        if any(
            row < 0
            or row >= self.row_count
            or column < 0
            or column >= self.column_count
            for row, column, _ in self.entries
        ):
            raise _err(
                "differential_entries",
                "differential entries must lie on the declared axes",
            )
        return self


class ModuleKoszulComplex(StrictModel):
    algebra: FiniteCommutativeAlgebra
    module: BasedFiniteModule
    sequence: tuple[tuple[CanonicalRational, ...], ...]
    basis_sizes: tuple[int, ...]
    differentials: tuple[ModuleDifferential, ...]
    square_zero: bool = True

    @model_validator(mode="after")
    def result_shape(self) -> Self:
        if self.module.algebra != self.algebra or len(self.differentials) != len(
            self.sequence
        ):
            raise _err(
                "result_binding",
                "complex must retain its algebra, module, and sequence",
            )
        if len(self.basis_sizes) != len(self.sequence) + 1:
            raise _err("result_shape", "basis sizes must cover every Koszul degree")
        if any(
            (differential.row_count, differential.column_count)
            != (self.basis_sizes[index], self.basis_sizes[index + 1])
            for index, differential in enumerate(self.differentials)
        ):
            raise _err("result_shape", "differentials must use consecutive Koszul axes")
        return self


class ModuleKoszulSequencePermutationRequest(StrictModel):
    """Permute a retained Koszul sequence using new-index to old-index order."""

    complex: ModuleKoszulComplex
    new_to_old: tuple[int, ...] = Field(max_length=MAX_MODULE_SEQUENCE_LENGTH)

    @model_validator(mode="after")
    def permutation_axes(self) -> Self:
        length = len(self.complex.sequence)
        if len(self.new_to_old) != length or tuple(sorted(self.new_to_old)) != tuple(
            range(length)
        ):
            raise _err(
                "sequence_permutation",
                "new_to_old must be a permutation of the source sequence indices",
            )
        return self


class ModuleKoszulSequencePermutation(StrictModel):
    """A permuted Koszul complex with explicit inverse chain isomorphisms."""

    source_complex: ModuleKoszulComplex
    target_complex: ModuleKoszulComplex
    new_to_old: tuple[int, ...]
    source_to_target: tuple[ModuleDifferential, ...]
    target_to_source: tuple[ModuleDifferential, ...]

    @model_validator(mode="after")
    def isomorphism_axes(self) -> Self:
        length = len(self.source_complex.sequence)
        expected_sequence = (
            tuple(self.source_complex.sequence[index] for index in self.new_to_old)
            if len(self.new_to_old) == length
            else ()
        )
        if (
            tuple(sorted(self.new_to_old)) != tuple(range(length))
            or self.target_complex.sequence != expected_sequence
            or self.target_complex.algebra != self.source_complex.algebra
            or self.target_complex.module != self.source_complex.module
            or self.target_complex.basis_sizes != self.source_complex.basis_sizes
            or len(self.source_to_target) != length + 1
            or len(self.target_to_source) != length + 1
            or any(
                (forward.row_count, forward.column_count)
                != (
                    self.target_complex.basis_sizes[degree],
                    self.source_complex.basis_sizes[degree],
                )
                or (backward.row_count, backward.column_count)
                != (
                    self.source_complex.basis_sizes[degree],
                    self.target_complex.basis_sizes[degree],
                )
                for degree, (forward, backward) in enumerate(
                    zip(self.source_to_target, self.target_to_source, strict=True)
                )
            )
        ):
            raise _err(
                "sequence_permutation_result_axes",
                "permutation complexes and degreewise isomorphisms must share exact axes",
            )
        return self


class ModuleKoszulUnitContractionRequest(StrictModel):
    """Contract a finite-module Koszul complex at a unit sequence entry."""

    complex: ModuleKoszulComplex
    unit_index: int = Field(ge=0, le=MAX_MODULE_SEQUENCE_LENGTH - 1)


class ModuleKoszulUnitContraction(StrictModel):
    """An algebra inverse and degree-raising homotopy contracting a complex."""

    complex: ModuleKoszulComplex
    unit_index: int
    inverse: tuple[CanonicalRational, ...]
    homotopy: tuple[ModuleDifferential, ...]

    @model_validator(mode="after")
    def contraction_axes(self) -> Self:
        sizes = self.complex.basis_sizes
        if (
            self.unit_index < 0
            or self.unit_index >= len(self.complex.sequence)
            or len(self.inverse) != len(self.complex.algebra.basis)
            or len(self.homotopy) != len(sizes)
            or any(
                (mapping.row_count, mapping.column_count)
                != (
                    sizes[degree + 1] if degree + 1 < len(sizes) else 0,
                    sizes[degree],
                )
                for degree, mapping in enumerate(self.homotopy)
            )
        ):
            raise _err(
                "unit_contraction_result_axes",
                "inverse and homotopy maps must use the retained algebra and degree axes",
            )
        return self


class ModuleKoszulZeroExtensionRequest(StrictModel):
    """Append the zero element to a retained finite-module Koszul sequence."""

    complex: ModuleKoszulComplex


class ModuleKoszulZeroExtension(StrictModel):
    """The zero-extended complex split into the source and its degree shift."""

    source_complex: ModuleKoszulComplex
    target_complex: ModuleKoszulComplex
    unshifted_inclusions: tuple[ModuleDifferential, ...]
    unshifted_projections: tuple[ModuleDifferential, ...]
    shifted_inclusions: tuple[ModuleDifferential, ...]
    shifted_projections: tuple[ModuleDifferential, ...]

    @model_validator(mode="after")
    def splitting_axes(self) -> Self:
        source_length = len(self.source_complex.sequence)
        source_sizes = self.source_complex.basis_sizes
        target_sizes = self.target_complex.basis_sizes
        if (
            len(self.target_complex.sequence) != source_length + 1
            or self.target_complex.sequence[:-1] != self.source_complex.sequence
            or any(value.num for value in self.target_complex.sequence[-1])
            or self.target_complex.algebra != self.source_complex.algebra
            or self.target_complex.module != self.source_complex.module
            or len(self.unshifted_inclusions) != source_length + 1
            or len(self.unshifted_projections) != source_length + 2
            or len(self.shifted_inclusions) != source_length + 1
            or len(self.shifted_projections) != source_length + 2
            or any(
                (inc.row_count, inc.column_count) != (target_sizes[k], source_sizes[k])
                for k, inc in enumerate(self.unshifted_inclusions)
            )
            or any(
                (proj.row_count, proj.column_count)
                != (source_sizes[k] if k <= source_length else 0, target_sizes[k])
                for k, proj in enumerate(self.unshifted_projections)
            )
            or any(
                (inc.row_count, inc.column_count)
                != (target_sizes[k + 1], source_sizes[k])
                for k, inc in enumerate(self.shifted_inclusions)
            )
            or any(
                (proj.row_count, proj.column_count)
                != (
                    source_sizes[k - 1] if k > 0 else 0,
                    target_sizes[k],
                )
                for k, proj in enumerate(self.shifted_projections)
            )
        ):
            raise _err(
                "zero_extension_result_axes",
                "zero extension complexes and splitting maps must use canonical degree axes",
            )
        return self


class ModuleKoszulDGAProduct(StrictModel):
    """One nonzero structure coefficient in the Koszul DGA multiplication."""

    left_degree: int = Field(ge=0, le=MAX_MODULE_SEQUENCE_LENGTH)
    left_index: int = Field(ge=0)
    right_degree: int = Field(ge=0, le=MAX_MODULE_SEQUENCE_LENGTH)
    right_index: int = Field(ge=0)
    result_index: int = Field(ge=0)
    coefficient: CanonicalRational

    @model_validator(mode="after")
    def require_nonzero(self) -> Self:
        if self.coefficient.num == 0:
            raise _err("dga_zero_product_entry", "zero products must be omitted")
        return self


class ModuleKoszulDGA(StrictModel):
    """A source-bound unital differential graded algebra ``K(f; A)``."""

    algebra: FiniteCommutativeAlgebra
    sequence: tuple[tuple[CanonicalRational, ...], ...]
    complex: ModuleKoszulComplex
    unit_coordinates: tuple[CanonicalRational, ...]
    products: tuple[ModuleKoszulDGAProduct, ...] = Field(
        max_length=MAX_KOSZUL_DGA_PRODUCT_ENTRIES
    )

    @model_validator(mode="after")
    def require_dga_axes(self) -> Self:
        if (
            self.algebra.unit is None
            or self.unit_coordinates != self.algebra.unit
            or self.complex.algebra != self.algebra
            or self.complex.sequence != self.sequence
            or self.complex.module.algebra != self.algebra
            or self.complex.module.basis != self.algebra.basis
            or len(self.sequence) != len(self.complex.basis_sizes) - 1
        ):
            raise _err(
                "dga_parent", "DGA value must retain its unital algebra and Koszul axes"
            )
        previous_key = None
        for entry in self.products:
            key = (
                entry.left_degree,
                entry.left_index,
                entry.right_degree,
                entry.right_index,
                entry.result_index,
            )
            if previous_key is not None and key <= previous_key:
                raise _err(
                    "dga_product_order",
                    "DGA product entries must be unique and canonical",
                )
            previous_key = key
            if (
                entry.left_degree + entry.right_degree >= len(self.complex.basis_sizes)
                or entry.left_index >= self.complex.basis_sizes[entry.left_degree]
                or entry.right_index >= self.complex.basis_sizes[entry.right_degree]
                or entry.result_index
                >= self.complex.basis_sizes[entry.left_degree + entry.right_degree]
            ):
                raise _err(
                    "dga_product_axis",
                    "DGA product entries must use canonical Koszul axes",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values) -> Self:
        """Construct an admitted DGA without replaying its product table."""
        return cls.model_construct(**values)


class ModuleKoszulHomologyRequest(StrictModel):
    complex: ModuleKoszulComplex


class ModuleKoszulHomologyDegree(StrictModel):
    """Exact bases in one degree, with vectors in the retained chain basis."""

    degree: int = Field(ge=0)
    cycle_basis: tuple[tuple[CanonicalRational, ...], ...]
    boundary_basis: tuple[tuple[CanonicalRational, ...], ...]
    homology_basis: tuple[tuple[CanonicalRational, ...], ...]


class ModuleKoszulHomology(StrictModel):
    complex: ModuleKoszulComplex
    dimensions: tuple[int, ...]
    cycle_dimensions: tuple[int, ...]
    boundary_dimensions: tuple[int, ...]
    degrees: tuple[ModuleKoszulHomologyDegree, ...]

    @model_validator(mode="after")
    def profile_shape(self) -> Self:
        if any(
            len(values) != len(self.complex.basis_sizes)
            for values in (
                self.dimensions,
                self.cycle_dimensions,
                self.boundary_dimensions,
                self.degrees,
            )
        ):
            raise _err("homology_shape", "homology profile must cover every degree")
        for degree, (size, profile) in enumerate(
            zip(self.complex.basis_sizes, self.degrees, strict=True)
        ):
            if (
                profile.degree != degree
                or len(profile.cycle_basis) != self.cycle_dimensions[degree]
                or len(profile.boundary_basis) != self.boundary_dimensions[degree]
                or len(profile.homology_basis) != self.dimensions[degree]
                or any(
                    len(vector) != size
                    for basis in (
                        profile.cycle_basis,
                        profile.boundary_basis,
                        profile.homology_basis,
                    )
                    for vector in basis
                )
            ):
                raise _err(
                    "homology_shape",
                    "homology bases must match their degree and chain axes",
                )
        return self


class ModuleKoszulExactnessProfile(StrictModel):
    """Positive-degree exactness data for one retained Koszul complex."""

    complex: ModuleKoszulComplex
    homology_dimensions: tuple[int, ...]
    acyclic_above_zero: bool
    first_nonzero_degree: int | None
    first_nonzero_class: tuple[CanonicalRational, ...] | None

    @model_validator(mode="after")
    def exactness_shape(self) -> Self:
        positive_dimensions = self.homology_dimensions[1:]
        first = next(
            (
                degree
                for degree, dimension in enumerate(positive_dimensions, start=1)
                if dimension
            ),
            None,
        )
        if (
            len(self.homology_dimensions) != len(self.complex.basis_sizes)
            or any(dimension < 0 for dimension in self.homology_dimensions)
            or self.acyclic_above_zero != (first is None)
            or self.first_nonzero_degree != first
            or (first is None) != (self.first_nonzero_class is None)
            or (
                first is not None
                and (
                    len(self.first_nonzero_class or ())
                    != self.complex.basis_sizes[first]
                    or not any(
                        coefficient.num
                        for coefficient in self.first_nonzero_class or ()
                    )
                )
            )
        ):
            raise _err(
                "exactness_profile_shape",
                "exactness data must match the positive-degree homology axes",
            )
        return self


class ModuleQuotientValue(StrictModel):
    """The exact quotient module ``M/(f_1,...,f_r)M`` with its projection."""

    algebra: FiniteCommutativeAlgebra
    source_module: BasedFiniteModule
    sequence: tuple[tuple[CanonicalRational, ...], ...]
    relation_basis: tuple[tuple[CanonicalRational, ...], ...]
    quotient_module: BasedFiniteModule
    quotient_basis_representatives: tuple[tuple[CanonicalRational, ...], ...]
    # Rows are quotient coordinates and columns are source-module coordinates.
    projection: tuple[tuple[CanonicalRational, ...], ...]

    @model_validator(mode="after")
    def quotient_shape(self) -> Self:
        source_dimension = len(self.source_module.basis)
        quotient_dimension = len(self.quotient_module.basis)
        if (
            self.source_module.algebra != self.algebra
            or self.quotient_module.algebra != self.algebra
            or len(self.quotient_basis_representatives) != quotient_dimension
            or len(self.projection) != quotient_dimension
            or any(len(row) != source_dimension for row in self.projection)
            or any(len(vector) != source_dimension for vector in self.relation_basis)
            or any(
                len(vector) != source_dimension
                for vector in self.quotient_basis_representatives
            )
        ):
            raise _err(
                "quotient_shape", "quotient data must retain compatible module axes"
            )
        return self


class ModuleKoszulDirectSumRequest(StrictModel):
    algebra: FiniteCommutativeAlgebra
    left: BasedFiniteModule
    right: BasedFiniteModule
    sequence: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_MODULE_SEQUENCE_LENGTH
    )

    @model_validator(mode="after")
    def matching_parents(self) -> Self:
        if self.left.algebra != self.algebra or self.right.algebra != self.algebra:
            raise _err("parent_mismatch", "both modules must use the supplied algebra")
        if any(len(element) != len(self.algebra.basis) for element in self.sequence):
            raise _err(
                "sequence_shape", "sequence coordinates must use the algebra basis"
            )
        if len(self.left.basis) + len(self.right.basis) > MAX_MODULE_DIMENSION:
            raise _err(
                "direct_sum_dimension", "direct sum exceeds the module dimension limit"
            )
        return self


class ModuleKoszulDirectSumValue(StrictModel):
    algebra: FiniteCommutativeAlgebra
    left: BasedFiniteModule
    right: BasedFiniteModule
    sequence: tuple[tuple[CanonicalRational, ...], ...]
    direct_sum_module: BasedFiniteModule
    left_complex: ModuleKoszulComplex
    right_complex: ModuleKoszulComplex
    direct_sum_complex: ModuleKoszulComplex
    # Each inclusion matrix is target-basis by source-basis, with the module
    # basis varying fastest inside each canonical increasing wedge axis.
    left_inclusions: tuple[tuple[tuple[CanonicalRational, ...], ...], ...]
    right_inclusions: tuple[tuple[tuple[CanonicalRational, ...], ...], ...]

    @model_validator(mode="after")
    def bound_axes(self) -> Self:
        if (
            self.direct_sum_module.algebra != self.algebra
            or self.left.algebra != self.algebra
            or self.right.algebra != self.algebra
            or self.left_complex.module != self.left
            or self.right_complex.module != self.right
            or self.direct_sum_complex.module != self.direct_sum_module
            or any(
                value != self.sequence
                for value in (
                    self.left_complex.sequence,
                    self.right_complex.sequence,
                    self.direct_sum_complex.sequence,
                )
            )
        ):
            raise _err(
                "direct_sum_binding",
                "direct sum complexes must retain the same algebra and sequence",
            )
        if len(self.left_inclusions) != len(self.direct_sum_complex.basis_sizes) or len(
            self.right_inclusions
        ) != len(self.direct_sum_complex.basis_sizes):
            raise _err(
                "direct_sum_axes", "inclusion maps must cover every Koszul degree"
            )
        for degree, target_size in enumerate(self.direct_sum_complex.basis_sizes):
            for inclusion, source_size in (
                (self.left_inclusions[degree], self.left_complex.basis_sizes[degree]),
                (self.right_inclusions[degree], self.right_complex.basis_sizes[degree]),
            ):
                if len(inclusion) != target_size or any(
                    len(row) != source_size for row in inclusion
                ):
                    raise _err(
                        "direct_sum_axes",
                        "inclusion matrix axes must match complex bases",
                    )
        return self


class ModuleKoszulDifferentialRequest(StrictModel):
    request: ModuleKoszulRequest
    degree: int = Field(ge=1, le=MAX_MODULE_SEQUENCE_LENGTH)

    @model_validator(mode="after")
    def requested_degree_exists(self) -> Self:
        if self.degree > len(self.request.sequence):
            raise _err(
                "differential_degree", "degree must lie between 1 and sequence length"
            )
        return self


class ModuleKoszulDifferentialValue(StrictModel):
    algebra: FiniteCommutativeAlgebra
    module: BasedFiniteModule
    sequence: tuple[tuple[CanonicalRational, ...], ...]
    degree: int = Field(ge=1, le=MAX_MODULE_SEQUENCE_LENGTH)
    source_wedges: tuple[tuple[int, ...], ...]
    target_wedges: tuple[tuple[int, ...], ...]
    differential: ModuleDifferential

    @model_validator(mode="after")
    def bound_axes(self) -> Self:
        if self.module.algebra != self.algebra or self.degree > len(self.sequence):
            raise _err(
                "differential_parent", "differential must retain its source parent"
            )
        if any(len(element) != len(self.algebra.basis) for element in self.sequence):
            raise _err(
                "differential_sequence",
                "sequence coordinates must use the algebra basis",
            )
        expected_source = tuple(combinations(range(len(self.sequence)), self.degree))
        expected_target = tuple(
            combinations(range(len(self.sequence)), self.degree - 1)
        )
        module_dimension = len(self.module.basis)
        if (self.source_wedges, self.target_wedges) != (
            expected_source,
            expected_target,
        ):
            raise _err(
                "differential_axes",
                "wedge axes must be the canonical increasing subsets",
            )
        if (self.differential.row_count, self.differential.column_count) != (
            len(expected_target) * module_dimension,
            len(expected_source) * module_dimension,
        ):
            raise _err(
                "differential_axes", "matrix dimensions must match module-wedge axes"
            )
        return self
