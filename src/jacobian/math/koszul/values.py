"""Provider-independent exact values for sequence-derived Koszul complexes.

Version 1 owns the polynomial-ring slice: one ordered sequence
``f = (f_1, ..., f_c)`` of elements of ``R = QQ[x_1, ..., x_m]`` with bounded
``m``, degree, and term count, and the module fixed to the ring itself, so the
Koszul complex is ``K(f) = R (x) Lambda(R^c)``. Each degree-``k`` chain group is
the free ``R``-module on the ``C(c, k)`` wedge basis elements
``e_{i_1} ^ ... ^ e_{i_k}`` with ``i_1 < ... < i_k`` in canonical increasing
order, and each differential ``d_k(e_I) = sum_j (-1)^{position(j)} f_{I_j}
e_{I \\ {j}}`` is an exact sparse matrix bound to the ordered wedge bases.
Finite-dimensional commutative-algebra modules also have an exact sequence-
derived complex, homology-dimension profile, and degree-zero quotient module.
DG-algebra structure, sequence transforms, cycle representatives, and
polynomial-ring module homology remain deferred.
"""

from __future__ import annotations

from math import comb
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleKoszulComplex,
    ModuleKoszulHomology,
    ModuleKoszulHomologyDegree,
    ModuleQuotientValue,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalPolynomial,
)
from jacobian.math.topology.chain_complexes.values import ChainComplexValue

MAX_KOSZUL_SEQUENCE_LENGTH = 8
MAX_KOSZUL_VARIABLES = min(4, MAX_POLYNOMIAL_VARIABLES)
MAX_KOSZUL_TERMS = 64
MAX_KOSZUL_DEGREE = 16
MAX_KOSZUL_COEFFICIENT_DIGITS = 64
# Derived expansion ceilings: sum_k C(8, k) = 256 wedge basis elements and
# sum_k C(8, k) * k = 8 * 2^7 = 1024 differential contributions.
MAX_KOSZUL_TOTAL_BASIS = 256
MAX_KOSZUL_DIFFERENTIAL_ENTRIES = 1_024
# The d^2 = 0 replay multiplies pairs of sequence elements; every product
# stays inside the squared term bound, and the aggregate replay work is
# preflighted in the shared admission helper.
MAX_KOSZUL_PRODUCT_TERMS = MAX_KOSZUL_TERMS * MAX_KOSZUL_TERMS
MAX_KOSZUL_REPLAY_TERM_WORK = 4_000_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"koszul.{reason}", message)


def _require_wedge_bases(
    length: int,
    degrees: tuple[tuple[tuple[int, ...], ...], ...],
    basis_sizes: tuple[int, ...],
) -> None:
    """Bind every retained wedge basis to its canonical degree card."""

    for degree, basis in enumerate(degrees):
        if len(basis) != basis_sizes[degree]:
            raise _validation_error(
                "basis_cardinality",
                "each degree must list exactly its basis cardinality",
            )
        if len(basis) != comb(length, degree):
            raise _validation_error(
                "basis_binomial_cardinality",
                "degree k of a length-c Koszul complex has exactly "
                "C(c, k) wedge basis elements",
            )
        if basis != tuple(sorted(set(basis))):
            raise _validation_error(
                "basis_order",
                "wedge basis elements must be unique increasing "
                "sequence-index tuples in lexicographic order",
            )
        if any(
            len(indices) != degree or any(not 0 <= index < length for index in indices)
            for indices in basis
        ):
            raise _validation_error(
                "basis_indices",
                "wedge basis indices must be sequence positions of the "
                "declared homological degree",
            )


class KoszulDifferentialEntry(StrictModel):
    """One nonzero sparse cell of a Koszul differential matrix."""

    row: StrictInt = Field(ge=0)
    column: StrictInt = Field(ge=0)
    polynomial: RationalPolynomial


class KoszulDifferentialMatrix(StrictModel):
    """One exact sparse differential ``d_k`` bound to ordered wedge bases.

    Rows index the degree ``k-1`` wedge basis and columns index the degree
    ``k`` wedge basis; ``entry.polynomial`` is the exact coefficient of the
    target basis element applied to the source basis element.
    """

    variables: tuple[PolynomialVariable, ...] = Field(
        default=(), max_length=MAX_KOSZUL_VARIABLES
    )
    row_count: StrictInt = Field(ge=0)
    column_count: StrictInt = Field(ge=0)
    entries: tuple[KoszulDifferentialEntry, ...] = Field(
        default=(),
        max_length=MAX_KOSZUL_DIFFERENTIAL_ENTRIES,
        description=(
            "Nonzero cells in strictly increasing (row, column) order. "
            "Omitted cells are exact zero coefficients."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_sparse_matrix(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "differential_variables", "differential variables must be unique"
            )
        coordinates = tuple((entry.row, entry.column) for entry in self.entries)
        if coordinates != tuple(sorted(set(coordinates))):
            raise _validation_error(
                "differential_entry_order",
                "differential entries must be unique and sorted by "
                "(row, column); omitted cells are exact zeros",
            )
        for entry in self.entries:
            if entry.row >= self.row_count or entry.column >= self.column_count:
                raise _validation_error(
                    "differential_entry_axis",
                    "every differential entry must lie inside the declared "
                    "target-by-source matrix axes",
                )
            if entry.polynomial.variables != self.variables:
                raise _validation_error(
                    "differential_entry_ring",
                    "every differential coefficient must belong to the "
                    "declared ordered polynomial ring",
                )
            if not entry.polynomial.polynomial.terms:
                raise _validation_error(
                    "differential_zero_entry",
                    "zero coefficients must be omitted from the sparse matrix",
                )
        return self


class KoszulComplexValue(StrictModel):
    """The complete exact Koszul complex ``K(f)`` of one retained sequence.

    ``degrees[k]`` lists every wedge basis element of homological degree ``k``
    as the increasing tuple of sequence indices, in lexicographic order.
    ``differentials[k - 1]`` is the sparse matrix of ``d_k`` from degree ``k``
    to degree ``k - 1``. The kernel replays ``d^2 = 0`` exactly before
    publication; result construction does not recompute it.
    ``chain_complex`` retains the exact conversion to the shared based
    chain-complex value when the ambient ring is ``QQ`` (no variables) and the
    wedge ranks fit the shared chain-complex envelope; otherwise it is ``None``
    and homology composes through the retained sparse polynomial matrices.
    """

    domain: Literal["QQ"] = "QQ"
    variables: tuple[PolynomialVariable, ...] = Field(
        default=(), max_length=MAX_KOSZUL_VARIABLES
    )
    sequence: tuple[RationalPolynomial, ...] = Field(
        default=(),
        max_length=MAX_KOSZUL_SEQUENCE_LENGTH,
        description=(
            "The ordered sequence (f_1, ..., f_c). Order is presentation data "
            "affecting wedge basis signs; a permutation yields an isomorphic "
            "complex, not byte-identical matrices. The empty sequence gives "
            "the identity complex R concentrated in degree 0."
        ),
    )
    degrees: tuple[tuple[tuple[StrictInt, ...], ...], ...] = Field(
        description=(
            "Per-degree wedge bases: degrees[k] lists the C(c, k) increasing "
            "sequence-index tuples in lexicographic order."
        )
    )
    basis_sizes: tuple[StrictInt, ...]
    differentials: tuple[KoszulDifferentialMatrix, ...] = Field(
        description=(
            "One sparse differential per positive degree: differentials[k - 1] "
            "maps degree k into degree k - 1."
        )
    )
    differential_relation: Literal["CONSECUTIVE_DIFFERENTIALS_COMPOSE_TO_ZERO"] = (
        "CONSECUTIVE_DIFFERENTIALS_COMPOSE_TO_ZERO"
    )
    chain_complex: ChainComplexValue | None = None

    @model_validator(mode="after")
    def require_structural_koszul_complex(self) -> Self:
        length = len(self.sequence)
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "complex_variables", "complex variables must be unique"
            )
        if any(
            element.domain != "QQ" or element.variables != self.variables
            for element in self.sequence
        ):
            raise _validation_error(
                "sequence_ring",
                "every sequence element must belong to the declared ordered "
                "QQ polynomial ring",
            )
        if len(self.basis_sizes) != length + 1 or len(self.degrees) != length + 1:
            raise _validation_error(
                "degree_coverage",
                "basis sizes and wedge bases must cover every homological "
                "degree 0..c exactly once",
            )
        if len(self.differentials) != length:
            raise _validation_error(
                "differential_count",
                "a Koszul complex of a length-c sequence has exactly c "
                "differentials d_1..d_c",
            )
        if sum(self.basis_sizes) > MAX_KOSZUL_TOTAL_BASIS:
            raise _validation_error(
                "total_basis_budget",
                "the complete wedge basis exceeds the "
                f"{MAX_KOSZUL_TOTAL_BASIS}-element envelope",
            )
        _require_wedge_bases(length, self.degrees, self.basis_sizes)
        for position, matrix in enumerate(self.differentials):
            if matrix.variables != self.variables:
                raise _validation_error(
                    "differential_ring",
                    "every differential must be bound to the declared polynomial ring",
                )
            if (matrix.row_count, matrix.column_count) != (
                self.basis_sizes[position],
                self.basis_sizes[position + 1],
            ):
                raise _validation_error(
                    "differential_shape",
                    "differential d_k must map the degree-k wedge basis into "
                    "the degree-(k-1) wedge basis",
                )
        if self.chain_complex is not None:
            converted = self.chain_complex
            if (
                converted.coefficient_ring.value != "QQ"
                or converted.prime is not None
                or converted.degree_min != 0
                or converted.degree_max != length
                or converted.basis_sizes != self.basis_sizes
            ):
                raise _validation_error(
                    "chain_conversion_context",
                    "the retained chain-complex conversion must be the exact "
                    "QQ scalar complex on the same wedge ranks and degrees",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        variables: tuple[PolynomialVariable, ...],
        sequence: tuple[RationalPolynomial, ...],
        degrees: tuple[tuple[tuple[int, ...], ...], ...],
        basis_sizes: tuple[int, ...],
        differentials: tuple[KoszulDifferentialMatrix, ...],
        chain_complex: ChainComplexValue | None,
    ) -> Self:
        """Build the result after the admitted kernel established its values.

        The kernel replayed ``d^2 = 0`` exactly; the constructor does not
        recompute the differentials or replay the identity again.
        """

        return cls.model_construct(
            domain="QQ",
            variables=variables,
            sequence=sequence,
            degrees=degrees,
            basis_sizes=basis_sizes,
            differentials=differentials,
            differential_relation="CONSECUTIVE_DIFFERENTIALS_COMPOSE_TO_ZERO",
            chain_complex=chain_complex,
        )


# Finite-module carriers are defined separately to keep the polynomial-ring
# value implementation independent; re-export their canonical names here.
FiniteAlgebra = FiniteCommutativeAlgebra
FiniteModule = BasedFiniteModule

__all__ = [
    "MAX_KOSZUL_COEFFICIENT_DIGITS",
    "MAX_KOSZUL_DEGREE",
    "MAX_KOSZUL_DIFFERENTIAL_ENTRIES",
    "MAX_KOSZUL_PRODUCT_TERMS",
    "MAX_KOSZUL_REPLAY_TERM_WORK",
    "MAX_KOSZUL_SEQUENCE_LENGTH",
    "MAX_KOSZUL_TERMS",
    "MAX_KOSZUL_TOTAL_BASIS",
    "MAX_KOSZUL_VARIABLES",
    "BasedFiniteModule",
    "FiniteAlgebra",
    "FiniteCommutativeAlgebra",
    "FiniteModule",
    "KoszulComplexValue",
    "KoszulDifferentialEntry",
    "KoszulDifferentialMatrix",
    "ModuleKoszulComplex",
    "ModuleKoszulHomology",
    "ModuleKoszulHomologyDegree",
    "ModuleQuotientValue",
]
