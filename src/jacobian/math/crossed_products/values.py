"""Canonical values for finite-coset crossed products over prime fields."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math._labels import OpaqueLabel

MAX_CHARACTERISTIC = 2_147_483_647
MAX_COSETS = 16
MAX_LATTICE_RANK = 8
MAX_PRESENTATION_INTEGER_DIGITS = 16
MAX_EXPONENT_DIGITS = 64
MAX_ELEMENT_TERMS = 1_024
MAX_PRESENTATION_SCALAR_WORK = 600_000

PresentationInteger = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=MAX_PRESENTATION_INTEGER_DIGITS + 1),
]
ExponentInteger = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=MAX_EXPONENT_DIGITS + 1),
]
QuotientMultiplicationRow = Annotated[
    tuple[OpaqueLabel, ...],
    Field(max_length=MAX_COSETS),
]
ActionMatrixRow = Annotated[
    tuple[PresentationInteger, ...],
    Field(max_length=MAX_LATTICE_RANK),
]
ActionMatrix = Annotated[
    tuple[ActionMatrixRow, ...],
    Field(max_length=MAX_LATTICE_RANK),
]
CocycleVector = Annotated[
    tuple[PresentationInteger, ...],
    Field(max_length=MAX_LATTICE_RANK),
]
CocycleTableRow = Annotated[
    tuple[CocycleVector, ...],
    Field(max_length=MAX_COSETS),
]


def _integer_digits(value: str) -> int:
    return len(value.lstrip("-"))


def _integer_matrix_product(
    left: tuple[tuple[int, ...], ...],
    right: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    dimension = len(left)
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(dimension))
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _integer_matrix_vector_product(
    matrix: tuple[tuple[int, ...], ...],
    vector: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        sum(entry * coordinate for entry, coordinate in zip(row, vector, strict=True))
        for row in matrix
    )


def _add_vectors(*vectors: tuple[int, ...]) -> tuple[int, ...]:
    if not vectors:
        return ()
    return tuple(sum(coordinates) for coordinates in zip(*vectors, strict=True))


class FiniteCosetCrossedProductPresentation(StrictModel):
    """A fully explicit group extension ``Z^d x_c Q`` over ``F_p``.

    A basis element is written ``t^u s(q)`` with exponent column vector
    ``u in Z^d`` and ``q in Q``.  ``action_matrices[i]`` is the matrix of the
    left action ``rho(q_i)`` on exponent columns, and ``cocycle_table[i][j]``
    is ``c(q_i, q_j)``.  Thus basis multiplication is

    ``(u, q_i) (v, q_j) = (u + rho(q_i)v + c(q_i,q_j), q_i q_j)``.

    Validation proves that the quotient table is a group, the matrices define
    an integral group action, and the cocycle is normalized and associative.
    This is explicit finite data; it does not parse or solve a group
    presentation.
    """

    crossed_product_schema_version: Literal["1"] = "1"
    characteristic: StrictInt = Field(
        ge=2,
        le=MAX_CHARACTERISTIC,
        description="Prime p for the coefficient field F_p.",
    )
    lattice_basis: tuple[OpaqueLabel, ...] = Field(
        default=(),
        max_length=MAX_LATTICE_RANK,
        description="Ordered labels for the exponent coordinates of Z^d.",
    )
    cosets: tuple[OpaqueLabel, ...] = Field(
        min_length=1,
        max_length=MAX_COSETS,
        description="Ordered labels for the finite quotient group Q.",
    )
    identity_coset: OpaqueLabel
    quotient_multiplication: tuple[QuotientMultiplicationRow, ...] = Field(
        max_length=MAX_COSETS,
        description=(
            "Square table in coset order: entry [i][j] is cosets[i]*cosets[j]."
        ),
    )
    action_matrices: tuple[ActionMatrix, ...] = Field(
        max_length=MAX_COSETS,
        description=(
            "One d by d integer matrix per coset, acting on exponent column "
            "vectors; entries are canonical decimal strings."
        ),
    )
    cocycle_table: tuple[CocycleTableRow, ...] = Field(
        max_length=MAX_COSETS,
        description=(
            "Square table in coset order whose entries are d-coordinate "
            "canonical integer vectors c(q_i,q_j)."
        ),
    )

    @property
    def lattice_rank(self) -> int:
        return len(self.lattice_basis)

    @model_validator(mode="after")
    def require_crossed_product_presentation(self) -> Self:
        self._require_base_presentation()
        index = {label: position for position, label in enumerate(self.cosets)}
        self._require_quotient_group(index)
        actions = self._require_action_shapes_and_bounds()
        cocycle = self._require_cocycle_shapes_and_bounds()
        self._require_action_laws(index, actions)
        self._require_cocycle_laws(index, actions, cocycle)
        return self

    def _require_base_presentation(self) -> None:
        from sympy import isprime

        if not isprime(self.characteristic):
            raise ValueError("characteristic must be prime")
        if len(set(self.lattice_basis)) != self.lattice_rank:
            raise ValueError("lattice basis labels must be distinct")
        if len(set(self.cosets)) != len(self.cosets):
            raise ValueError("coset labels must be distinct")
        if self.identity_coset not in self.cosets:
            raise ValueError("identity_coset must be a declared coset")

        coset_count = len(self.cosets)
        dimension = self.lattice_rank
        presentation_work = (
            coset_count**3
            + coset_count**2 * dimension**3
            + coset_count**3 * (dimension**2 + 4 * dimension)
            + coset_count * dimension**3
        )
        if presentation_work > MAX_PRESENTATION_SCALAR_WORK:
            raise ValueError(
                "quotient/action/cocycle validation exceeds the bounded "
                "presentation-work budget"
            )

    def _require_action_laws(
        self,
        index: dict[str, int],
        actions: tuple[tuple[tuple[int, ...], ...], ...],
    ) -> None:
        from sympy import Matrix

        coset_count = len(self.cosets)
        dimension = self.lattice_rank
        identity_matrix = tuple(
            tuple(1 if row == column else 0 for column in range(dimension))
            for row in range(dimension)
        )
        identity_index = index[self.identity_coset]
        if actions[identity_index] != identity_matrix:
            raise ValueError("the identity coset action must be the identity matrix")

        for matrix in actions:
            determinant = 1 if dimension == 0 else int(Matrix(matrix).det())
            if determinant not in {-1, 1}:
                raise ValueError("every action matrix must be unimodular over Z")

        for left in range(coset_count):
            for right in range(coset_count):
                product = index[self.quotient_multiplication[left][right]]
                if (
                    _integer_matrix_product(actions[left], actions[right])
                    != actions[product]
                ):
                    raise ValueError(
                        "action matrices must satisfy rho(qr) = rho(q) rho(r)"
                    )

    def _require_cocycle_laws(
        self,
        index: dict[str, int],
        actions: tuple[tuple[tuple[int, ...], ...], ...],
        cocycle: tuple[tuple[tuple[int, ...], ...], ...],
    ) -> None:
        coset_count = len(self.cosets)
        dimension = self.lattice_rank
        identity_index = index[self.identity_coset]
        zero = (0,) * dimension
        for position in range(coset_count):
            if (
                cocycle[identity_index][position] != zero
                or cocycle[position][identity_index] != zero
            ):
                raise ValueError("the cocycle must be normalized at the identity")

        for first in range(coset_count):
            for second in range(coset_count):
                first_second = index[self.quotient_multiplication[first][second]]
                for third in range(coset_count):
                    second_third = index[self.quotient_multiplication[second][third]]
                    left_side = _add_vectors(
                        cocycle[first][second], cocycle[first_second][third]
                    )
                    right_side = _add_vectors(
                        _integer_matrix_vector_product(
                            actions[first], cocycle[second][third]
                        ),
                        cocycle[first][second_third],
                    )
                    if left_side != right_side:
                        raise ValueError(
                            "cocycle must satisfy c(q,r)+c(qr,s) = rho(q)c(r,s)+c(q,rs)"
                        )

    def _require_quotient_group(self, index: dict[str, int]) -> None:
        coset_count = len(self.cosets)
        labels = set(self.cosets)
        if len(self.quotient_multiplication) != coset_count:
            raise ValueError("quotient multiplication must have one row per coset")
        for row in self.quotient_multiplication:
            if len(row) != coset_count:
                raise ValueError("quotient multiplication must be square")
            if any(product not in labels for product in row):
                raise ValueError("every quotient product must be a declared coset")

        identity = index[self.identity_coset]
        for position, label in enumerate(self.cosets):
            if (
                self.quotient_multiplication[identity][position] != label
                or self.quotient_multiplication[position][identity] != label
            ):
                raise ValueError("identity_coset must be a two-sided identity")
            if not any(
                self.quotient_multiplication[position][candidate] == self.identity_coset
                and self.quotient_multiplication[candidate][position]
                == self.identity_coset
                for candidate in range(coset_count)
            ):
                raise ValueError("every quotient coset must have a two-sided inverse")

        for first in range(coset_count):
            for second in range(coset_count):
                first_second = index[self.quotient_multiplication[first][second]]
                for third in range(coset_count):
                    second_third = index[self.quotient_multiplication[second][third]]
                    if (
                        self.quotient_multiplication[first_second][third]
                        != self.quotient_multiplication[first][second_third]
                    ):
                        raise ValueError("quotient multiplication must be associative")

    def _require_action_shapes_and_bounds(
        self,
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        coset_count = len(self.cosets)
        dimension = self.lattice_rank
        if len(self.action_matrices) != coset_count:
            raise ValueError("action_matrices must have one matrix per coset")
        parsed: list[tuple[tuple[int, ...], ...]] = []
        for matrix in self.action_matrices:
            if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
                raise ValueError("every action matrix must have shape d by d")
            if any(
                _integer_digits(entry) > MAX_PRESENTATION_INTEGER_DIGITS
                for row in matrix
                for entry in row
            ):
                raise ValueError(
                    "action entries exceed the 16-digit presentation bound"
                )
            parsed.append(
                tuple(
                    tuple(parse_canonical_integer(entry) for entry in row)
                    for row in matrix
                )
            )
        return tuple(parsed)

    def _require_cocycle_shapes_and_bounds(
        self,
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        coset_count = len(self.cosets)
        dimension = self.lattice_rank
        if len(self.cocycle_table) != coset_count or any(
            len(row) != coset_count for row in self.cocycle_table
        ):
            raise ValueError("cocycle_table must be square in coset order")
        if any(
            len(vector) != dimension for row in self.cocycle_table for vector in row
        ):
            raise ValueError("every cocycle vector must have d coordinates")
        if any(
            _integer_digits(entry) > MAX_PRESENTATION_INTEGER_DIGITS
            for row in self.cocycle_table
            for vector in row
            for entry in vector
        ):
            raise ValueError("cocycle entries exceed the 16-digit presentation bound")
        return tuple(
            tuple(
                tuple(parse_canonical_integer(entry) for entry in vector)
                for vector in row
            )
            for row in self.cocycle_table
        )


class FiniteCosetCrossedProductTerm(StrictModel):
    """One nonzero coefficient on a labelled coset and Laurent exponent."""

    coefficient: StrictInt = Field(ge=1, le=MAX_CHARACTERISTIC - 1)
    coset: OpaqueLabel
    exponents: tuple[ExponentInteger, ...] = Field(
        default=(), max_length=MAX_LATTICE_RANK
    )

    @model_validator(mode="after")
    def require_bounded_exponents(self) -> Self:
        if any(
            _integer_digits(entry) > MAX_EXPONENT_DIGITS for entry in self.exponents
        ):
            raise ValueError("term exponents exceed the 64-digit carrier bound")
        return self


class FiniteCosetCrossedProductElement(StrictModel):
    """A canonical finite-support element of one explicit crossed product.

    Terms omit zero coefficients and are ordered first by the presentation's
    coset order, then by numeric lexicographic exponent order.  The empty term
    tuple is the zero element and still retains its complete parent.
    """

    presentation: FiniteCosetCrossedProductPresentation
    terms: tuple[FiniteCosetCrossedProductTerm, ...] = Field(
        default=(), max_length=MAX_ELEMENT_TERMS
    )

    @model_validator(mode="after")
    def require_canonical_support(self) -> Self:
        coset_index = {
            label: position for position, label in enumerate(self.presentation.cosets)
        }
        keys: list[tuple[int, tuple[int, ...]]] = []
        for term in self.terms:
            if term.coset not in coset_index:
                raise ValueError("every term coset must belong to the presentation")
            if term.coefficient >= self.presentation.characteristic:
                raise ValueError("term coefficients must be canonical residues in F_p")
            if len(term.exponents) != self.presentation.lattice_rank:
                raise ValueError(
                    "every term exponent vector must match the lattice basis"
                )
            keys.append(
                (
                    coset_index[term.coset],
                    tuple(parse_canonical_integer(entry) for entry in term.exponents),
                )
            )
        if len(set(keys)) != len(keys):
            raise ValueError("crossed-product support keys must be unique")
        if keys != sorted(keys):
            raise ValueError(
                "terms must use coset order then numeric lexicographic exponent order"
            )
        return self


__all__ = [
    "FiniteCosetCrossedProductElement",
    "FiniteCosetCrossedProductPresentation",
    "FiniteCosetCrossedProductTerm",
]
