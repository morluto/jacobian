"""Exact bounded native kernel for the Koszul complex construction."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.koszul._admission import (
    SparsePolynomial,
    admit_koszul_construction,
)
from jacobian.math.koszul.values import (
    KoszulComplexValue,
    KoszulDifferentialEntry,
    KoszulDifferentialMatrix,
)
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.math.topology.chain_complexes.operations import construct_chain_complex
from jacobian.math.topology.chain_complexes.values import (
    MAX_BASIS_SIZE,
    MAX_MATRIX_CELLS,
    ChainCoefficient,
    ChainComplexValue,
    CoefficientRing,
)


def _negate(polynomial: SparsePolynomial) -> SparsePolynomial:
    return {exponents: -coefficient for exponents, coefficient in polynomial.items()}


def _multiply(left: SparsePolynomial, right: SparsePolynomial) -> SparsePolynomial:
    product: SparsePolynomial = {}
    for left_exponents, left_coefficient in left.items():
        for right_exponents, right_coefficient in right.items():
            exponents = tuple(
                a + b for a, b in zip(left_exponents, right_exponents, strict=True)
            )
            accumulated = product.get(exponents, Fraction(0)) + (
                left_coefficient * right_coefficient
            )
            if accumulated:
                product[exponents] = accumulated
            else:
                product.pop(exponents, None)
    return product


def _as_value(
    variables: tuple[PolynomialVariable, ...],
    polynomial: SparsePolynomial,
    *,
    negate: bool = False,
) -> RationalPolynomial:
    terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational.from_fraction(
                -coefficient if negate else coefficient
            ),
            exponents=exponents,
        )
        for exponents, coefficient in sorted(
            polynomial.items(), key=lambda item: item[0], reverse=True
        )
    )
    return RationalPolynomial(
        variables=variables, polynomial=SparseRationalPolynomial(terms=terms)
    )


def _scalar_coefficient(polynomial: SparsePolynomial) -> ChainCoefficient:
    """Return one constant polynomial as a native canonical chain coefficient.

    The conversion exists only for the variable-free ambient ring, so each
    polynomial holds at most one term; a unit denominator is spelled as the
    integer the canonical grammar would parse back.
    """

    if not polynomial:
        return 0
    coefficient = next(iter(polynomial.values()))
    return coefficient.numerator if coefficient.denominator == 1 else coefficient


def _converted_chain_complex(
    length: int,
    basis_sizes: tuple[int, ...],
    differentials: tuple[KoszulDifferentialMatrix, ...],
) -> ChainComplexValue | None:
    """Build the exact scalar conversion when the ambient ring is QQ.

    The conversion exists only when it preserves every coordinate: the
    ambient ring must be exactly QQ (no variables, so every differential
    entry is a constant) and the wedge ranks and cell counts must fit the
    shared based chain-complex envelope. Otherwise the conversion is absent
    rather than approximated.
    """

    if any(size > MAX_BASIS_SIZE for size in basis_sizes):
        return None
    cells = sum(
        basis_sizes[index] * basis_sizes[index + 1]
        for index in range(len(basis_sizes) - 1)
    )
    if cells > MAX_MATRIX_CELLS:
        return None
    matrices: list[tuple[tuple[ChainCoefficient, ...], ...]] = []
    for matrix in differentials:
        dense: list[list[ChainCoefficient]] = [
            [0] * matrix.column_count for _ in range(matrix.row_count)
        ]
        for entry in matrix.entries:
            dense[entry.row][entry.column] = _scalar_coefficient(
                {
                    term.exponents: Fraction(term.coefficient.num, term.coefficient.den)
                    for term in entry.polynomial.polynomial.terms
                }
            )
        matrices.append(tuple(tuple(row) for row in dense))
    return construct_chain_complex(
        basis_sizes,
        tuple(matrices),
        coefficient_ring=CoefficientRing.RATIONAL,
    )


def koszul_complex(
    variables: tuple[PolynomialVariable, ...],
    sequence: tuple[RationalPolynomial, ...],
) -> KoszulComplexValue:
    """Construct the complete exact Koszul complex ``K(f)`` of one sequence.

    The module is the ring ``R = QQ[x_1, ..., x_m]`` itself. Degree ``k`` has
    the ``C(c, k)`` wedge basis elements ``e_I`` with ``I`` increasing, and
    ``d_k(e_I) = sum_{j in I} (-1)^{position(j)} f_j e_{I \\ {j}}``. The
    kernel replays the ``d^2 = 0`` identity exactly, charged in the same
    admission as the construction, before publication.
    """

    prepared = admit_koszul_construction(variables, sequence)
    length = len(sequence)

    degrees = tuple(
        tuple(combinations(range(length), degree)) for degree in range(length + 1)
    )
    basis_sizes = tuple(len(basis) for basis in degrees)
    row_index = tuple(
        {indices: position for position, indices in enumerate(basis)}
        for basis in degrees
    )

    differentials: list[KoszulDifferentialMatrix] = []
    for degree in range(1, length + 1):
        entries: list[KoszulDifferentialEntry] = []
        for column, indices in enumerate(degrees[degree]):
            for position, element_index in enumerate(indices):
                if not prepared[element_index]:
                    # A zero sequence entry contributes exact zero cells,
                    # which the sparse representation omits.
                    continue
                target = indices[:position] + indices[position + 1 :]
                entries.append(
                    KoszulDifferentialEntry(
                        row=row_index[degree - 1][target],
                        column=column,
                        polynomial=_as_value(
                            variables,
                            prepared[element_index],
                            negate=position % 2 == 1,
                        ),
                    )
                )
        entries.sort(key=lambda entry: (entry.row, entry.column))
        differentials.append(
            KoszulDifferentialMatrix(
                variables=variables,
                row_count=basis_sizes[degree - 1],
                column_count=basis_sizes[degree],
                entries=tuple(entries),
            )
        )

    _replay_differential_square(length, degrees, prepared)

    chain_complex = None
    if not variables:
        chain_complex = _converted_chain_complex(
            length, basis_sizes, tuple(differentials)
        )

    return KoszulComplexValue._from_kernel(
        variables=variables,
        sequence=sequence,
        degrees=degrees,
        basis_sizes=basis_sizes,
        differentials=tuple(differentials),
        chain_complex=chain_complex,
    )


def _replay_differential_square(
    length: int,
    degrees: tuple[tuple[tuple[int, ...], ...], ...],
    prepared: tuple[SparsePolynomial, ...],
) -> None:
    """Replay ``d_{k-1} d_k = 0`` exactly on every wedge basis element.

    Each pair of sequence positions contributes the two signed orderings
    ``f_i f_j`` and ``f_j f_i`` to the coefficient of the basis element with
    both positions removed; commutativity of ``R`` makes the exact sum zero.
    Pair products are formed once and reused, and every coefficient is
    materialized and checked, never sampled.
    """

    products: dict[tuple[int, int], SparsePolynomial] = {}

    def pair_product(a: int, b: int) -> SparsePolynomial:
        if (a, b) not in products:
            products[(a, b)] = _multiply(prepared[a], prepared[b])
        return products[(a, b)]

    for degree in range(2, length + 1):
        for indices in degrees[degree]:
            coefficients: dict[tuple[int, ...], SparsePolynomial] = {}
            for first in range(degree):
                for second in range(first + 1, degree):
                    i = indices[first]
                    j = indices[second]
                    target = tuple(
                        index
                        for position, index in enumerate(indices)
                        if position not in (first, second)
                    )
                    accumulated = coefficients.setdefault(target, {})
                    total_sign = first + second
                    # Removing i first, then j, contributes
                    # (-1)^(first+second-1) f_j f_i; removing j first, then
                    # i, contributes (-1)^(first+second) f_i f_j. Both
                    # orderings are materialized exactly and summed.
                    for sign, product in (
                        (-1 if total_sign % 2 == 0 else 1, pair_product(j, i)),
                        (-1 if total_sign % 2 else 1, pair_product(i, j)),
                    ):
                        for exponents, coefficient in product.items():
                            total = accumulated.get(exponents, Fraction(0)) + (
                                sign * coefficient
                            )
                            if total:
                                accumulated[exponents] = total
                            else:
                                accumulated.pop(exponents, None)
            for target, wedge_coefficient in coefficients.items():
                if wedge_coefficient:
                    raise OperationDomainValidationError(
                        location=("sequence",),
                        code="koszul.differential_square_replay",
                        message=(
                            "the exact d^2 replay found a nonzero coefficient "
                            f"on wedge basis element {target}"
                        ),
                    )


__all__ = ["koszul_complex"]
