"""Exact homogeneous monomial-system operations on algebraic tori."""

from __future__ import annotations

from math import prod

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.algebraic_tori.values import (
    AlgebraicTorusSolutionSubgroup,
    HomogeneousMonomialSystem,
    TorsionCharacterGroup,
)
from jacobian.math.matrices.certified_snf.operations import (
    Matrix,
    certificate_from_reduction,
    identity_matrix,
    matrix_determinant,
    matrix_multiply,
    smith_reduce,
    verify_smith_normal_form_certificate,
)
from jacobian.math.matrices.values import IntegerMatrix


def _matrix_value(
    entries: Matrix,
    *,
    rows: int,
    columns: int,
) -> IntegerMatrix:
    return IntegerMatrix(
        row_count=rows,
        column_count=columns,
        entries=tuple(
            tuple(format_canonical_integer(value) for value in row) for row in entries
        ),
    )


def _transpose(matrix: Matrix, *, rows_if_empty: int) -> Matrix:
    if not matrix:
        return [[] for _ in range(rows_if_empty)]
    return [
        [matrix[row][column] for row in range(len(matrix))]
        for column in range(len(matrix[0]))
    ]


def _lll_free_basis(smith_free_map: Matrix) -> tuple[Matrix, Matrix]:
    """Return ``(R,C)`` with ``R = F C`` and unimodular parameter map ``C``."""

    coordinate_count = len(smith_free_map)
    free_rank = len(smith_free_map[0]) if smith_free_map else 0
    if free_rank == 0:
        return ([[] for _ in range(coordinate_count)], [])

    from sympy import QQ, ZZ
    from sympy.polys.matrices import DomainMatrix

    smith_basis_rows = _transpose(smith_free_map, rows_if_empty=free_rank)
    domain = DomainMatrix.from_list_sympy(
        free_rank,
        coordinate_count,
        smith_basis_rows,
    ).convert_to(ZZ)
    reduced, row_transport = domain.lll_transform(delta=QQ(3, 4))
    reduced_rows = [
        [int(value) for value in row] for row in reduced.to_Matrix().tolist()
    ]
    transport_rows = [
        [int(value) for value in row] for row in row_transport.to_Matrix().tolist()
    ]
    if matrix_multiply(transport_rows, smith_basis_rows) != reduced_rows:
        raise ArithmeticError("LLL transport does not reconstruct the reduced basis")
    if abs(matrix_determinant(transport_rows)) != 1:
        raise ArithmeticError("LLL free-basis transport is not unimodular")

    reduced_map = _transpose(reduced_rows, rows_if_empty=coordinate_count)
    smith_parameters_from_reduced = _transpose(transport_rows, rows_if_empty=free_rank)
    if matrix_multiply(smith_free_map, smith_parameters_from_reduced) != reduced_map:
        raise ArithmeticError("free-parameter transport does not reconstruct the map")
    return reduced_map, smith_parameters_from_reduced


def _admit_monomial_system(system: HomogeneousMonomialSystem) -> None:
    """Re-enforce the public execution envelope for native callers.

    A native caller can bypass model validators via ``model_construct`` or
    an unvalidated ``model_copy``, so the owner must re-check the dimension
    and exponent bounds before invoking the certified-SNF backend.
    """

    from jacobian.math.matrices.certified_snf.values import (
        MAX_CERTIFIED_SNF_INPUT_DIGITS,
        MAX_CERTIFIED_SNF_INPUT_DIMENSION,
    )

    matrix = system.exponent_matrix
    if (
        matrix.row_count > MAX_CERTIFIED_SNF_INPUT_DIMENSION
        or matrix.column_count > MAX_CERTIFIED_SNF_INPUT_DIMENSION
    ):
        raise ValueError(
            "monomial systems exponent matrix exceeds the certified-SNF "
            f"dimension bound of {MAX_CERTIFIED_SNF_INPUT_DIMENSION}"
        )
    if any(
        len(value.lstrip("-")) > MAX_CERTIFIED_SNF_INPUT_DIGITS
        for row in matrix.entries
        for value in row
    ):
        raise ValueError(
            "monomial system exponents exceed the certified-SNF digit bound"
        )


def homogeneous_monomial_solution_subgroup(
    system: HomogeneousMonomialSystem,
) -> AlgebraicTorusSolutionSubgroup:
    """Return the exact torsion-by-free-torus subgroup solving ``x^A = 1``."""

    _admit_monomial_system(system)
    source = [
        [parse_canonical_integer(value) for value in row]
        for row in system.exponent_matrix.entries
    ]
    equation_count = system.exponent_matrix.row_count
    coordinate_count = system.exponent_matrix.column_count
    reduction = smith_reduce(
        source,
        row_count=equation_count,
        column_count=coordinate_count,
    )
    rank = reduction.rank
    free_rank = coordinate_count - rank
    certificate = certificate_from_reduction(reduction)

    torsion_positions = tuple(
        index for index, factor in enumerate(reduction.invariant_factors) if factor > 1
    )
    torsion_factors = tuple(
        reduction.invariant_factors[index] for index in torsion_positions
    )
    torsion_map = [
        [
            reduction.right[coordinate][index] % reduction.invariant_factors[index]
            for index in torsion_positions
        ]
        for coordinate in range(coordinate_count)
    ]
    smith_free_map = [
        reduction.right[coordinate][rank:] for coordinate in range(coordinate_count)
    ]
    reduced_free_map, smith_parameters_from_reduced = _lll_free_basis(smith_free_map)

    if matrix_multiply(
        source,
        smith_free_map,
        right_columns_if_empty=free_rank,
    ) != [[0] * free_rank for _ in range(equation_count)]:
        raise ArithmeticError("Smith free parameters do not satisfy the source system")
    if matrix_multiply(
        source,
        reduced_free_map,
        right_columns_if_empty=free_rank,
    ) != [[0] * free_rank for _ in range(equation_count)]:
        raise ArithmeticError(
            "reduced free parameters do not satisfy the source system"
        )
    torsion_relations = matrix_multiply(
        source,
        torsion_map,
        right_columns_if_empty=len(torsion_factors),
    )
    if any(
        value % torsion_factors[column]
        for row in torsion_relations
        for column, value in enumerate(row)
    ):
        raise ArithmeticError("torsion characters do not satisfy the source system")

    return AlgebraicTorusSolutionSubgroup._from_kernel(
        source=system,
        smith_certificate=certificate,
        torsion_character_group=TorsionCharacterGroup(
            invariant_factors=tuple(
                format_canonical_integer(value) for value in torsion_factors
            ),
        ),
        connected_component_count=format_canonical_integer(
            prod(torsion_factors, start=1)
        ),
        torsion_parameter_axis=tuple(
            f"zeta_{index}" for index in range(len(torsion_factors))
        ),
        smith_free_parameter_axis=tuple(
            f"smith_t_{index}" for index in range(free_rank)
        ),
        reduced_free_parameter_axis=tuple(f"t_{index}" for index in range(free_rank)),
        torsion_exponent_map=_matrix_value(
            torsion_map,
            rows=coordinate_count,
            columns=len(torsion_factors),
        ),
        smith_free_exponent_map=_matrix_value(
            smith_free_map,
            rows=coordinate_count,
            columns=free_rank,
        ),
        reduced_free_exponent_map=_matrix_value(
            reduced_free_map,
            rows=coordinate_count,
            columns=free_rank,
        ),
        smith_free_parameters_from_reduced=_matrix_value(
            smith_parameters_from_reduced if free_rank else identity_matrix(0),
            rows=free_rank,
            columns=free_rank,
        ),
        free_rank=free_rank,
    )


def verify_solution_subgroup(claim: AlgebraicTorusSolutionSubgroup) -> bool:
    """Check the complete subgroup parameterization, not LLL reducedness.

    The source and all transformation axes have at most 16 entries. Exact
    scalar bounds come from the canonical Smith and integer-matrix carriers.
    A different unimodular choice of free parameters remains valid.
    """
    _admit_monomial_system(claim.source)
    certificate = claim.smith_certificate
    if not verify_smith_normal_form_certificate(certificate):
        return False
    rank = certificate.rank
    if claim.free_rank != len(claim.source.coordinate_axis) - rank:
        return False
    factors = tuple(map(parse_canonical_integer, certificate.invariant_factors))
    torsion_positions = tuple(i for i, factor in enumerate(factors) if factor > 1)
    torsion = tuple(factors[i] for i in torsion_positions)
    if (
        tuple(
            map(
                parse_canonical_integer, claim.torsion_character_group.invariant_factors
            )
        )
        != torsion
    ):
        return False
    if parse_canonical_integer(claim.connected_component_count) != prod(torsion):
        return False

    def entries(matrix: IntegerMatrix) -> Matrix:
        return [list(map(parse_canonical_integer, row)) for row in matrix.entries]

    right = entries(certificate.right_transformation)
    if entries(claim.torsion_exponent_map) != [
        [row[i] % factors[i] for i in torsion_positions] for row in right
    ]:
        return False
    free = [row[rank:] for row in right]
    if entries(claim.smith_free_exponent_map) != free:
        return False
    change = entries(claim.smith_free_parameters_from_reduced)
    return abs(matrix_determinant(change)) == 1 and matrix_multiply(
        free, change, right_columns_if_empty=claim.free_rank
    ) == entries(claim.reduced_free_exponent_map)


__all__ = ["homogeneous_monomial_solution_subgroup", "verify_solution_subgroup"]
