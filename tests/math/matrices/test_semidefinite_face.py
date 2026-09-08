"""Defining feasible-set and reconstruction identities for exposed faces."""

from collections.abc import Sequence
from fractions import Fraction

import pytest
from pydantic import ValidationError
from sympy import Matrix, Rational, zeros

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.semidefinite import (
    RationalSemidefiniteSystem,
    SemidefiniteFaceReduction,
    reduce_exposed_face,
)
from jacobian.math.matrices.semidefinite._models import SemidefiniteFaceReductionRequest
from jacobian.math.matrices.values import RationalMatrix, rational_matrix_from_fractions


def _q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _matrix(rows: Sequence[Sequence[int | Fraction]]) -> RationalMatrix:
    return rational_matrix_from_fractions([[Fraction(q) for q in row] for row in rows])


def _sympy(matrix: RationalMatrix) -> Matrix:
    return Matrix(
        matrix.row_count,
        matrix.column_count,
        [Rational(q.num, q.den) for row in matrix.entries for q in row],
    )


def _system(
    matrices: tuple[RationalMatrix, ...], rhs: tuple[int, ...]
) -> RationalSemidefiniteSystem:
    return RationalSemidefiniteSystem(
        order=matrices[0].row_count, matrices=matrices, rhs=tuple(map(_q, rhs))
    )


def _identities(result: SemidefiniteFaceReduction) -> None:
    w, v = _sympy(result.exposing_matrix), _sympy(result.embedding)
    source = result.source
    assert w == sum(
        (
            _qv.as_fraction() * _sympy(a)
            for _qv, a in zip(result.multipliers, source.matrices, strict=True)
        ),
        zeros(source.order),
    )
    assert w * v == zeros(source.order, v.cols)
    assert v.rank() == v.cols == source.order - w.rank()
    assert w.is_positive_semidefinite
    for a, compressed in zip(source.matrices, result.reduced.matrices, strict=True):
        assert v.T * _sympy(a) * v == _sympy(compressed)
    assert result.reduced.rhs == source.rhs


def test_issue_fixture_and_psd_lift() -> None:
    system = _system((_matrix([[1, 0], [0, 0]]), _matrix([[1, 0], [0, 1]])), (0, 1))
    result = reduce_exposed_face(system, (_q(1), _q(0)))
    _identities(result)
    assert _sympy(result.embedding) == Matrix([[0], [1]])
    assert [_sympy(a) for a in result.reduced.matrices] == [
        Matrix([[0]]),
        Matrix([[1]]),
    ]
    assert _sympy(result.embedding) * _sympy(result.embedding).T == Matrix(
        [[0, 0], [0, 1]]
    )


def test_noncoordinate_kernel_preserves_all_feasible_psd_matrices() -> None:
    # W annihilates (-2,1,0) and (0,0,1); rational nonorthonormal V is enough.
    system = _system(
        (
            _matrix([[1, 2, 0], [2, 4, 0], [0, 0, 0]]),
            _matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ),
        (0, 5),
    )
    result = reduce_exposed_face(system, (_q(1), _q(0)))
    _identities(result)
    v = _sympy(result.embedding)
    for a, b in [(0, 0), (1, 0), (0, 2), (2, -1)]:
        z = Matrix([a, b])
        y = z * z.T
        x = v * y * v.T
        assert x.is_positive_semidefinite
        assert (_sympy(result.exposing_matrix) * x).trace() == 0
        for original, reduced in zip(
            system.matrices, result.reduced.matrices, strict=True
        ):
            assert (_sympy(original) * x).trace() == (_sympy(reduced) * y).trace()
        # The left inverse recovers the unique reduced PSD coordinate matrix.
        left_inverse = (v.T * v).inv() * v.T
        assert left_inverse * x * left_inverse.T == y


def test_full_rank_face_retains_contradictory_equalities_and_empty_axes() -> None:
    system = _system((_matrix([[1, 0], [0, 1]]), _matrix([[0, 0], [0, 0]])), (0, 1))
    result = reduce_exposed_face(system, (_q(1), _q(0)))
    _identities(result)
    assert result.reduced.order == 0
    assert result.reduced.rhs == (_q(0), _q(1))
    assert (result.embedding.row_count, result.embedding.column_count) == (2, 0)
    assert (
        SemidefiniteFaceReduction.model_validate_json(result.model_dump_json())
        == result
    )


def test_serialized_reduced_system_supports_another_supplied_step() -> None:
    system = _system(
        (
            _matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            _matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        ),
        (0, 0),
    )
    first = reduce_exposed_face(system, (_q(1), _q(0)))
    reduced = RationalSemidefiniteSystem.model_validate_json(
        first.reduced.model_dump_json()
    )
    second = reduce_exposed_face(reduced, (_q(0), _q(1)))
    _identities(second)
    assert _sympy(first.embedding) * _sympy(second.embedding) == Matrix([[0], [0], [1]])


@pytest.mark.parametrize(
    "matrix,rhs,y",
    [
        ([[1, 0], [0, -1]], 0, 1),
        ([[0, 1], [1, 0]], 0, 1),
        ([[0, 0], [0, 0]], 0, 1),
        ([[1, 0], [0, 1]], 1, 1),
        ([[1, 0], [0, 1]], 0, -1),
    ],
)
def test_false_exposing_relations_are_rejected(
    matrix: list[list[int]], rhs: int, y: int
) -> None:
    with pytest.raises(OperationDomainValidationError, match=r"expos|positive|nonzero"):
        reduce_exposed_face(_system((_matrix(matrix),), (rhs,)), (_q(y),))


def test_multipliers_may_have_negative_entries_when_the_relation_is_valid() -> None:
    system = _system((_matrix([[2, 0], [0, 1]]), _matrix([[1, 0], [0, 1]])), (3, 3))
    result = reduce_exposed_face(system, (_q(1), _q(-1)))
    _identities(result)
    assert result.reduced.order == 1


def test_rational_inputs_and_nonunit_pivot() -> None:
    system = _system(
        (
            _matrix(
                [[Fraction(2, 3), Fraction(1, 3)], [Fraction(1, 3), Fraction(1, 6)]]
            ),
        ),
        (0,),
    )
    result = reduce_exposed_face(system, (_q(Fraction(3, 7)),))
    _identities(result)
    assert _sympy(result.embedding) == Matrix([[Rational(-1, 2)], [1]])


def test_dense_rank_one_exposure_at_useful_order() -> None:
    n = 32
    system = _system((_matrix([[1 for _ in range(n)] for _ in range(n)]),), (0,))
    result = reduce_exposed_face(system, (_q(1),))
    assert result.reduced.order == 31
    v = _sympy(result.embedding)
    assert v.rank() == 31
    assert _sympy(result.exposing_matrix) * v == zeros(n, 31)


def test_excessive_height_and_dense_output_are_rejected() -> None:
    # The exposing entry itself would have 40,001 digits, beyond the scalar
    # codomain. Rejection is necessary, rather than an inflated minor bound.
    large = 10**20000 + 1
    huge = _matrix([[large]])
    with pytest.raises(OperationResourceAdmissionError):
        reduce_exposed_face(_system((huge,), (0,)), (_q(large),))
    n = 164
    dense = _matrix([[int(i == j) for j in range(n)] for i in range(n)])
    with pytest.raises(OperationResourceAdmissionError):
        reduce_exposed_face(_system((dense,), (0,)), (_q(1),))


def test_large_scalar_exposure_needs_no_elimination_growth() -> None:
    matrix = _matrix([[Fraction(1, 10**20000 + 1)]])
    system = _system((matrix,), (0,))
    result = reduce_exposed_face(system, (_q(1),))
    assert result.exposing_matrix == matrix
    assert result.embedding.row_count == 1
    assert result.embedding.column_count == result.reduced.order == 0
    assert (
        SemidefiniteFaceReduction.model_validate_json(result.model_dump_json())
        == result
    )


def test_diagonal_exposure_has_coordinate_kernel_and_independent_denominators() -> None:
    n = 128
    diagonal = [Fraction(1, 2**128 + i + 1) if i % 2 else Fraction() for i in range(n)]
    matrix = _matrix(
        [[diagonal[i] if i == j else 0 for j in range(n)] for i in range(n)]
    )
    result = reduce_exposed_face(_system((matrix,), (0,)), (_q(1),))
    assert result.exposing_matrix == matrix
    assert result.reduced.order == n // 2
    assert all(
        not q.num for a in result.reduced.matrices for row in a.entries for q in row
    )
    for i, row in enumerate(result.embedding.entries):
        assert [q.as_fraction() for q in row] == [
            Fraction(int(i == 2 * j)) for j in range(n // 2)
        ]


def test_shape_validation_and_zero_cone_has_no_proper_exposure() -> None:
    with pytest.raises(ValidationError, match="symmetric"):
        _system((_matrix([[0, 1], [0, 0]]),), (0,))
    with pytest.raises(OperationDomainValidationError):
        reduce_exposed_face(
            RationalSemidefiniteSystem(order=0, matrices=(), rhs=()), ()
        )
    with pytest.raises(OperationDomainValidationError, match="index"):
        reduce_exposed_face(_system((_matrix([[1]]),), (0,)), ())


def test_raw_request_preflight_rejects_over_budget_cells() -> None:
    with pytest.raises(ValidationError, match="dense cell envelope"):
        SemidefiniteFaceReductionRequest.model_validate(
            {
                "system": {
                    "order": 128,
                    "matrices": [{}] * 128,
                    "rhs": [{"num": "0", "den": "1"}] * 128,
                },
                "multipliers": [{"num": "1", "den": "1"}] * 128,
            }
        )


def test_raw_request_preflight_counts_actual_nested_cells() -> None:
    row = [{"num": "0", "den": "1"}] * 128
    with pytest.raises(ValidationError, match="dense cell envelope"):
        SemidefiniteFaceReductionRequest.model_validate(
            {
                "system": {
                    "order": 1,
                    "matrices": [{"entries": [row] * 128}] * 9,
                    "rhs": [{"num": "0", "den": "1"}] * 9,
                },
                "multipliers": [{"num": "1", "den": "1"}] * 9,
            }
        )


def test_python_mode_generators_reduce_the_issue_fixture_face() -> None:
    # Nested generators are the motivating Python-mode payload: preflight must
    # install them before Pydantic sees an empty matrices or entries sequence.
    request = SemidefiniteFaceReductionRequest.model_validate(
        {
            "system": {
                "order": 2,
                "matrices": (
                    matrix
                    for matrix in (
                        {"entries": (row for row in ((_q(1), _q(0)), (_q(0), _q(0))))},
                        {"entries": ((_q(1), _q(0)), (_q(0), _q(1)))},
                    )
                ),
                "rhs": (_q(0), _q(1)),
            },
            "multipliers": (_q(1), _q(0)),
        }
    )
    result = reduce_exposed_face(request.system, request.multipliers)
    _identities(result)
    assert _sympy(result.embedding) == Matrix([[0], [1]])
    assert [_sympy(a) for a in result.reduced.matrices] == [
        Matrix([[0]]),
        Matrix([[1]]),
    ]


def test_raw_request_preflight_rejects_over_budget_generated_cells() -> None:
    row = [{"num": "0", "den": "1"}] * 128
    with pytest.raises(ValidationError, match="dense cell envelope"):
        SemidefiniteFaceReductionRequest.model_validate(
            {
                "system": {
                    "order": 1,
                    "matrices": ({"entries": [row] * 128} for _ in range(9)),
                    "rhs": [{"num": "0", "den": "1"}] * 9,
                },
                "multipliers": [{"num": "1", "den": "1"}] * 9,
            }
        )


def test_inactive_constraint_denominators_are_charged_during_compression() -> None:
    huge = Fraction(1, 10**20_001)
    system = _system(
        (_matrix([[1, 1], [1, 1]]), _matrix([[huge, 0], [0, 0]])),
        (0, 0),
    )
    with pytest.raises(OperationResourceAdmissionError, match="canonical rational"):
        reduce_exposed_face(system, (_q(1), _q(0)))


def test_inactive_rhs_heights_count_toward_output() -> None:
    huge = Fraction(10**32_000)
    matrices = (
        _matrix([[1, 1], [1, 1]]),
        *(_matrix([[0, 0], [0, 0]]) for _ in range(200)),
    )
    rhs = (0, *(huge for _ in range(200)))
    system = RationalSemidefiniteSystem(
        order=2, matrices=matrices, rhs=tuple(_q(value) for value in rhs)
    )
    with pytest.raises(OperationResourceAdmissionError, match="output digit"):
        reduce_exposed_face(system, (_q(1), *(_q(0) for _ in range(200))))


def test_coprime_constraint_denominators_are_bounded_in_compression() -> None:
    primes = [10**12_000 + 2 * i + 1 for i in range(3)]
    dense = _matrix(
        [
            [Fraction(1, primes[0]), Fraction(1, primes[1])],
            [Fraction(1, primes[1]), Fraction(1, primes[2])],
        ]
    )
    system = _system((_matrix([[0, 1], [1, 0]]), dense), (0, 0))
    with pytest.raises(OperationResourceAdmissionError, match="canonical rational"):
        reduce_exposed_face(system, (_q(1), _q(1)))
    with pytest.raises(OperationResourceAdmissionError, match="canonical rational"):
        reduce_exposed_face(system, (_q(1), _q(0)))
