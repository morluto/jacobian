from fractions import Fraction
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lie_algebras.matrix_span._models import (
    LieMatrixSpanRealization,
    LieMatrixSpanRequest,
)
from jacobian.math.lie_algebras.matrix_span.operations import (
    _admit,
    lie_algebra_from_matrix_span,
)
from jacobian.math.matrices.values import RationalMatrix


def _matrix(entries: tuple[tuple[int, ...], ...]) -> dict[str, object]:
    return {
        "domain": "QQ",
        "entries": [[{"num": value, "den": 1} for value in row] for row in entries],
    }


def _request(*matrices: tuple[tuple[int, ...], ...]) -> LieMatrixSpanRequest:
    return LieMatrixSpanRequest.model_validate(
        {"matrices": [_matrix(matrix) for matrix in matrices]}
    )


def _oracle_bracket(
    left: tuple[tuple[int, ...], ...], right: tuple[tuple[int, ...], ...]
) -> tuple[Fraction, ...]:
    """Independent exact matrix multiplication oracle used by these fixtures."""
    n = len(left)
    return tuple(
        sum(
            (
                Fraction(left[i][k] * right[k][j] - right[i][k] * left[k][j])
                for k in range(n)
            ),
            Fraction(),
        )
        for i in range(n)
        for j in range(n)
    )


E = ((0, 1), (0, 0))
F = ((0, 0), (1, 0))
H = ((1, 0), (0, -1))


def test_sl2_matrix_span_constructs_induced_bracket_and_roundtrips() -> None:
    request = _request(E, F, H)
    result = lie_algebra_from_matrix_span(request.matrices)
    assert result.algebra.basis == ("M0", "M1", "M2")
    assert [
        (item.i, item.j, item.k, item.coefficient.as_fraction())
        for item in result.algebra.structure_constants
    ] == [(0, 1, 2, Fraction(1)), (0, 2, 0, Fraction(-2)), (1, 2, 1, Fraction(2))]
    for item in result.algebra.structure_constants:
        oracle = _oracle_bracket((E, F, H)[item.i], (E, F, H)[item.j])
        expected = tuple(
            Fraction(value) for value in (E, F, H)[item.k][0] + (E, F, H)[item.k][1]
        )
        assert oracle == tuple(
            item.coefficient.as_fraction() * value for value in expected
        )
    assert LieMatrixSpanRealization.model_validate(result.model_dump()) == result


def test_shared_basis_denominator_is_cleared_before_growth_admission() -> None:
    matrices = tuple(
        RationalMatrix(
            entries=tuple(
                tuple(
                    CanonicalRational.from_fraction(Fraction(value, 2))
                    for value in row
                )
                for row in matrix
            )
        )
        for matrix in (E, F, H)
    )
    result = lie_algebra_from_matrix_span(matrices)

    assert [
        item.coefficient.as_fraction() for item in result.algebra.structure_constants
    ] == [Fraction(1, 2), Fraction(-1), Fraction(1)]


def test_standard_m2_matrix_units_fit_the_four_dimensional_boundary() -> None:
    units = (
        ((1, 0), (0, 0)),
        ((0, 1), (0, 0)),
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
    )
    result = lie_algebra_from_matrix_span(_request(*units).matrices)
    assert result.algebra.basis == ("M0", "M1", "M2", "M3")
    constants = {
        (item.i, item.j, item.k): item.coefficient.as_fraction()
        for item in result.algebra.structure_constants
    }
    for i, j in combinations(range(4), 2):
        bracket = _oracle_bracket(units[i], units[j])
        reconstructed = tuple(
            sum(
                (
                    constants.get((i, j, k), Fraction(0))
                    * units[k][row][column]
                    for k in range(4)
                ),
                Fraction(0),
            )
            for row in range(2)
            for column in range(2)
        )
        assert reconstructed == bracket


def test_nonclosed_or_dependent_span_is_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="not closed"):
        lie_algebra_from_matrix_span(_request(E, F).matrices)
    with pytest.raises(OperationDomainValidationError, match="independent"):
        lie_algebra_from_matrix_span(_request(E, E).matrices)


def test_one_dimensional_scalar_span_has_zero_bracket() -> None:
    identity = ((1, 0), (0, 1))
    result = lie_algebra_from_matrix_span(_request(identity).matrices)
    assert result.algebra.structure_constants == ()
    assert result.matrix_basis == _request(identity).matrices


def test_raw_scalar_height_is_rejected_before_matrix_canonicalization() -> None:
    payload = {"matrices": [_matrix(((10**64, 0), (0, 0)))]}
    with pytest.raises(ValidationError, match="limited to 64"):
        LieMatrixSpanRequest.model_validate(payload)


def test_canonical_rational_input_is_measured_by_its_components() -> None:
    large = CanonicalRational(num=10**54, den=1)
    matrix = RationalMatrix(
        entries=((large,),),
    )
    result = lie_algebra_from_matrix_span((matrix,))
    assert result.matrix_basis[0].entries[0][0] == large


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    ((1, 0), (2, 2), (True, 1), (0, -1)),
)
def test_native_execution_rejects_forged_noncanonical_rationals(
    numerator: object, denominator: object
) -> None:
    scalar = CanonicalRational.model_construct(num=numerator, den=denominator)
    matrix = RationalMatrix.model_construct(
        domain="QQ", row_count=1, column_count=1, entries=((scalar,),)
    )
    request = LieMatrixSpanRequest.model_construct(matrices=(matrix,))
    with pytest.raises(OperationDomainValidationError, match="reduced canonical"):
        lie_algebra_from_matrix_span(request.matrices)


def test_single_digit_denominators_include_multiplicative_carry() -> None:
    matrices = tuple(
        RationalMatrix(entries=((CanonicalRational(num=1, den=denominator),),))
        for denominator in (7, 8)
    )
    _, _, commutator_digits, _ = _admit(matrices)

    assert commutator_digits >= 2


def test_native_execution_rejects_forged_non_qq_matrix_domain() -> None:
    matrix = RationalMatrix.model_construct(
        domain="ZZ",
        row_count=1,
        column_count=1,
        entries=((CanonicalRational(num=1, den=1),),),
    )
    request = LieMatrixSpanRequest.model_construct(matrices=(matrix,))

    with pytest.raises(OperationDomainValidationError, match="QQ matrix domain"):
        lie_algebra_from_matrix_span(request.matrices)


def test_canonical_rational_scalar_ceiling_is_discoverable() -> None:
    from jacobian.math.lie_algebras.matrix_span._tools import TOOLS

    description = TOOLS[0].description
    assert "64 decimal digits" in description


def test_canonical_rational_component_width_is_measured_individually() -> None:
    value = CanonicalRational(num=10**54, den=1)
    matrix = RationalMatrix(entries=((value,),))

    result = lie_algebra_from_matrix_span((matrix,))

    assert result.matrix_basis[0].entries[0][0] == value


def test_commuting_rational_span_skips_irrelevant_structure_constant_bound() -> None:
    first_denominator = 1_000_000_000_039
    second_denominator = 1_000_000_000_061
    matrices = (
        RationalMatrix(
            entries=(
                (CanonicalRational(num=1, den=first_denominator), CanonicalRational(num=0, den=1)),
                (CanonicalRational(num=0, den=1), CanonicalRational(num=0, den=1)),
            )
        ),
        RationalMatrix(
            entries=(
                (CanonicalRational(num=0, den=1), CanonicalRational(num=0, den=1)),
                (CanonicalRational(num=0, den=1), CanonicalRational(num=1, den=second_denominator)),
            )
        ),
    )

    result = lie_algebra_from_matrix_span(matrices)

    assert result.matrix_basis == matrices
    assert result.algebra.structure_constants == ()


def test_sl2_basis_with_independent_large_denominators_is_admitted() -> None:
    from fractions import Fraction

    from jacobian.math.matrices.values import rational_matrix_from_fractions

    scales = (1_000_003, 1_000_033, 1_000_037)
    units = (
        ((0, 1), (0, 0)),
        ((0, 0), (1, 0)),
        ((1, 0), (0, -1)),
    )
    matrices = tuple(
        rational_matrix_from_fractions(
            tuple(tuple(Fraction(value, scale) for value in row) for row in matrix)
        )
        for matrix, scale in zip(units, scales, strict=True)
    )

    result = lie_algebra_from_matrix_span(matrices)

    assert len(result.algebra.structure_constants) == 3


def test_admitted_commutators_are_reused_during_construction(monkeypatch) -> None:
    import jacobian.math.lie_algebras.matrix_span.operations as operations

    matrices = _request(((0, 1), (0, 0)), ((0, 0), (1, 0))).matrices
    original = operations._commutator
    calls = 0

    def count(left, right):
        nonlocal calls
        calls += 1
        return original(left, right)

    monkeypatch.setattr(operations, "_commutator", count)
    with pytest.raises(OperationDomainValidationError, match="not closed"):
        lie_algebra_from_matrix_span(matrices)

    assert calls == 1
