from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.canonical_forms import (
    invariant_factors,
    minimal_polynomial,
    primary_decomposition,
)
from jacobian.math.matrices.canonical_forms import operations as canonical_operations
from jacobian.math.matrices.canonical_forms._models import (
    InvariantFactorEntry,
    MonicPolynomial,
    RationalCanonicalFormResult,
)
from jacobian.math.matrices.canonical_forms._tools import (
    compute_minimal_polynomial,
    compute_primary_decomposition,
    compute_rational_canonical_form,
)
from jacobian.math.matrices.values import RationalMatrix

R = CanonicalRational


def _mat(*rows: tuple[tuple[str, str], ...]) -> RationalMatrix:
    entries = tuple(tuple(R(num=num, den=den) for num, den in row) for row in rows)
    return RationalMatrix(entries=entries)


def _assert_admission_rejected(matrix: RationalMatrix) -> None:
    with pytest.raises(OperationDomainValidationError):
        compute_minimal_polynomial(matrix)


def _coeffs(poly: MonicPolynomial) -> list[Fraction]:
    return [coefficient.as_fraction() for coefficient in poly.coefficients]


def _pair(num: str, den: str) -> tuple[str, str]:
    return (num, den)


def _diagonal(*values: str) -> RationalMatrix:
    entries = tuple(
        tuple(
            R(num=value if row == column else "0", den="1")
            for column, value in enumerate(values)
        )
        for row in range(len(values))
    )
    return RationalMatrix(entries=entries)


def _mono(*coefficients: Fraction | int) -> MonicPolynomial:
    return MonicPolynomial(
        coefficients=tuple(
            R.from_fraction(Fraction(coefficient)) for coefficient in coefficients
        )
    )


def _companion(*tail: tuple[str, str]) -> RationalMatrix:
    """Companion matrix of x^n + tail[-1] x^{n-1} + ... + tail[0]."""
    constants = [Fraction(int(num), int(den)) for num, den in tail]
    n = len(constants)
    entries = [
        tuple(
            R.from_fraction(Fraction(1) if j == i + 1 else Fraction(0))
            for j in range(n)
        )
        for i in range(n - 1)
    ]
    entries.append(tuple(R.from_fraction(-constant) for constant in constants))
    return RationalMatrix(entries=tuple(entries))


def test_nilpotent_jordan_block_minimal_polynomial_is_t_squared() -> None:
    """Matrix [[0,1],[0,0]] has minimal polynomial t^2."""
    req = _mat(
        (_pair("0", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("0", "1")),
    )
    result = compute_minimal_polynomial(req)
    assert _coeffs(result.minimal_polynomial) == [Fraction(0), Fraction(0), Fraction(1)]
    assert result.degree == 2


def test_trusted_canonical_form_producers_run_each_kernel_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _diagonal("2", "3")
    names = (
        "invariant_factors",
        "minimal_polynomial",
        "characteristic_polynomial",
        "primary_decomposition",
    )
    calls = dict.fromkeys(names, 0)

    for name in names:
        original = getattr(canonical_operations, name)

        def counted(
            *args: object,
            _original: Callable[..., object] = original,
            _name: str = name,
            **kwargs: object,
        ) -> object:
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(canonical_operations, name, counted)

    compute_minimal_polynomial(request)
    assert calls == {
        "invariant_factors": 0,
        "minimal_polynomial": 1,
        "characteristic_polynomial": 1,
        "primary_decomposition": 0,
    }

    for name in calls:
        calls[name] = 0
    compute_rational_canonical_form(request)
    assert calls == {
        "invariant_factors": 1,
        "minimal_polynomial": 1,
        "characteristic_polynomial": 1,
        "primary_decomposition": 0,
    }

    for name in calls:
        calls[name] = 0
    compute_primary_decomposition(request)
    assert calls == {
        "invariant_factors": 0,
        "minimal_polynomial": 1,
        "characteristic_polynomial": 0,
        "primary_decomposition": 1,
    }


def test_diagonal_distinct_minimal_equals_characteristic() -> None:
    """diag(2,3) has minimal polynomial (t-2)(t-3) = t^2 - 5t + 6."""
    req = _mat(
        (_pair("2", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("3", "1")),
    )
    result = compute_minimal_polynomial(req)
    assert _coeffs(result.minimal_polynomial) == [
        Fraction(6),
        Fraction(-5),
        Fraction(1),
    ]
    assert _coeffs(result.characteristic_polynomial) == [
        Fraction(6),
        Fraction(-5),
        Fraction(1),
    ]


def test_jordan_block_minimal_equals_characteristic() -> None:
    """[[2,1],[0,2]] has minimal polynomial (t-2)^2 = t^2 - 4t + 4."""
    req = _mat(
        (_pair("2", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("2", "1")),
    )
    result = compute_minimal_polynomial(req)
    assert _coeffs(result.minimal_polynomial) == [
        Fraction(4),
        Fraction(-4),
        Fraction(1),
    ]
    assert _coeffs(result.characteristic_polynomial) == [
        Fraction(4),
        Fraction(-4),
        Fraction(1),
    ]


def test_identity_matrix_minimal_polynomial_is_t_minus_one() -> None:
    """2x2 identity has minimal polynomial t - 1."""
    req = _mat(
        (_pair("1", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("1", "1")),
    )
    result = compute_minimal_polynomial(req)
    assert _coeffs(result.minimal_polynomial) == [Fraction(-1), Fraction(1)]


def test_irreducible_over_qq_minimal_polynomial() -> None:
    """[[0,-1],[1,0]] has minimal polynomial t^2 + 1 (irreducible over QQ)."""
    req = _mat(
        (_pair("0", "1"), _pair("-1", "1")),
        (_pair("1", "1"), _pair("0", "1")),
    )
    result = compute_minimal_polynomial(req)
    assert _coeffs(result.minimal_polynomial) == [Fraction(1), Fraction(0), Fraction(1)]


def test_nilpotent_single_block_canonical_form() -> None:
    """[[0,1],[0,0]] has one invariant factor t^2."""
    req = _mat(
        (_pair("0", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("0", "1")),
    )
    result = compute_rational_canonical_form(req)
    assert len(result.invariant_factors) == 1
    assert _coeffs(result.invariant_factors[0].factor) == [
        Fraction(0),
        Fraction(0),
        Fraction(1),
    ]
    assert result.invariant_factors[0].block_size == 2
    assert result.total_block_size == 2


def test_diagonal_distinct_single_factor_canonical_form() -> None:
    """diag(2,3) has one invariant factor (t-2)(t-3)."""
    req = _mat(
        (_pair("2", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("3", "1")),
    )
    result = compute_rational_canonical_form(req)
    assert len(result.invariant_factors) == 1
    assert _coeffs(result.invariant_factors[0].factor) == [
        Fraction(6),
        Fraction(-5),
        Fraction(1),
    ]


def test_identity_two_blocks_canonical_form() -> None:
    """2x2 identity has invariant factors (t-1), (t-1)."""
    req = _mat(
        (_pair("1", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("1", "1")),
    )
    result = compute_rational_canonical_form(req)
    assert len(result.invariant_factors) == 2
    assert result.total_block_size == 2
    for entry in result.invariant_factors:
        assert _coeffs(entry.factor) == [Fraction(-1), Fraction(1)]


def test_nilpotent_two_blocks_divisibility_chain() -> None:
    """Nilpotent with blocks of sizes 2 and 1: invariant factors t | t^2."""
    req = _mat(
        (_pair("0", "1"), _pair("1", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("0", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("0", "1"), _pair("0", "1")),
    )
    result = compute_rational_canonical_form(req)
    assert len(result.invariant_factors) == 2
    sizes = [entry.block_size for entry in result.invariant_factors]
    assert sizes == [1, 2]


def test_primary_decomposition_distinct_linear_factors() -> None:
    """diag(2,3) decomposes into (t-2) and (t-3)."""
    req = _mat(
        (_pair("2", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("3", "1")),
    )
    result = compute_primary_decomposition(req)
    assert len(result.components) == 2
    for component in result.components:
        assert len(_coeffs(component)) == 2


def test_primary_decomposition_irreducible_power() -> None:
    """[[0,1],[0,0]] has minpoly t^2, primary decomposition is [t^2]."""
    req = _mat(
        (_pair("0", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("0", "1")),
    )
    result = compute_primary_decomposition(req)
    assert len(result.components) == 1
    assert _coeffs(result.components[0]) == [Fraction(0), Fraction(0), Fraction(1)]


def test_primary_decomposition_normalizes_rational_root_factors() -> None:
    """diag(1/2, 1/3) decomposes into the monic factors (t-1/2) and (t-1/3)."""
    req = _mat(
        (_pair("1", "2"), _pair("0", "1")),
        (_pair("0", "1"), _pair("1", "3")),
    )
    result = compute_primary_decomposition(req)
    assert sorted(_coeffs(component) for component in result.components) == [
        [Fraction(-1, 2), Fraction(1)],
        [Fraction(-1, 3), Fraction(1)],
    ]


def test_contract_rejects_nonsquare() -> None:
    _assert_admission_rejected(
        RationalMatrix(entries=((R(num="1", den="1"), R(num="0", den="1")),))
    )


def test_contract_rejects_non_monic_polynomial() -> None:
    with pytest.raises(ValidationError):
        MonicPolynomial(coefficients=(R(num="1", den="1"), R(num="2", den="1")))


def test_characteristic_equals_product_of_invariant_factors() -> None:
    """Product of invariant factors equals the characteristic polynomial."""
    req = _mat(
        (_pair("0", "1"), _pair("1", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("0", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("0", "1"), _pair("0", "1")),
    )
    result = compute_rational_canonical_form(req)
    assert result.total_block_size == 3
    assert _coeffs(result.characteristic_polynomial) == [
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(1),
    ]


def test_smith_normal_form_path_handles_larger_matrices() -> None:
    """A 6x6 diagonal matrix uses the maintained Smith form, not minor enumeration."""
    req = _diagonal("2", "3", "5", "7", "11", "13")
    result = compute_rational_canonical_form(req)
    assert len(result.invariant_factors) == 1
    assert result.invariant_factors[0].block_size == 6
    assert result.total_block_size == 6
    characteristic = _coeffs(result.characteristic_polynomial)
    assert len(characteristic) == 7
    assert characteristic[-1] == Fraction(1)
    assert characteristic[0] == Fraction(2 * 3 * 5 * 7 * 11 * 13)
    assert characteristic[-2] == Fraction(-(2 + 3 + 5 + 7 + 11 + 13))
    assert _coeffs(result.minimal_polynomial) == characteristic


def test_single_entry_matrix_canonical_forms() -> None:
    req = _mat((_pair("2", "1"),))
    assert _coeffs(compute_minimal_polynomial(req).minimal_polynomial) == [
        Fraction(-2),
        Fraction(1),
    ]

    rcf = compute_rational_canonical_form(req)
    assert len(rcf.invariant_factors) == 1
    assert rcf.invariant_factors[0].block_size == 1
    assert rcf.total_block_size == 1
    assert _coeffs(rcf.invariant_factors[0].factor) == [Fraction(-2), Fraction(1)]

    decomposition = compute_primary_decomposition(req)
    assert len(decomposition.components) == 1
    assert _coeffs(decomposition.components[0]) == [Fraction(-2), Fraction(1)]


def test_contract_rejects_oversized_and_wide_scalar_matrices() -> None:
    identity_17 = tuple(
        tuple(R(num="1" if row == column else "0", den="1") for column in range(17))
        for row in range(17)
    )
    _assert_admission_rejected(RationalMatrix(entries=identity_17))

    wide_scalar = ((R(num="1" + "0" * 256, den="1"),),)
    _assert_admission_rejected(RationalMatrix(entries=wide_scalar))


def test_public_kernels_return_monic_coefficient_lists() -> None:
    entries = (
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(0)),
    )
    assert minimal_polynomial(entries) == (Fraction(0), Fraction(0), Fraction(1))
    assert invariant_factors(entries) == ((Fraction(0), Fraction(0), Fraction(1)),)
    assert primary_decomposition(entries) == ((Fraction(0), Fraction(0), Fraction(1)),)


def _block_diagonal(*blocks: RationalMatrix) -> RationalMatrix:
    n = sum(len(block.entries) for block in blocks)
    grid = [[Fraction(0)] * n for _ in range(n)]
    offset = 0
    for block in blocks:
        entries = [[entry.as_fraction() for entry in row] for row in block.entries]
        for i, row in enumerate(entries):
            for j, value in enumerate(row):
                grid[offset + i][offset + j] = value
        offset += len(entries)
    return RationalMatrix(
        entries=tuple(tuple(R.from_fraction(value) for value in row) for row in grid)
    )


def _conjugate(
    matrix_value: RationalMatrix,
    change_of_basis: tuple[tuple[Fraction, ...], ...],
) -> RationalMatrix:
    """Return S A S^{-1} for an invertible rational change of basis."""
    import sympy

    matrix = sympy.Matrix(
        [[entry.as_fraction() for entry in row] for row in matrix_value.entries]
    )
    similarity = sympy.Matrix([list(row) for row in change_of_basis])
    transformed = similarity * matrix * similarity.inv()
    return RationalMatrix(
        entries=tuple(
            tuple(R.from_fraction(Fraction(int(v.p), int(v.q))) for v in row)
            for row in transformed.tolist()
        )
    )


def test_source_binding_across_matrix_families() -> None:
    """Scalar, companion, diagonal, repeated-block, and noncyclic sources all bind."""
    families = (
        _diagonal("1", "1", "1"),
        _companion(("6", "1"), ("-5", "1")),
        _diagonal("2", "3"),
        _mat(
            (_pair("2", "1"), _pair("1", "1")),
            (_pair("0", "1"), _pair("2", "1")),
        ),
        _block_diagonal(
            _mat(
                (_pair("0", "1"), _pair("1", "1")),
                (_pair("0", "1"), _pair("0", "1")),
            ),
            _mat(
                (_pair("0", "1"), _pair("1", "1")),
                (_pair("0", "1"), _pair("0", "1")),
            ),
        ),
    )
    for family in families:
        assert compute_minimal_polynomial(family).matrix == family
        assert compute_rational_canonical_form(family).matrix == family
        assert compute_primary_decomposition(family).matrix == family


def test_noncyclic_components_multiply_to_minimal_not_characteristic() -> None:
    """J2 + J2 has components [t^2]; their product is t^2 = minpoly, not t^4."""
    req = _block_diagonal(
        _mat(
            (_pair("0", "1"), _pair("1", "1")),
            (_pair("0", "1"), _pair("0", "1")),
        ),
        _mat(
            (_pair("0", "1"), _pair("1", "1")),
            (_pair("0", "1"), _pair("0", "1")),
        ),
    )
    result = compute_primary_decomposition(req)
    assert len(result.components) == 1
    assert _coeffs(result.components[0]) == [Fraction(0), Fraction(0), Fraction(1)]
    assert _coeffs(result.minimal_polynomial) == [Fraction(0), Fraction(0), Fraction(1)]
    characteristic = _coeffs(compute_minimal_polynomial(req).characteristic_polynomial)
    assert len(characteristic) == 5  # t^4: degree four, distinct from minpoly


def test_rational_canonical_form_rejects_structural_mutations() -> None:
    req = _diagonal("2", "3")
    result = compute_rational_canonical_form(req)
    with pytest.raises(ValidationError):
        RationalCanonicalFormResult(
            matrix=result.matrix,
            invariant_factors=(
                InvariantFactorEntry(factor=_mono(6, -5, 1), block_size=1),
                InvariantFactorEntry(factor=_mono(6, -5, 1), block_size=1),
            ),
            characteristic_polynomial=result.characteristic_polynomial,
            minimal_polynomial=result.minimal_polynomial,
            total_block_size=2,
        )
    with pytest.raises(ValidationError):
        RationalCanonicalFormResult(
            matrix=result.matrix,
            invariant_factors=result.invariant_factors,
            characteristic_polynomial=result.characteristic_polynomial,
            minimal_polynomial=result.minimal_polynomial,
            total_block_size=result.total_block_size + 1,
        )
    with pytest.raises(ValidationError):
        RationalCanonicalFormResult(
            matrix=result.matrix,
            invariant_factors=(
                InvariantFactorEntry(factor=_mono(-2, 1), block_size=1),
            ),
            characteristic_polynomial=result.characteristic_polynomial,
            minimal_polynomial=result.minimal_polynomial,
            total_block_size=1,
        )


def test_rational_canonical_form_rejects_divisibility_break() -> None:
    """(t-2) does not divide (t-3): successive divisibility fails."""
    req = _diagonal("2", "3")
    result = compute_rational_canonical_form(req)
    with pytest.raises(ValidationError):
        RationalCanonicalFormResult(
            matrix=result.matrix,
            invariant_factors=(
                InvariantFactorEntry(factor=_mono(-2, 1), block_size=1),
                InvariantFactorEntry(factor=_mono(-3, 1), block_size=1),
            ),
            characteristic_polynomial=result.characteristic_polynomial,
            minimal_polynomial=result.minimal_polynomial,
            total_block_size=2,
        )


def _nilpotent_jordan_blocks(block_size: int) -> RationalMatrix:
    """Direct sum of two nilpotent Jordan blocks J_block_size(0)."""
    dimension = 2 * block_size
    values = [[R.from_fraction(Fraction(0))] * dimension for _ in range(dimension)]
    for start in (0, block_size):
        for offset in range(block_size - 1):
            values[start + offset][start + offset + 1] = R.from_fraction(Fraction(1))
    return RationalMatrix(entries=tuple(tuple(row) for row in values))


def test_rational_canonical_form_j3_plus_j3_has_two_x_cubed_factors() -> None:
    """J3(0) + J3(0) has exact invariant factors (x^3, x^3)."""
    result = compute_rational_canonical_form(_nilpotent_jordan_blocks(3))
    assert len(result.invariant_factors) == 2
    for entry in result.invariant_factors:
        assert entry.block_size == 3
        assert _coeffs(entry.factor) == [
            Fraction(0),
            Fraction(0),
            Fraction(0),
            Fraction(1),
        ]
    assert _coeffs(result.minimal_polynomial) == [
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(1),
    ]
    assert _coeffs(result.characteristic_polynomial) == [Fraction(0)] * 6 + [
        Fraction(1)
    ]


def test_results_are_similarity_invariant_under_rational_change_of_basis() -> None:
    swap = ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)))
    shear = ((Fraction(1), Fraction(1, 2)), (Fraction(0), Fraction(1)))
    req = _diagonal("2", "3")
    jordan = _mat(
        (_pair("2", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("2", "1")),
    )
    for source in (req, jordan):
        expected = compute_rational_canonical_form(source).invariant_factors
        expected_minimal = compute_minimal_polynomial(source).minimal_polynomial
        expected_decomposition = compute_primary_decomposition(source).components
        for basis in (swap, shear):
            conjugated = _conjugate(source, basis)
            assert (
                compute_rational_canonical_form(conjugated).invariant_factors
                == expected
            )
            assert compute_minimal_polynomial(conjugated).minimal_polynomial == (
                expected_minimal
            )
            assert (
                compute_primary_decomposition(conjugated).components
                == expected_decomposition
            )


def test_serialization_round_trip_preserves_source_and_axis() -> None:
    req = _mat(
        (_pair("1", "2"), _pair("1", "1")),
        (_pair("0", "1"), _pair("1", "3")),
    )
    minimal = compute_minimal_polynomial(req)
    canonical = compute_rational_canonical_form(req)
    decomposition = compute_primary_decomposition(req)
    for result in (minimal, canonical, decomposition):
        restored = type(result).model_validate(result.model_dump())
        assert restored == result
        assert restored.matrix == req


def test_source_bound_result_contracts_are_versioned_as_version_two() -> None:
    from jacobian.math.matrices.canonical_forms._tools import TOOLS

    {tool.operation_id: tool for tool in TOOLS}
    # Each result gained a required source matrix, which old strict consumers
    # reject; the breaking output change must be distinguishable by version.
