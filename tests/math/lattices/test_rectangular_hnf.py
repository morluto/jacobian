"""Row-HNF reconstruction and unimodularity beyond the LLL envelope."""

from fractions import Fraction

import pytest

from jacobian.math.lattices._hnf import HERMITE_NORMAL_FORM_OPERATION
from jacobian.math.lattices._hnf_bounds import admit_hermite_normal_form


def _determinant(entries: tuple[tuple[int, ...], ...]) -> Fraction:
    """Independent rational elimination, without the HNF backend."""
    a = [[Fraction(x) for x in row] for row in entries]
    det = Fraction(1)
    for j in range(len(a)):
        pivot = next((i for i in range(j, len(a)) if a[i][j]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != j:
            a[j], a[pivot] = a[pivot], a[j]
            det = -det
        p = a[j][j]
        det *= p
        for i in range(j + 1, len(a)):
            q = a[i][j] / p
            for k in range(j + 1, len(a)):
                a[i][k] -= q * a[j][k]
    return det


@pytest.mark.parametrize(
    "shape", [(61, 25), (25, 61), (33, 33), (64, 64), (128, 25), (128, 128)]
)
def test_rectangular_coefficient_lattice(shape: tuple[int, int]) -> None:
    rows, cols = shape
    entries = [
        [((i + 1) * (j + 3) + j * j) % 101 - 50 for j in range(cols)]
        for i in range(rows)
    ]
    tool = HERMITE_NORMAL_FORM_OPERATION
    request = tool.request_type.model_validate(
        {"matrix": {"entries": [list(row) for row in entries]}}
    )
    result = tool.run(request)
    h, u = result.normal_form.entries, result.transformation.entries
    assert result.source_matrix == request.matrix
    assert len(h) == rows and len(h[0]) == cols
    assert len(u) == rows and len(u[0]) == rows
    assert all(
        sum(int(u[i][k]) * entries[k][j] for k in range(rows)) == int(h[i][j])
        for i in range(rows)
        for j in range(cols)
    )
    assert abs(_determinant(u)) == 1
    # Positive increasing pivots; reduced entries above every pivot; zero rows last.
    previous = -1
    zero_seen = False
    for i, row in enumerate(h):
        pivot = next((j for j, x in enumerate(row) if int(x)), None)
        if pivot is None:
            zero_seen = True
            continue
        assert not zero_seen and pivot > previous
        assert int(row[pivot]) > 0
        assert all(0 <= int(h[k][pivot]) < int(row[pivot]) for k in range(i))
        previous = pivot
    assert type(result).model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "entries",
    [
        [[0] for _ in range(128)],
        [[i % 7, 2 * (i % 7)] for i in range(61)],
        [[0, 0, 0]],
        [[-7]],
    ],
)
def test_degenerate_inputs_and_native_parity(entries: list[list[int]]) -> None:
    from jacobian.math.lattices.operations import hermite_normal_form

    tool = HERMITE_NORMAL_FORM_OPERATION
    result = tool.run(
        tool.request_type.model_validate({"matrix": {"entries": entries}})
    )
    h, u = hermite_normal_form(entries)
    assert result.normal_form.entries == tuple(
        tuple(int(h[i, j]) for j in range(h.ncols())) for i in range(h.nrows())
    )
    assert abs(_determinant(result.transformation.entries)) == 1
    assert all(
        sum(int(u[i, k]) * entries[k][j] for k in range(len(entries))) == int(h[i, j])
        for i in range(len(entries))
        for j in range(len(entries[0]))
    )


@pytest.mark.parametrize(
    ("rows", "columns", "value", "message"),
    [
        (64, 64, 10**255, "intermediate height"),
        (128, 128, 10**255, "minor height"),
    ],
)
def test_hnf_specific_excessive_envelopes(
    rows: int,
    columns: int,
    value: int,
    message: str,
) -> None:
    from jacobian.catalog.models import OperationDomainValidationError
    from jacobian.math.lattices.operations import hermite_normal_form

    entries = [
        [value + ((i + 1) * (j + 3) + j * j) % 101 for j in range(columns)]
        for i in range(rows)
    ]
    tool = HERMITE_NORMAL_FORM_OPERATION
    request = tool.request_type.model_validate(
        {"matrix": {"entries": [list(row) for row in entries]}}
    )
    with pytest.raises(OperationDomainValidationError, match=message):
        tool.run(request)
    with pytest.raises(OperationDomainValidationError, match=message):
        hermite_normal_form(entries)


def test_hnf_admission_bounds_both_nonmodular_transformation_passes() -> None:
    admission = admit_hermite_normal_form([[1, 0], [0, 1]])

    # Each W column can receive one extended-GCD update and one above-pivot
    # reduction. Both height-growth passes belong in the preflight bound.
    assert admission.minor_bits == 2
    assert admission.intermediate_bits == 20


@pytest.mark.parametrize("order", [2, 5, 9])
def test_determinant_modulus_is_not_a_cancelled_rref_denominator(order: int) -> None:
    # The rational RREF has denominator 2, but the square pivot determinant
    # is 2**order. Passing 2 to modular HNF loses lattice index information.
    entries = [[2 * int(i == j) for j in range(order)] for i in range(order)]
    tool = HERMITE_NORMAL_FORM_OPERATION
    result = tool.run(
        tool.request_type.model_validate(
            {"matrix": {"entries": [list(row) for row in entries]}}
        )
    )
    assert result.normal_form.entries == tuple(tuple(x for x in row) for row in entries)
    assert result.transformation.entries == tuple(
        tuple(int(i == j) for j in range(order)) for i in range(order)
    )


def test_small_normal_form_retains_large_exact_transformation() -> None:
    order, scale = 8, 10**50
    # A=I+scale*N for the nilpotent upper shift. Its inverse has alternating
    # powers of scale, even though its row HNF is the identity.
    entries = [
        [int(i == j) + scale * int(j == i + 1) for j in range(order)]
        for i in range(order)
    ]
    tool = HERMITE_NORMAL_FORM_OPERATION
    result = tool.run(
        tool.request_type.model_validate(
            {"matrix": {"entries": [list(row) for row in entries]}}
        )
    )
    assert result.normal_form.entries == tuple(
        tuple(int(i == j) for j in range(order)) for i in range(order)
    )
    assert result.transformation.entries == tuple(
        tuple((-scale) ** (j - i) if j >= i else 0 for j in range(order))
        for i in range(order)
    )


def test_seeded_rectangles_match_independent_reconstruction_and_flint_hnf() -> None:
    from random import Random

    from flint import fmpz_mat

    from jacobian.math.lattices.operations import hermite_normal_form

    rng = Random(3225)
    for rows in range(1, 8):
        for columns in range(1, 8):
            entries = [
                [rng.randrange(-9, 10) for _ in range(columns)] for _ in range(rows)
            ]
            # Some zero leading columns force nontrivial pivot selection.
            if columns > 2 and rows % 2:
                for row in entries:
                    row[0] = 0
            h, u = hermite_normal_form(entries)
            assert h == fmpz_mat(entries).hnf()
            assert (
                abs(
                    _determinant(
                        tuple(
                            tuple(int(u[i, j]) for j in range(rows))
                            for i in range(rows)
                        )
                    )
                )
                == 1
            )
            assert all(
                sum(int(u[i, k]) * entries[k][j] for k in range(rows)) == int(h[i, j])
                for i in range(rows)
                for j in range(columns)
            )
