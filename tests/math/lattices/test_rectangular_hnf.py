"""Row-HNF reconstruction and unimodularity beyond the LLL envelope."""

from fractions import Fraction

import pytest

from jacobian.math.lattices._hnf import HERMITE_NORMAL_FORM_OPERATION


def _determinant(entries: tuple[tuple[str, ...], ...]) -> Fraction:
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


@pytest.mark.parametrize("shape", [(61, 25), (25, 61), (33, 33)])
def test_rectangular_coefficient_lattice(shape: tuple[int, int]) -> None:
    rows, cols = shape
    entries = [
        [((i + 1) * (j + 3) + j * j) % 101 - 50 for j in range(cols)]
        for i in range(rows)
    ]
    tool = HERMITE_NORMAL_FORM_OPERATION
    request = tool.request_type.model_validate(
        {"matrix": {"entries": [[str(x) for x in row] for row in entries]}}
    )
    result = tool.run(request)
    h, u = result.normal_form.entries, result.transformation.entries
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
        tool.request_type.model_validate(
            {"matrix": {"entries": [[str(x) for x in row] for row in entries]}}
        )
    )
    h, u = hermite_normal_form(entries)
    assert result.normal_form.entries == tuple(
        tuple(str(h[i, j]) for j in range(h.ncols())) for i in range(h.nrows())
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
        (64, 64, 10**255, "digit-work"),
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
        {"matrix": {"entries": [[str(x) for x in row] for row in entries]}}
    )
    with pytest.raises(OperationDomainValidationError, match=message):
        tool.run(request)
    with pytest.raises(OperationDomainValidationError, match=message):
        hermite_normal_form(entries)
