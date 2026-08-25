import pytest

from jacobian.math.arithmetic._integer_predicates import is_square_free


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, True),
        (2, True),
        (6, True),
        (12, False),
        (49, False),
        (999_983, True),
    ],
)
def test_square_free_integer_predicate(value: int, expected: bool) -> None:
    assert is_square_free(value) is expected
