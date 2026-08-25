"""Tests for Latin square operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.latin_squares._models import (
    LatinSquareRequest,
    OrthogonalityRequest,
    TransposeRequest,
)
from jacobian.math.combinatorics.latin_squares._operations import (
    compute_latin_square_check,
    compute_latin_square_transpose,
    compute_orthogonality,
)
from jacobian.math.combinatorics.latin_squares._tools import TOOLS

Z2 = {"order": 2, "cells": [[0, 1], [1, 0]]}


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "latin_square.check",
        "latin_square.orthogonality.check",
        "latin_square.transpose.compute",
    }


def test_latin_square_check_valid() -> None:
    request = LatinSquareRequest(square={"order": 2, "cells": ((0, 1), (1, 0))})
    result = compute_latin_square_check(request)
    assert result.is_latin is True


def test_latin_square_check_invalid() -> None:
    request = LatinSquareRequest(square={"order": 2, "cells": ((0, 0), (1, 1))})
    result = compute_latin_square_check(request)
    assert result.is_latin is False


def test_orthogonality_identical_not_orthogonal() -> None:
    request = OrthogonalityRequest(
        square_a=Z2,
        square_b=Z2,
    )
    result = compute_orthogonality(request)
    assert result.is_orthogonal is False


def test_orthogonality_orthogonal() -> None:
    request = OrthogonalityRequest(
        square_a={"order": 3, "cells": ((0, 1, 2), (1, 2, 0), (2, 0, 1))},
        square_b={"order": 3, "cells": ((0, 1, 2), (2, 0, 1), (1, 2, 0))},
    )
    result = compute_orthogonality(request)
    assert result.is_orthogonal is True
    assert result.pair_count == 9


def test_transpose() -> None:
    request = TransposeRequest(square={"order": 2, "cells": ((0, 1), (1, 0))})
    result = compute_latin_square_transpose(request)
    assert result.transposed == ((0, 1), (1, 0))


def test_orthogonality_rejects_non_latin_square() -> None:
    """Orthogonality must only be computed on actual Latin squares."""
    non_latin = {"order": 2, "cells": [[0, 0], [1, 1]]}
    with pytest.raises(ValidationError):
        OrthogonalityRequest(square_a=non_latin, square_b=Z2)
    with pytest.raises(ValidationError):
        OrthogonalityRequest(square_a=Z2, square_b=non_latin)


def test_transpose_rejects_non_latin_square() -> None:
    """Transpose must only be computed on actual Latin squares."""
    non_latin = {"order": 2, "cells": [[0, 0], [1, 1]]}
    with pytest.raises(ValidationError):
        TransposeRequest(square=non_latin)


def test_check_accepts_non_latin_square() -> None:
    """The check operation must accept non-Latin squares to test them."""
    request = LatinSquareRequest(square={"order": 2, "cells": ((0, 0), (1, 1))})
    result = compute_latin_square_check(request)
    assert result.is_latin is False
