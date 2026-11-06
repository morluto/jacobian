"""Exact JSON integer spellings through the singular Smith scorer."""

from pathlib import Path

import pytest

from ._integer_wire import check_equivalent_integer_tokens


@pytest.mark.parametrize("suffix", [".0", "e0"])
def test_equivalent_integer_tokens(tmp_path: Path, suffix: str) -> None:
    check_equivalent_integer_tokens(tmp_path, "smith-rank-deficient", suffix)
