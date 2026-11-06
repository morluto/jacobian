"""Exact JSON integer spellings through the rectangular Smith scorer."""

from pathlib import Path

import pytest

from ._integer_wire import check_equivalent_integer_tokens, check_invalid_integer_claim


@pytest.mark.parametrize("suffix", [".0", "e0"])
def test_equivalent_integer_tokens(tmp_path: Path, suffix: str) -> None:
    check_equivalent_integer_tokens(tmp_path, "smith-rectangular", suffix)


@pytest.mark.parametrize(
    "rank",
    ["true", '"2"', "2.0000000000000001", "2.5", "1e9999999", "1e-9999999", "1.0"],
)
def test_invalid_or_wrong_claim(tmp_path: Path, rank: str) -> None:
    check_invalid_integer_claim(tmp_path, rank)
