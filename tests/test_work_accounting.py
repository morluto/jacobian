"""Tests for the owner-local charged-work parity assertion."""

from __future__ import annotations

import pytest

from tests.fixtures.accounting import assert_charged_work_parity


def test_parity_accepts_instrumented_work_within_its_charge() -> None:
    assert_charged_work_parity(
        charged={"sort": 3, "scan": 7},
        executed={"sort": 3, "scan": 6},
    )


def test_parity_rejects_missing_and_overcharged_work() -> None:
    with pytest.raises(AssertionError, match="no admission charge"):
        assert_charged_work_parity(charged={}, executed={"sort": 1})
    with pytest.raises(AssertionError, match="exceeds its admission charge"):
        assert_charged_work_parity(charged={"scan": 1}, executed={"scan": 2})
