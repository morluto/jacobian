"""Small test-only assertions for owner-local work accounting."""

from __future__ import annotations


def assert_executed_work_is_charged(*, charged: int, executed: int) -> None:
    """Fail when an owner executes more priced units than it admitted."""

    assert charged >= 0
    assert executed >= 0
    assert executed <= charged, (
        f"executed work ({executed}) exceeds the admitted charge ({charged})"
    )
