"""Small test-only assertions for owner-local work-accounting evidence."""

from __future__ import annotations

from collections.abc import Mapping


def assert_charged_work_parity(
    *,
    charged: Mapping[str, int],
    executed: Mapping[str, int],
) -> None:
    """Require every observed primitive to fit its admission charge.

    Owner tests supply quantities in the same units as their admission model.
    The assertion deliberately has no production counterpart: a kernel owns
    its actual work measurement, while tests prove that each observed unit is
    charged before execution.
    """

    assert all(amount >= 0 for amount in charged.values()), (
        "charged work must be nonnegative"
    )
    assert all(amount >= 0 for amount in executed.values()), (
        "executed work must be nonnegative"
    )
    uncharged = set(executed).difference(charged)
    assert not uncharged, f"executed work has no admission charge: {sorted(uncharged)}"
    for primitive, amount in executed.items():
        assert amount <= charged[primitive], (
            f"executed {primitive} work ({amount}) exceeds its admission charge "
            f"({charged[primitive]})"
        )
