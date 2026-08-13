"""Lean-provider fixtures that require the complete application runtime."""

from __future__ import annotations

from tests.support.complete_runtime_fixtures import (
    attached_complete_runtime,
    authorized_complete_runtime,
    authorized_portfolio_template,
    complete_portfolio_template,
)

__all__ = (
    "attached_complete_runtime",
    "authorized_complete_runtime",
    "authorized_portfolio_template",
    "complete_portfolio_template",
)
