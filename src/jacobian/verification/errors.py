"""Operational errors raised while an authorized checker is running."""

from __future__ import annotations


class CheckerExecutionError(RuntimeError):
    """An authorized checker failed operationally."""


class CheckerExecutionCancelledError(CheckerExecutionError):
    """An authorized checker was cancelled by its caller."""
