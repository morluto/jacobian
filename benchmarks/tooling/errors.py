"""Shared benchmark tooling error type.

``HarborSuiteError`` is the canonical exception for the benchmark control
plane.  It lives here, in a dependency-free leaf module, so that strict
boundary validators (``strict_boundaries``), suite validators
(``harbor_suite``), and downstream tooling (``heldout_bundle``,
``heldout_runner``, ``observation_results``, ...) can all raise and catch
the same error without creating import cycles.
"""

from __future__ import annotations


class HarborSuiteError(ValueError):
    """The pinned benchmark control plane detected a contract violation."""


__all__ = ["HarborSuiteError"]
