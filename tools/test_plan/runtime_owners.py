"""Shared ownership rules for complete-runtime construction and fixtures."""

from __future__ import annotations

from pathlib import PurePosixPath

# Fixtures: complete portfolio templates/instances may only appear under these
# owning trees (plugin collection policy).
COMPLETE_RUNTIME_FIXTURE_OWNERS: tuple[str, ...] = (
    "tests/composition/",
    "tests/e2e/",
    "tests/boundary/storage/",
    "tests/boundary/providers/",
    "tests/boundary/mcp/",
    "tests/support/",
)

# Direct create_runtime / portfolio construction allowlist for architecture lint.
_SUPPORT_CREATE_RUNTIME_MODULES = frozenset(
    {
        "tests/support/runtime_templates.py",
        "tests/support/runtime_instances.py",
        "tests/support/runtime_profiles.py",
    }
)
_CREATE_RUNTIME_BOUNDARY_PARTS = frozenset({"runtime", "startup", "recovery"})


def allows_complete_runtime_fixture(relative: str) -> bool:
    """Return whether a test module may request complete-runtime fixtures."""

    return any(
        relative.startswith(prefix) for prefix in COMPLETE_RUNTIME_FIXTURE_OWNERS
    )


def allows_create_runtime(relative: str, *, tier: str | None = None) -> bool:
    """Return whether a module may call ``create_runtime`` / install portfolios."""

    path = PurePosixPath(relative)
    if relative in _SUPPORT_CREATE_RUNTIME_MODULES:
        return True
    resolved_tier = tier
    if resolved_tier is None:
        parts = path.parts
        if len(parts) >= 2 and parts[0] == "tests":
            resolved_tier = parts[1]
    if resolved_tier in {"composition", "e2e"}:
        return True
    if resolved_tier == "boundary":
        return any(part in _CREATE_RUNTIME_BOUNDARY_PARTS for part in path.parts)
    return False
