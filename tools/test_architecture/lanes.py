"""Directory-prefix ownership for the test suite.

The filesystem is the metadata. A test under ``tests/domain/`` is a domain
test; Lean lives under ``tests/boundary/providers/lean/``.
"""

from __future__ import annotations

# Longest prefix wins. Lean is a provider subdirectory with its own lane.
DIRECTORY_LANES: tuple[tuple[str, str, str], ...] = (
    ("lean", "boundary", "tests/boundary/providers/lean"),
    ("provider", "boundary", "tests/boundary/providers"),
    ("storage", "boundary", "tests/boundary/storage"),
    ("process", "boundary", "tests/boundary/process"),
    ("mcp", "boundary", "tests/boundary/mcp"),
    ("unit", "unit", "tests/unit"),
    ("component", "component", "tests/component"),
    ("domain", "domain", "tests/domain"),
    ("composition", "composition", "tests/composition"),
    ("e2e", "e2e", "tests/e2e"),
)


def owner_for(relative: str) -> str | None:
    """Return the unique semantic owner for a test path, if any."""

    path = relative.replace("\\", "/")
    for name, _tier, prefix in DIRECTORY_LANES:
        if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
            return name
    return None


def owners(relative: str) -> tuple[str, ...]:
    """Return matching owners; directory prefixes are exclusive."""

    owner = owner_for(relative)
    return (owner,) if owner is not None else ()


def tier_for(relative: str) -> str | None:
    """Return the semantic tier for a test path, if any."""

    path = relative.replace("\\", "/")
    for _name, tier, prefix in DIRECTORY_LANES:
        if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
            return tier
    return None
