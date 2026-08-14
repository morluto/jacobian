"""Test-only switches for exercising catalog compilation variants."""

from __future__ import annotations

from enum import StrEnum


class CheckerAuthorityMode(StrEnum):
    """Legacy test vocabulary mapped onto explicit catalog-build behavior."""

    NONE = "NONE"
    INSTALL_BUNDLED = "INSTALL_BUNDLED"
    HYDRATE_EXISTING = "HYDRATE_EXISTING"


__all__ = ["CheckerAuthorityMode"]
