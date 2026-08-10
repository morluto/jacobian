"""Shared signals that justify authorized-checker hydration in tests."""

from __future__ import annotations

VERIFY_AUTHORITY_SIGNALS: tuple[str, ...] = (
    "VERIFIED",
    "VERIFY",
    "verify_",
    ".verify",
    "certificate.verify",
    "CheckerAuthorityMode",
    "HYDRATE_EXISTING",
    "INSTALL_BUNDLED",
    "authorized checker",
    "verification_record",
    "authority",
    "checker_id",
    "jacobian_authorized_runtime",
)
