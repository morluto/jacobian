"""Shared signals that justify authorized-checker hydration in tests.

Substring matches are intentionally avoided: ``VERIFIED`` must not match inside
``UNVERIFIED``, and a bare ``.verify`` fragment is not enough unless it appears
as an invoked ``capability_id="….verify"`` literal (or a verification
service/call).
"""

from __future__ import annotations

import re

_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<![A-Za-z_])VERIFIED(?![A-Za-z_])"),
    re.compile(r"(?<![A-Za-z_])CapabilityAssuranceLevel\.VERIFIED(?![A-Za-z_])"),
    re.compile(r"(?<![A-Za-z_])CheckerAuthorityMode(?![A-Za-z_])"),
    re.compile(r"(?<![A-Za-z_])HYDRATE_EXISTING(?![A-Za-z_])"),
    re.compile(r"(?<![A-Za-z_])INSTALL_BUNDLED(?![A-Za-z_])"),
    re.compile(r"(?<![A-Za-z_])verification_record(?![A-Za-z_])"),
    re.compile(r"(?<![A-Za-z_])authorize_checker(?![A-Za-z_])"),
    re.compile(r"\bverify_witness\b"),
    re.compile(r"\bverify_certificate\b"),
    re.compile(r"\bverify_transformation\b"),
    re.compile(r"\bservices\.verification\b"),
    re.compile(r'capability_id\s*=\s*["\'][^"\']+\.verify["\']'),
    re.compile(r"checker_id\s+is\s+not\s+None"),
    re.compile(r"checker_ids\s*(?:!=\s*\(\)|is\s+not\s+None)"),
)


def has_verify_authority_signal(source: str) -> bool:
    """Return whether module source asserts verify/authority behavior."""

    return any(pattern.search(source) for pattern in _TOKEN_PATTERNS)
