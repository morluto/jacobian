"""Shared MCP adapter constants."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Literal

_LOGGER = logging.getLogger(__name__)
CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT = 16_384
CAPABILITY_INSPECTION_RELATIONSHIPS_BYTE_LIMIT = 16_384
CapabilityDescriptionView = Literal["SUMMARY", "CONTRACT", "FULL"]


class ReasoningLogMode(StrEnum):
    REQUIRED = "REQUIRED"
    AUDIT = "AUDIT"
    OFF = "OFF"


_CAPABILITY_SCOPE_RULE = {
    "conclusion_scope": "Only the exact supplied input or claim is covered.",
    "bounded_repetition": (
        "Additional finite or bounded invocations remain finite evidence; they do "
        "not establish an all-orders, all-parameters, or otherwise unbounded "
        "conclusion."
    ),
}
