"""Operator-controlled visibility for the compiled mathematical catalog."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from jacobian.canonical import canonicalize_json
from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_discovery import normalize_domain


@dataclass(frozen=True, slots=True)
class OperationVisibilityPolicy:
    """Filter search, inspection, and execution without rebuilding a catalog."""

    profile: Literal["DEFAULT", "COMPUTE_VERIFY_NO_RETRIEVAL"] = "DEFAULT"
    allowed_operation_ids: frozenset[str] = frozenset()
    denied_operation_ids: frozenset[str] = frozenset()
    allowed_domains: frozenset[str] = frozenset()
    denied_domains: frozenset[str] = frozenset()
    allowed_tags: frozenset[str] = frozenset()
    denied_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.profile not in {"DEFAULT", "COMPUTE_VERIFY_NO_RETRIEVAL"}:
            raise ValueError(f"unknown operation policy profile: {self.profile!r}")
        if self.profile == "COMPUTE_VERIFY_NO_RETRIEVAL":
            object.__setattr__(self, "denied_tags", self.denied_tags | {"retrieval"})
        for allowed, denied, label in (
            (self.allowed_operation_ids, self.denied_operation_ids, "operation IDs"),
            (self.allowed_domains, self.denied_domains, "domains"),
            (self.allowed_tags, self.denied_tags, "tags"),
        ):
            overlap = allowed & denied
            if overlap:
                raise ValueError(
                    f"operation policy allows and denies the same {label}: "
                    + ", ".join(sorted(str(item) for item in overlap))
                )
        for value in (
            *self.allowed_operation_ids,
            *self.denied_operation_ids,
            *self.allowed_domains,
            *self.denied_domains,
            *self.allowed_tags,
            *self.denied_tags,
        ):
            if not value.strip():
                raise ValueError("operation policy values must not be blank")

    @property
    def definition(self) -> dict[str, object]:
        return {
            "policy_version": "1",
            "profile": self.profile,
            "allowed_operation_ids": sorted(self.allowed_operation_ids),
            "denied_operation_ids": sorted(self.denied_operation_ids),
            "allowed_domains": sorted(self.allowed_domains),
            "denied_domains": sorted(self.denied_domains),
            "allowed_tags": sorted(self.allowed_tags),
            "denied_tags": sorted(self.denied_tags),
            "checker_authorization_affected": False,
        }

    @property
    def digest(self) -> str:
        return (
            "sha256:" + hashlib.sha256(canonicalize_json(self.definition)).hexdigest()
        )

    def project(self, descriptor: OperationDescriptor) -> OperationDescriptor | None:
        return (
            descriptor
            if self.allows(descriptor.operation_id, descriptor.tags)
            else None
        )

    def allows(self, operation_id: str, tags: tuple[str, ...]) -> bool:
        """Return whether compact catalog metadata passes this policy."""

        return not self._denial_reasons(operation_id, tags)

    def denial_reasons(self, descriptor: OperationDescriptor) -> tuple[str, ...]:
        return self._denial_reasons(descriptor.operation_id, descriptor.tags)

    def _denial_reasons(
        self,
        operation_id: str,
        operation_tags: tuple[str, ...],
    ) -> tuple[str, ...]:
        domain = normalize_domain(operation_id.partition(".")[0])
        tags = {normalize_domain(tag) for tag in operation_tags}
        reasons: list[str] = []
        if (
            self.allowed_operation_ids
            and operation_id not in self.allowed_operation_ids
        ):
            reasons.append("operation_id_not_allowed")
        if operation_id in self.denied_operation_ids:
            reasons.append("operation_id_denied")
        if self.allowed_domains and domain not in {
            normalize_domain(value) for value in self.allowed_domains
        }:
            reasons.append("domain_not_allowed")
        if domain in {normalize_domain(value) for value in self.denied_domains}:
            reasons.append("domain_denied")
        allowed_tags = {normalize_domain(value) for value in self.allowed_tags}
        if allowed_tags and not tags & allowed_tags:
            reasons.append("tag_not_allowed")
        if tags & {normalize_domain(value) for value in self.denied_tags}:
            reasons.append("tag_denied")
        return tuple(reasons)


__all__ = ["OperationVisibilityPolicy"]
