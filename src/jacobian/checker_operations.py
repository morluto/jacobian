"""Typed declarations for operator-authorized checker implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.operations import ProviderObservation
from jacobian.contracts.results import ContractModel

CheckerObservationLoader = Callable[[], ProviderObservation]

# Producer operation verb segments stripped when deriving a verifier operation
# ID. Each producer operation ID contains exactly one of these segments; the
# derived verifier ID removes it and appends ``.verify``.
_VERB_SEGMENTS: tuple[str, ...] = (
    "compute",
    "decide",
    "materialize",
    "canonicalize",
    "evaluate",
    "count",
    "classify",
)


def derive_verification_operation_id(producer_operation_id: str) -> str:
    """Derive a per-producer verifier ID by stripping the verb and appending ``.verify``.

    The producer operation verb (``compute``, ``decide``, ``materialize``,
    ``canonicalize``, ``evaluate``, or ``count``) is removed from wherever it appears in the
    producer operation ID, and ``.verify`` is appended. For example
    ``polynomial.compute.gcd`` becomes ``polynomial.gcd.verify`` and
    ``integer.decide.powerful`` becomes ``integer.powerful.verify``.
    """

    segments = producer_operation_id.split(".")
    verb_indices = [
        index for index, segment in enumerate(segments) if segment in _VERB_SEGMENTS
    ]
    if len(verb_indices) != 1:
        raise ValueError(
            "producer operation ID must contain exactly one operation verb "
            f"segment {_VERB_SEGMENTS}: {producer_operation_id!r}"
        )
    remaining = [
        segment for index, segment in enumerate(segments) if index != verb_indices[0]
    ]
    if not remaining:
        raise ValueError(
            "producer operation ID must have a non-empty stem after removing "
            f"the operation verb: {producer_operation_id!r}"
        )
    return ".".join(remaining) + ".verify"


def _domain_label(producer_operation_id: str) -> str:
    return producer_operation_id.split(".", 1)[0]


def derive_verification_title(producer_operation_id: str) -> str:
    """Strictly construct a verifier title from the producer operation ID."""

    return f"Verify an exact {producer_operation_id} result"


def derive_verification_description(producer_operation_id: str) -> str:
    """Strictly construct a verifier description from the producer operation ID."""

    return (
        f"Independently replay the exact {producer_operation_id} relation "
        "against its inline input and candidate, binding their canonical "
        "digests in the accepted verification record through the "
        "operator-authorized independent checker."
    )


def derive_verification_tags(producer_operation_id: str) -> tuple[str, ...]:
    """Strictly construct verifier tags from the producer operation ID."""

    return ("verification", "exact", _domain_label(producer_operation_id))


@dataclass(frozen=True, slots=True)
class AuthorizedChecker:
    """Domain-owned declaration of an independently replayable exact result.

    Every declaration carries complete verification operation metadata after
    construction. If the domain provides explicit metadata, the
    ``verification_operation_id`` must equal the derived form (see
    :func:`derive_verification_operation_id`) and the title, description, and
    tags are used as written. If the domain omits the metadata, all four fields
    are strictly constructed from the producer operation ID so that no
    verifier metadata is absent at installation.
    """

    operation_id: str
    request_model: type[ContractModel]
    function: str
    format_id: str
    observation_loader: CheckerObservationLoader
    entrypoint_module: str = "jacobian_checkers.exact_domain_operations"
    replay_method: str = "Python-FLINT exact replay"
    reason: str = (
        "operator-authorized Python-FLINT exact replay independent of the "
        "SymPy producer"
    )
    verification_operation_id: str | None = None
    verification_title: str | None = None
    verification_description: str | None = None
    verification_tags: tuple[str, ...] = ()
    optional: bool = False
    supports_input: Callable[[object], bool] | None = None

    def __post_init__(self) -> None:
        for field_name, value in {
            "operation_id": self.operation_id,
            "function": self.function,
            "format_id": self.format_id,
            "entrypoint_module": self.entrypoint_module,
            "replay_method": self.replay_method,
            "reason": self.reason,
        }.items():
            if not value.strip():
                raise ValueError(
                    f"exact replay checker declaration {field_name} must not be empty"
                )
        derived_id = derive_verification_operation_id(self.operation_id)
        explicit_text = (
            self.verification_title,
            self.verification_description,
        )
        if self.verification_operation_id is None:
            if any(value is not None for value in explicit_text):
                raise ValueError(
                    "verification title or description require a verification "
                    "operation ID"
                )
            object.__setattr__(self, "verification_operation_id", derived_id)
            object.__setattr__(
                self,
                "verification_title",
                derive_verification_title(self.operation_id),
            )
            object.__setattr__(
                self,
                "verification_description",
                derive_verification_description(self.operation_id),
            )
            object.__setattr__(
                self,
                "verification_tags",
                derive_verification_tags(self.operation_id),
            )
            return
        if self.verification_operation_id != derived_id:
            raise ValueError(
                "verification operation ID must equal the derived form "
                f"{derived_id!r} (strip the producer verb and append '.verify'): "
                f"{self.verification_operation_id!r}"
            )
        if not all(isinstance(value, str) and value.strip() for value in explicit_text):
            raise ValueError(
                "verification operation ID, title, and description must be "
                "declared together"
            )
        if not self.verification_tags:
            object.__setattr__(
                self,
                "verification_tags",
                derive_verification_tags(self.operation_id),
            )


@dataclass(frozen=True, slots=True)
class CheckerOperation:
    """One independently executable checker and its compatibility scope."""

    name: str
    entrypoint: str
    evidence_kind: EvidenceKind
    format_id: str
    format_version: str
    claim_schema_uris: tuple[str, ...]
    semantics_uris: tuple[str, ...]
    candidate_schema_uris: tuple[str, ...]
    reason: str
    target_schema_uris: tuple[str, ...] = ()
    target_semantics_uris: tuple[str, ...] = ()
    provider_runtime: ProviderObservation | None = None

    def __post_init__(self) -> None:
        required_text = {
            "name": self.name,
            "entrypoint": self.entrypoint,
            "format_id": self.format_id,
            "format_version": self.format_version,
            "reason": self.reason,
        }
        for field_name, value in required_text.items():
            if not value.strip():
                raise ValueError(f"checker operation {field_name} must not be empty")
        if not self.claim_schema_uris:
            raise ValueError("checker operation must declare a claim schema")
        if not self.semantics_uris:
            raise ValueError("checker operation must declare semantics")


@dataclass(frozen=True, slots=True)
class InstalledChecker:
    """Authorization result for one checker operation."""

    operation: CheckerOperation
    checker_id: str | None

    @property
    def authorized(self) -> bool:
        return self.checker_id is not None

    def require_checker_id(self) -> str:
        if self.checker_id is None:
            raise ValueError("checker operation is not authorized")
        return self.checker_id
