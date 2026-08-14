"""Pinned quantifier-free SMT-LIB and unverified Alethe evidence contracts."""

from __future__ import annotations

import base64
import binascii
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    model_validator,
)

from jacobian.canonical import sha256_digest
from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderDigestKind,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.contracts.results import ContractModel

SmtLogic = Literal["QF_UF", "QF_LIA", "QF_LRA"]
SmtLibText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=1_000_000, strict=True),
]
CanonicalBase64 = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$",
        max_length=8_000_000,
        strict=True,
    ),
]

_PROFILE: Literal["jacobian.smtlib2.qf-unsat/v1"] = "jacobian.smtlib2.qf-unsat/v1"
_INPUT_LANGUAGE: Literal["SMT-LIB-2.6"] = "SMT-LIB-2.6"
_PROOF_FORMAT_VERSION: Literal["cvc5.alethe/1.3.4"] = "cvc5.alethe/1.3.4"
_ALLOWED_COMMANDS = frozenset(
    {
        "set-logic",
        "declare-sort",
        "declare-fun",
        "declare-const",
        "assert",
        "check-sat",
    }
)
_ALETHE_HOLE_MARKER = b":rule hole"


def _decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("proof bytes must use valid canonical base64") from exc


def _scan_smtlib_string_literal(
    text: str, index: int, length: int, tokens: list[str]
) -> int:
    start = index
    index += 1
    while index < length:
        if text[index] != '"':
            index += 1
            continue
        if index + 1 < length and text[index + 1] == '"':
            index += 2
            continue
        index += 1
        tokens.append(text[start:index])
        break
    else:
        raise ValueError("SMT-LIB input contains an unterminated string")
    return index


def _scan_smtlib_quoted_symbol(
    text: str, index: int, length: int, tokens: list[str]
) -> int:
    start = index
    index += 1
    while index < length and text[index] != "|":
        if text[index] == "\\":
            raise ValueError(
                "SMT-LIB quoted symbols in this profile cannot contain backslash"
            )
        index += 1
    if index >= length:
        raise ValueError("SMT-LIB input contains an unterminated quoted symbol")
    index += 1
    tokens.append(text[start:index])
    return index


def _scan_smtlib_simple_token(
    text: str, index: int, length: int, tokens: list[str]
) -> int:
    start = index
    while index < length and not text[index].isspace() and text[index] not in "();":
        if text[index] in '"|':
            raise ValueError("SMT-LIB token contains an unexpected quote")
        index += 1
    if start == index:
        raise ValueError("SMT-LIB input contains an invalid token")
    tokens.append(text[start:index])
    return index


def _smtlib_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        character = text[index]
        if character.isspace():
            index += 1
            continue
        if character == ";":
            newline = text.find("\n", index)
            index = length if newline < 0 else newline + 1
            continue
        if character in "()":
            tokens.append(character)
            index += 1
            continue
        if character == '"':
            index = _scan_smtlib_string_literal(text, index, length, tokens)
            continue
        if character == "|":
            index = _scan_smtlib_quoted_symbol(text, index, length, tokens)
            continue
        index = _scan_smtlib_simple_token(text, index, length, tokens)
    return tuple(tokens)


def _open_smtlib_command(
    depth: int, direct_atoms: list[str] | None
) -> tuple[int, list[str] | None]:
    if depth == 0:
        direct_atoms = []
    depth += 1
    if depth > 512:
        raise ValueError("SMT-LIB input exceeds the profile nesting limit")
    return depth, direct_atoms


def _close_smtlib_command(
    depth: int,
    direct_atoms: list[str] | None,
    commands: list[tuple[str, ...]],
) -> tuple[int, list[str] | None]:
    if depth == 0:
        raise ValueError("SMT-LIB input contains an unmatched closing parenthesis")
    if depth == 1:
        assert direct_atoms is not None
        if not direct_atoms:
            raise ValueError("SMT-LIB top-level command cannot be empty")
        commands.append(tuple(direct_atoms))
        direct_atoms = None
    depth -= 1
    return depth, direct_atoms


def _top_level_commands(text: str) -> tuple[tuple[str, ...], ...]:
    commands: list[tuple[str, ...]] = []
    direct_atoms: list[str] | None = None
    depth = 0
    for token in _smtlib_tokens(text):
        if token == "(":
            depth, direct_atoms = _open_smtlib_command(depth, direct_atoms)
            continue
        if token == ")":
            depth, direct_atoms = _close_smtlib_command(depth, direct_atoms, commands)
            continue
        if depth == 0:
            raise ValueError("SMT-LIB input must contain only top-level commands")
        if depth == 1:
            assert direct_atoms is not None
            direct_atoms.append(token)
    if depth:
        raise ValueError("SMT-LIB input contains an unmatched opening parenthesis")
    return tuple(commands)


def _validate_single_query_profile(text: str, logic: SmtLogic) -> None:
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("SMT-LIB profile requires ASCII input") from exc
    if not text.endswith("\n") or "\r" in text:
        raise ValueError("SMT-LIB profile requires LF endings and a final LF")
    if any(byte < 32 and byte not in {9, 10} for byte in raw):
        raise ValueError("SMT-LIB profile contains a disallowed control byte")
    commands = _top_level_commands(text)
    if not commands:
        raise ValueError("SMT-LIB input must contain one query")
    heads = tuple(command[0] for command in commands)
    unsupported = tuple(head for head in heads if head not in _ALLOWED_COMMANDS)
    if unsupported:
        raise ValueError(f"SMT-LIB profile does not allow command {unsupported[0]!r}")
    if commands[0] != ("set-logic", logic):
        raise ValueError("SMT-LIB query must begin with its exact declared set-logic")
    if heads.count("set-logic") != 1:
        raise ValueError("SMT-LIB query must contain exactly one set-logic command")
    if commands[-1] != ("check-sat",) or heads.count("check-sat") != 1:
        raise ValueError(
            "SMT-LIB query must end with exactly one argument-free check-sat"
        )


class SmtResourceBudget(ContractModel):
    """Declared wall-time budget that produced unverified SMT evidence."""

    budget_version: Literal["1"] = "1"
    wall_seconds: StrictInt = Field(ge=1, le=300)


class SmtExplorationBudget(ContractModel):
    """cvc5 limits enforced by both its solver and the worker process."""

    budget_version: Literal["1"] = "1"
    wall_seconds: StrictInt = Field(
        ge=1,
        le=150,
        description=(
            "Synchronous cvc5 wall-time budget. Partition searches expected to "
            "exceed the cross-client-safe 150-second ceiling."
        ),
    )

    def artifact_budget(self) -> SmtResourceBudget:
        return SmtResourceBudget(wall_seconds=self.wall_seconds)


class SmtUnsatProofFindRequest(ContractModel):
    """Run one bounded Alethe producer on one pinned-profile query."""

    logic: SmtLogic
    smtlib_text: SmtLibText
    resource_budget: SmtExplorationBudget

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        _validate_single_query_profile(self.smtlib_text, self.logic)
        return self


class SmtProblemArtifact(ContractModel):
    """Exact single-query SMT-LIB input framed by the pinned profile."""

    problem_schema_version: Literal["1"] = "1"
    profile: Literal["jacobian.smtlib2.qf-unsat/v1"] = _PROFILE
    input_language: Literal["SMT-LIB-2.6"] = _INPUT_LANGUAGE
    logic: SmtLogic
    query_scope: Literal["SINGLE_CHECK_SAT"] = "SINGLE_CHECK_SAT"
    smtlib_text: SmtLibText
    smtlib_digest: Sha256Digest

    @model_validator(mode="after")
    def require_exact_profile_input(self) -> Self:
        _validate_single_query_profile(self.smtlib_text, self.logic)
        if self.smtlib_digest != sha256_digest(self.smtlib_text.encode("ascii")):
            raise ValueError("SMT-LIB digest does not match the exact input bytes")
        return self

    @classmethod
    def from_text(
        cls,
        *,
        logic: SmtLogic,
        smtlib_text: str,
    ) -> SmtProblemArtifact:
        return cls(
            logic=logic,
            smtlib_text=smtlib_text,
            smtlib_digest=sha256_digest(smtlib_text.encode("ascii")),
        )

    def raw_bytes(self) -> bytes:
        return self.smtlib_text.encode("ascii")


class SmtProblemBinding(ContractModel):
    """Identity needed to replay a proof against one exact SMT-LIB query."""

    binding_version: Literal["1"] = "1"
    problem_artifact_uri: ArtifactUri
    problem_object_digest: Sha256Digest
    problem_payload_digest: Sha256Digest
    logic: SmtLogic
    profile: Literal["jacobian.smtlib2.qf-unsat/v1"]
    input_language: Literal["SMT-LIB-2.6"]
    smtlib_digest: Sha256Digest


class SmtAletheProofArtifact(ContractModel):
    """Raw Alethe bytes bound to one query without claiming proof validity."""

    proof_schema_version: Literal["1"] = "1"
    problem: SmtProblemBinding
    declared_scope: Literal["FULL_QUERY"] = "FULL_QUERY"
    proof_format: Literal["ALETHE"] = "ALETHE"
    proof_format_version: Literal["cvc5.alethe/1.3.4"] = _PROOF_FORMAT_VERSION
    proof_encoding: Literal["BASE64"] = "BASE64"
    proof_base64: CanonicalBase64
    proof_digest: Sha256Digest
    alethe_hole_count: StrictInt = Field(ge=0, le=1_000_000)
    contains_holes: StrictBool
    producer: ProviderObservation
    resource_budget: SmtResourceBudget

    @model_validator(mode="after")
    def require_exact_raw_proof(self) -> Self:
        proof = _decode_base64(self.proof_base64)
        if self.proof_base64 != base64.b64encode(proof).decode("ascii"):
            raise ValueError("proof bytes must use canonical base64")
        if self.proof_digest != sha256_digest(proof):
            raise ValueError("Alethe proof digest does not match the preserved bytes")
        holes = proof.count(_ALETHE_HOLE_MARKER)
        if self.alethe_hole_count != holes or self.contains_holes != (holes > 0):
            raise ValueError("Alethe hole metadata does not match the proof bytes")
        if (
            self.producer.provider != "cvc5"
            or self.producer.version != "1.3.4"
            or self.producer.availability is not ProviderAvailability.AVAILABLE
            or self.producer.digest is None
            or self.producer.digest_kind
            is not ProviderDigestKind.PYTHON_DISTRIBUTION_RECORD
            or self.producer.install_tier is not ProviderInstallTier.T1
            or "alethe-proof-production" not in self.producer.features
            or self.producer.configuration.get("profile") != _PROFILE
            or self.producer.configuration.get("proof_format") != _PROOF_FORMAT_VERSION
        ):
            raise ValueError("proof producer must be an available pinned cvc5 runtime")
        return self

    @classmethod
    def from_bytes(
        cls,
        *,
        problem: SmtProblemBinding,
        proof: bytes,
        producer: ProviderObservation,
        resource_budget: SmtResourceBudget,
    ) -> SmtAletheProofArtifact:
        holes = proof.count(_ALETHE_HOLE_MARKER)
        return cls(
            problem=problem,
            proof_base64=base64.b64encode(proof).decode("ascii"),
            proof_digest=sha256_digest(proof),
            alethe_hole_count=holes,
            contains_holes=holes > 0,
            producer=producer,
            resource_budget=resource_budget,
        )

    def raw_bytes(self) -> bytes:
        return _decode_base64(self.proof_base64)


class SmtUnsatProofFindOutput(ContractModel):
    """Unverified result of one bounded cvc5 Alethe-production attempt."""

    status: Literal["PROOF_PRODUCED", "NO_PROOF_PRODUCED"]
    solver_status: Literal["SATISFIABLE", "UNSATISFIABLE", "UNKNOWN"]
    problem_uri: ArtifactUri
    proof_uri: ArtifactUri | None = None
    contains_holes: StrictBool | None = None
    alethe_hole_count: StrictInt | None = Field(default=None, ge=0, le=1_000_000)
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_proof_to_status(self) -> Self:
        produced = self.status == "PROOF_PRODUCED"
        has_proof_fields = (
            self.proof_uri is not None
            and self.contains_holes is not None
            and self.alethe_hole_count is not None
        )
        if produced != has_proof_fields:
            raise ValueError("only a proof-produced result may carry Alethe evidence")
        if not produced and any(
            value is not None
            for value in (
                self.proof_uri,
                self.contains_holes,
                self.alethe_hole_count,
            )
        ):
            raise ValueError("a no-proof result cannot carry Alethe evidence")
        if produced and self.solver_status != "UNSATISFIABLE":
            raise ValueError("a proof requires an UNSATISFIABLE solver report")
        if produced and self.contains_holes != (self.alethe_hole_count > 0):  # type: ignore[operator]
            raise ValueError("Alethe hole fields are inconsistent")
        return self


class SmtUnsatProofVerificationRequest(ContractModel):
    """Verify one stored Alethe proof against its exact bound SMT query."""

    proof_uri: ArtifactUri


class SmtUnsatProofVerificationOutput(ContractModel):
    """Model-facing projection of one independent strict Carcara replay."""

    status: Literal[
        "VERIFIED_UNSAT",
        "REJECTED",
        "TIMEOUT",
        "CANCELLED",
        "ERROR",
    ]
    conclusion: Literal["TRUE", "UNKNOWN"]
    problem_uri: ArtifactUri
    proof_uri: ArtifactUri
    certificate_uri: ArtifactUri
    checker_id: CheckerUri
    verification_record_uri: ArtifactUri | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_verified_unsat_projection(self) -> Self:
        if self.status == "VERIFIED_UNSAT":
            if self.conclusion != "TRUE" or self.verification_record_uri is None:
                raise ValueError(
                    "verified UNSAT output requires TRUE and a verification record"
                )
        elif self.conclusion != "UNKNOWN" or self.verification_record_uri is not None:
            raise ValueError(
                "non-verified proof output cannot carry a conclusion or record"
            )
        return self
