"""Canonical SAT instance and unverified evidence artifact contracts."""

from __future__ import annotations

import base64
import binascii
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from jacobian.canonical import canonicalize_json, sha256_digest
from jacobian.contracts.capabilities import (
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.results import ContractModel

SatVariableName = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$",
        strict=True,
    ),
]
SatInputClause = Annotated[
    tuple[StrictInt, ...],
    Field(max_length=1_000_000),
]
CanonicalBase64 = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$",
        max_length=8_000_000,
        strict=True,
    ),
]

_PROJECTION_FORMAT: Literal["DIMACS-CNF"] = "DIMACS-CNF"
_PROJECTION_VERSION: Literal["jacobian.dimacs.cnf/v1"] = "jacobian.dimacs.cnf/v1"
_PROOF_FORMAT: Literal["DRAT"] = "DRAT"
_PROOF_FORMAT_VERSION: Literal["drat-text/v1"] = "drat-text/v1"


class SatVariableBinding(ContractModel):
    """One deterministic symbolic-name to DIMACS-ID binding."""

    id: StrictInt = Field(ge=1, le=1_000_000)
    name: SatVariableName

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> object:
        if isinstance(value, str):
            return unicodedata.normalize("NFC", value)
        return value


class SatClause(ContractModel):
    """One canonical disjunction of signed DIMACS literals."""

    literals: tuple[StrictInt, ...] = Field(max_length=1_000_000)

    @model_validator(mode="after")
    def require_canonical_literals(self) -> Self:
        if any(literal == 0 for literal in self.literals):
            raise ValueError("clause literals must be nonzero")
        if len(set(self.literals)) != len(self.literals):
            raise ValueError("clause literals must be unique and canonically ordered")
        if any(-literal in self.literals for literal in self.literals):
            raise ValueError("tautological clauses must be omitted")
        if self.literals != tuple(sorted(self.literals, key=_literal_sort_key)):
            raise ValueError("clause literals must be unique and canonically ordered")
        return self


class CanonicalCnf(ContractModel):
    """Canonical named-variable CNF with a deterministic DIMACS projection."""

    cnf_schema_version: Literal["1"] = "1"
    variables: tuple[SatVariableBinding, ...] = Field(max_length=1_000_000)
    clauses: tuple[SatClause, ...] = Field(max_length=1_000_000)
    projection_format: Literal["DIMACS-CNF"] = _PROJECTION_FORMAT
    projection_version: Literal["jacobian.dimacs.cnf/v1"] = _PROJECTION_VERSION
    variable_map_digest: Sha256Digest
    dimacs_digest: Sha256Digest

    @model_validator(mode="after")
    def require_canonical_instance(self) -> Self:
        expected_variables = tuple(
            (index, variable.name)
            for index, variable in enumerate(
                sorted(self.variables, key=lambda variable: variable.name),
                start=1,
            )
        )
        actual_variables = tuple(
            (variable.id, variable.name) for variable in self.variables
        )
        if actual_variables != expected_variables:
            raise ValueError("variables must use contiguous IDs and ascending names")
        variable_count = len(self.variables)
        if any(
            abs(literal) > variable_count
            for clause in self.clauses
            for literal in clause.literals
        ):
            raise ValueError("literal references an undeclared variable")
        if len(set(self.clauses)) != len(self.clauses) or self.clauses != tuple(
            sorted(self.clauses, key=_clause_sort_key)
        ):
            raise ValueError("clauses must be unique and canonically ordered")
        if self.variable_map_digest != sat_variable_map_digest(self.variables):
            raise ValueError("variable-map digest does not match the canonical map")
        if self.dimacs_digest != sha256_digest(self.to_dimacs_bytes()):
            raise ValueError(
                "DIMACS digest does not match the deterministic projection"
            )
        return self

    def to_dimacs_bytes(self) -> bytes:
        """Project this exact canonical instance to deterministic DIMACS bytes."""

        return _dimacs_bytes(len(self.variables), self.clauses)


class SatCnfBinding(ContractModel):
    """Identity needed to replay evidence against one exact CNF projection."""

    binding_version: Literal["1"] = "1"
    cnf_artifact_uri: ArtifactUri
    cnf_object_digest: Sha256Digest
    cnf_payload_digest: Sha256Digest
    variable_map_digest: Sha256Digest
    dimacs_digest: Sha256Digest
    projection_format: Literal["DIMACS-CNF"]
    projection_version: Literal["jacobian.dimacs.cnf/v1"]
    variable_count: StrictInt = Field(ge=0, le=1_000_000)
    clause_count: StrictInt = Field(ge=0, le=1_000_000)


class SatCnfMaterializationRequest(ContractModel):
    """Bounded symbolic CNF input accepted before any artifact write."""

    request_version: Literal["1"] = "1"
    variable_names: tuple[SatVariableName, ...] = Field(max_length=1_000_000)
    clauses: tuple[SatInputClause, ...] = Field(max_length=1_000_000)

    @model_validator(mode="after")
    def require_valid_bounded_cnf(self) -> Self:
        if sum(len(clause) for clause in self.clauses) > 2_000_000:
            raise ValueError("CNF input may contain at most 2,000,000 literals")
        canonicalize_cnf(
            variable_names=self.variable_names,
            clauses=self.clauses,
        )
        return self


class SatCnfMaterializationOutput(ContractModel):
    """Stored canonical CNF identity consumable by SAT capabilities."""

    materialization_version: Literal["1"] = "1"
    cnf_uri: ArtifactUri
    schema_uri: ArtifactUri
    semantics_uri: ArtifactUri
    cnf_object_digest: Sha256Digest
    cnf_payload_digest: Sha256Digest
    variable_map_digest: Sha256Digest
    dimacs_digest: Sha256Digest
    projection_format: Literal["DIMACS-CNF"]
    projection_version: Literal["jacobian.dimacs.cnf/v1"]
    variable_count: StrictInt = Field(ge=0, le=1_000_000)
    clause_count: StrictInt = Field(ge=0, le=1_000_000)
    variable_bindings: tuple[SatVariableBinding, ...] | None = Field(
        default=None,
        max_length=4096,
        description=(
            "Complete canonical DIMACS-ID-to-name map when it fits the bounded "
            "inline response; otherwise read the CNF artifact."
        ),
    )
    variable_bindings_complete: StrictBool
    caller_order_changed: StrictBool
    variable_order_note: str = Field(min_length=1, max_length=512)


class SatResourceBudget(ContractModel):
    """Declared search budget that produced an unverified evidence artifact."""

    budget_version: Literal["1"] = "1"
    wall_seconds: StrictInt = Field(ge=1, le=86_400)
    memory_bytes: StrictInt | None = Field(
        default=None,
        ge=1,
        le=1 << 50,
    )
    conflicts: StrictInt | None = Field(
        default=None,
        ge=1,
        le=(1 << 53) - 1,
    )


class SatExplorationBudget(ContractModel):
    """CaDiCaL limits that the exploration adapter actually enforces."""

    budget_version: Literal["1"] = "1"
    wall_seconds: StrictInt = Field(
        ge=1,
        le=150,
        description=(
            "Synchronous CaDiCaL wall-time budget. Partition searches expected "
            "to exceed the cross-client-safe 150-second ceiling."
        ),
    )
    conflicts: StrictInt | None = Field(
        default=None,
        ge=1,
        le=(1 << 53) - 1,
    )

    def artifact_budget(self) -> SatResourceBudget:
        """Project enforced limits into durable SAT evidence provenance."""

        return SatResourceBudget(
            wall_seconds=self.wall_seconds,
            conflicts=self.conflicts,
        )


class SatExplorationRequest(ContractModel):
    """Run one bounded producer against an exact canonical CNF artifact."""

    cnf_uri: ArtifactUri
    resource_budget: SatExplorationBudget


class SatAssignmentArtifact(ContractModel):
    """One total, exact, but not self-verifying assignment candidate."""

    assignment_schema_version: Literal["1"] = "1"
    cnf: SatCnfBinding
    declared_scope: Literal["FULL_CNF"] = "FULL_CNF"
    values: tuple[StrictBool, ...] = Field(max_length=1_000_000)
    producer: CapabilityProviderRuntime
    resource_budget: SatResourceBudget

    @model_validator(mode="after")
    def require_total_bound_assignment(self) -> Self:
        if len(self.values) != self.cnf.variable_count:
            raise ValueError(
                "assignment must contain one value for every bound variable"
            )
        _require_available_producer(self.producer)
        return self


class SatAssignmentVerificationRequest(ContractModel):
    """Verify one stored total assignment against its exact bound CNF."""

    assignment_uri: ArtifactUri


class SatAssignmentVerificationOutput(ContractModel):
    """Model-facing projection of one independent assignment replay."""

    status: Literal[
        "VERIFIED_SATISFYING",
        "REJECTED",
        "TIMEOUT",
        "CANCELLED",
        "ERROR",
    ]
    conclusion: Literal["TRUE", "UNKNOWN"]
    cnf_uri: ArtifactUri
    assignment_uri: ArtifactUri
    witness_uri: ArtifactUri
    checker_id: CheckerUri
    verification_record_uri: ArtifactUri | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_verified_projection(self) -> Self:
        if self.status == "VERIFIED_SATISFYING":
            if self.conclusion != "TRUE" or self.verification_record_uri is None:
                raise ValueError(
                    "verified satisfying output requires TRUE and a verification record"
                )
        elif self.conclusion != "UNKNOWN" or self.verification_record_uri is not None:
            raise ValueError(
                "non-verified assignment output cannot carry a conclusion or record"
            )
        return self


class SatModelFindOutput(ContractModel):
    """Unverified result of one bounded model-production attempt."""

    status: Literal["ASSIGNMENT_PRODUCED", "NO_ASSIGNMENT_PRODUCED"]
    solver_status: Literal["SATISFIABLE", "UNSATISFIABLE", "UNKNOWN"]
    conclusion: Literal["UNKNOWN"] = "UNKNOWN"
    cnf_uri: ArtifactUri
    assignment_uri: ArtifactUri | None = None
    assignment: dict[SatVariableName, StrictBool] | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_assignment_to_status(self) -> Self:
        produced = self.status == "ASSIGNMENT_PRODUCED"
        if produced != (
            self.assignment_uri is not None and self.assignment is not None
        ):
            raise ValueError(
                "an assignment-produced result requires both its URI and named values"
            )
        if not produced and (
            self.assignment_uri is not None or self.assignment is not None
        ):
            raise ValueError(
                "a no-assignment result cannot carry an assignment URI or named values"
            )
        if produced and self.solver_status != "SATISFIABLE":
            raise ValueError("an assignment requires a SATISFIABLE solver report")
        return self


class SatUnsatProofFindOutput(ContractModel):
    """Unverified result of one bounded DRAT-production attempt."""

    status: Literal["PROOF_PRODUCED", "NO_PROOF_PRODUCED"]
    solver_status: Literal["SATISFIABLE", "UNSATISFIABLE", "UNKNOWN"]
    conclusion: Literal["UNKNOWN"] = "UNKNOWN"
    cnf_uri: ArtifactUri
    proof_uri: ArtifactUri | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_proof_to_status(self) -> Self:
        produced = self.status == "PROOF_PRODUCED"
        if produced != (self.proof_uri is not None):
            raise ValueError("only a proof-produced result may carry a proof URI")
        if produced and self.solver_status != "UNSATISFIABLE":
            raise ValueError("a proof requires an UNSATISFIABLE solver report")
        return self


class SatProofArtifact(ContractModel):
    """Raw proof bytes bound to one exact CNF without claiming validity."""

    proof_schema_version: Literal["1"] = "1"
    cnf: SatCnfBinding
    declared_scope: Literal["FULL_CNF"] = "FULL_CNF"
    proof_format: Literal["DRAT"] = _PROOF_FORMAT
    proof_format_version: Literal["drat-text/v1"] = _PROOF_FORMAT_VERSION
    proof_encoding: Literal["BASE64"] = "BASE64"
    proof_base64: CanonicalBase64
    proof_digest: Sha256Digest
    producer: CapabilityProviderRuntime
    resource_budget: SatResourceBudget

    @model_validator(mode="after")
    def require_exact_raw_proof(self) -> Self:
        raw = _decode_base64(self.proof_base64)
        if self.proof_base64 != base64.b64encode(raw).decode("ascii"):
            raise ValueError("proof bytes must use canonical base64")
        if self.proof_digest != sha256_digest(raw):
            raise ValueError("raw proof digest does not match the preserved bytes")
        _require_available_producer(self.producer)
        return self

    @classmethod
    def from_bytes(
        cls,
        *,
        cnf: SatCnfBinding,
        proof: bytes,
        producer: CapabilityProviderRuntime,
        resource_budget: SatResourceBudget,
    ) -> SatProofArtifact:
        """Encode exact raw proof bytes without interpreting solver output."""

        return cls(
            cnf=cnf,
            proof_base64=base64.b64encode(proof).decode("ascii"),
            proof_digest=sha256_digest(proof),
            producer=producer,
            resource_budget=resource_budget,
        )

    def raw_bytes(self) -> bytes:
        """Recover the exact proof bytes preserved by this artifact."""

        return _decode_base64(self.proof_base64)


class SatUnsatProofVerificationRequest(ContractModel):
    """Verify one stored raw DRAT proof against its exact bound CNF."""

    proof_uri: ArtifactUri


class SatUnsatProofVerificationOutput(ContractModel):
    """Model-facing projection of one independent DRAT replay."""

    verified_claim_scope: Literal["CANONICAL_CNF_ONLY"] = "CANONICAL_CNF_ONLY"
    status: Literal[
        "VERIFIED_UNSAT",
        "REJECTED",
        "TIMEOUT",
        "CANCELLED",
        "ERROR",
    ]
    conclusion: Literal["TRUE", "UNKNOWN"]
    cnf_uri: ArtifactUri
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


class SatLratResourceLimits(ContractModel):
    """Explicit limits for bounded ASCII LRAT replay."""

    timeout_ms: StrictInt = Field(default=30_000, ge=0, le=120_000)
    max_proof_bytes: StrictInt = Field(default=4_194_304, ge=1, le=67_108_864)
    max_steps: StrictInt = Field(default=100_000, ge=1, le=2_000_000)
    max_hints_per_step: StrictInt = Field(default=100_000, ge=1, le=1_000_000)
    max_clause_literals: StrictInt = Field(default=100_000, ge=0, le=1_000_000)


class SatLratProofArtifact(ContractModel):
    """Exact, unverified ASCII LRAT bytes bound to one canonical CNF."""

    proof_format: Literal["LRAT-ASCII"] = "LRAT-ASCII"
    proof_format_version: Literal["jacobian.lrat.rup/v1"] = "jacobian.lrat.rup/v1"
    cnf: SatCnfBinding
    proof_base64: str = Field(min_length=1)
    proof_digest: Sha256Digest
    proof_byte_count: StrictInt = Field(ge=1)
    limits: SatLratResourceLimits

    @model_validator(mode="after")
    def bind_exact_bytes(self) -> Self:
        raw = _decode_base64(self.proof_base64)
        if self.proof_base64 != base64.b64encode(raw).decode("ascii"):
            raise ValueError("LRAT proof must use canonical base64")
        if self.proof_digest != sha256_digest(raw) or self.proof_byte_count != len(raw):
            raise ValueError("LRAT proof digest or byte count does not match")
        if len(raw) > self.limits.max_proof_bytes:
            raise ValueError("LRAT proof exceeds its declared byte limit")
        return self

    @classmethod
    def from_bytes(
        cls,
        *,
        cnf: SatCnfBinding,
        proof: bytes,
        limits: SatLratResourceLimits,
    ) -> Self:
        return cls(
            cnf=cnf,
            proof_base64=base64.b64encode(proof).decode("ascii"),
            proof_digest=sha256_digest(proof),
            proof_byte_count=len(proof),
            limits=limits,
        )


class SatLratVerificationRequest(ContractModel):
    """Replay exact ASCII LRAT against one canonical CNF."""

    cnf_uri: ArtifactUri
    proof_base64: str = Field(min_length=1)
    limits: SatLratResourceLimits = SatLratResourceLimits()
    cancelled: bool = False

    @model_validator(mode="after")
    def validate_proof_encoding(self) -> Self:
        max_encoded_length = 4 * ((self.limits.max_proof_bytes + 2) // 3)
        if len(self.proof_base64) > max_encoded_length:
            raise ValueError("LRAT proof exceeds max_proof_bytes")
        raw = _decode_base64(self.proof_base64)
        if self.proof_base64 != base64.b64encode(raw).decode("ascii"):
            raise ValueError("LRAT proof must use canonical base64")
        if not raw:
            raise ValueError("LRAT proof must not be empty")
        if len(raw) > self.limits.max_proof_bytes:
            raise ValueError("LRAT proof exceeds max_proof_bytes")
        return self


class SatLratVerificationOutput(ContractModel):
    """Fail-closed projection of independent LRAT replay."""

    status: Literal[
        "VERIFIED_UNSAT",
        "REJECTED",
        "UNSUPPORTED",
        "TIMEOUT",
        "CANCELLED",
        "ERROR",
    ]
    conclusion: Literal["TRUE", "UNKNOWN"]
    cnf_uri: ArtifactUri
    proof_uri: ArtifactUri
    certificate_uri: ArtifactUri
    checker_id: CheckerUri
    verification_record_uri: ArtifactUri | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_verified_projection(self) -> Self:
        if self.status == "VERIFIED_UNSAT":
            if self.conclusion != "TRUE" or self.verification_record_uri is None:
                raise ValueError(
                    "verified LRAT requires TRUE and a verification record"
                )
        elif self.conclusion != "UNKNOWN" or self.verification_record_uri is not None:
            raise ValueError("non-verified LRAT cannot carry a conclusion or record")
        return self


def canonicalize_cnf(
    *,
    variable_names: Sequence[str],
    clauses: Iterable[Iterable[int]],
) -> CanonicalCnf:
    """Normalize one named CNF and return its unique artifact payload.

    Input literal IDs refer to the supplied variable-name order. Names are NFC
    normalized and sorted; literals are renumbered with that map. Duplicate
    literals and clauses are removed, and tautological clauses are omitted.
    """

    normalized_names: list[str] = []
    for name in variable_names:
        if not isinstance(name, str):
            raise ValueError("variable names must be strings")
        normalized_names.append(unicodedata.normalize("NFC", name))
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("variable names must be unique after NFC normalization")

    indexed_names = tuple(enumerate(normalized_names, start=1))
    sorted_names = tuple(sorted(indexed_names, key=lambda item: item[1]))
    old_to_new = {
        old_id: new_id for new_id, (old_id, _name) in enumerate(sorted_names, start=1)
    }
    variables = tuple(
        SatVariableBinding(id=new_id, name=name)
        for new_id, (_old_id, name) in enumerate(sorted_names, start=1)
    )

    canonical_clauses: set[tuple[int, ...]] = set()
    for raw_clause in clauses:
        remapped: set[int] = set()
        tautological = False
        for literal in raw_clause:
            if (
                isinstance(literal, bool)
                or not isinstance(literal, int)
                or literal == 0
            ):
                raise ValueError("clause literals must be nonzero integers")
            old_id = abs(literal)
            try:
                new_id = old_to_new[old_id]
            except KeyError as exc:
                raise ValueError(
                    "clause literal references an undeclared variable"
                ) from exc
            mapped = new_id if literal > 0 else -new_id
            if -mapped in remapped:
                tautological = True
            remapped.add(mapped)
        if not tautological:
            canonical_clauses.add(tuple(sorted(remapped, key=_literal_sort_key)))

    clause_models = tuple(
        SatClause(literals=literals)
        for literals in sorted(canonical_clauses, key=_literal_tuple_sort_key)
    )
    return CanonicalCnf(
        variables=variables,
        clauses=clause_models,
        variable_map_digest=sat_variable_map_digest(variables),
        dimacs_digest=sha256_digest(_dimacs_bytes(len(variables), clause_models)),
    )


def sat_variable_map_digest(
    variables: Sequence[SatVariableBinding],
) -> str:
    """Digest one exact variable map under a domain-separated wire format."""

    payload = {
        "variable_map_format": "jacobian.sat.variable-map/v1",
        "variables": [variable.model_dump(mode="json") for variable in variables],
    }
    return sha256_digest(canonicalize_json(payload))


def _literal_sort_key(literal: int) -> tuple[int, bool]:
    return abs(literal), literal > 0


def _literal_tuple_sort_key(
    literals: tuple[int, ...],
) -> tuple[tuple[int, bool], ...]:
    return tuple(_literal_sort_key(literal) for literal in literals)


def _clause_sort_key(
    clause: SatClause,
) -> tuple[tuple[int, bool], ...]:
    return _literal_tuple_sort_key(clause.literals)


def _dimacs_bytes(
    variable_count: int,
    clauses: Sequence[SatClause],
) -> bytes:
    rows = [f"p cnf {variable_count} {len(clauses)}\n"]
    for clause in clauses:
        prefix = " ".join(str(literal) for literal in clause.literals)
        rows.append(f"{prefix} 0\n" if prefix else "0\n")
    return "".join(rows).encode("ascii")


def _decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
        raise ValueError("proof bytes must use canonical base64") from exc


def _require_available_producer(producer: CapabilityProviderRuntime) -> None:
    if producer.availability is not CapabilityProviderAvailability.AVAILABLE:
        raise ValueError("SAT evidence requires an available producer runtime")
