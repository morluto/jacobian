"""SymPy-backed exact rational univariate polynomial interval enclosures.

This module installs two domain-atomic capabilities over one mathematical
outcome: an exact rational enclosure of the values of one univariate rational
polynomial on one closed rational interval, derived from the Bernstein-
coefficient bound.

``polynomial.interval.enclose`` (EXPLORE) computes the enclosure with pinned
SymPy rational arithmetic and emits an inspectable enclosure artifact. The
result is ``COMPUTED`` and ``UNVERIFIED``; the descriptor advertises the
independent Bernstein-coefficient checker as the verification boundary, but the
capability does not invoke it.

``polynomial.interval.enclosure.verify`` (VERIFY) packages the claimed
enclosure as a replay certificate and asks the operator-authorized independent
checker to replay the Bernstein coefficients from the source polynomial and
interval using pure ``fractions.Fraction`` arithmetic. The checker does not
import SymPy and does not depend on this adapter.

The enclosure is a valid superset of the polynomial's range on the interval,
not the exact range. Both contracts carry ``range_exactness =
ENCLOSURE_VALID_NOT_EXACT`` so that a valid bound is never mistaken for the
exact image.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.capability_service import CapabilityInvocationError
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityMode,
    CapabilityRelationship,
    CapabilityRelationshipStatus,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.polynomial_intervals import (
    PolynomialIntervalEnclosure,
    PolynomialIntervalEnclosureClaim,
    PolynomialIntervalEnclosureOutput,
    PolynomialIntervalEnclosureReplay,
    PolynomialIntervalEnclosureRequest,
    PolynomialIntervalEnclosureVerifyOutput,
    PolynomialIntervalEnclosureVerifyRequest,
    RationalInterval,
    UnivariateRationalPolynomial,
)
from jacobian.contracts.results import (
    Conclusion,
    ContractModel,
    Execution,
    ExecutionStatus,
    Verification,
)
from jacobian.domains._examples import example
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime
from jacobian.providers import LazyLoader
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService

if TYPE_CHECKING:
    from sympy import Symbol


class _SympyBackend(NamedTuple):
    """Heavy SymPy implementation symbols loaded on first capability invocation."""

    QQ: Any
    Poly: Any
    expand: Any
    symbols: Any
    PolynomialError: type


def _load_sympy_backend() -> _SympyBackend:
    """Construct the pinned SymPy implementation bundle on first use."""
    from sympy import QQ, Poly, expand, symbols
    from sympy.polys.polyerrors import PolynomialError

    return _SympyBackend(QQ, Poly, expand, symbols, PolynomialError)


_sympy: LazyLoader[_SympyBackend] = LazyLoader(
    _load_sympy_backend, component_id="jacobian.sympy.polynomial-intervals"
)


@dataclass(frozen=True, slots=True)
class PolynomialIntervalInstallation:
    semantics_uri: str
    polynomial_semantics_uri: str
    interval_schema_uri: str
    polynomial_schema_uri: str
    enclosure_schema_uri: str
    claim_schema_uri: str
    certificate_schema_uri: str
    checker_id: str | None


@dataclass(frozen=True, slots=True)
class PolynomialIntervalResources:
    store: ArtifactRepository
    artifacts: ArtifactService
    verification: VerificationService
    installation: PolynomialIntervalInstallation


def install_polynomial_interval_capabilities(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[
    tuple[
        PolynomialIntervalEncloseAdapter,
        PolynomialIntervalEnclosureVerifyAdapter | None,
    ],
    PolynomialIntervalInstallation,
]:
    """Register exact univariate polynomial interval enclosure contracts."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.univariate-rational-polynomial-interval",
        version="1",
        definition={
            "description": (
                "exact rational enclosure of one univariate rational polynomial "
                "on one closed rational interval via the Bernstein-coefficient bound"
            ),
            "domain": "QQ",
            "enclosure_kind": "BERNSTEIN_COEFFICIENT_BOUND",
            "range_exactness": "ENCLOSURE_VALID_NOT_EXACT",
            "maximum_degree": 64,
            "maximum_terms": 1024,
            "interval": "closed rational [lo, hi] with lo < hi",
        },
    )
    polynomial_semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.univariate-rational-polynomial-interval-source",
        version="1",
        definition={
            "description": (
                "univariate sparse polynomials over QQ with canonical reduced "
                "rational coefficients serving as interval-enclosure sources"
            ),
            "coefficient_field": "QQ",
            "maximum_degree": 64,
            "maximum_terms": 1024,
            "monomial_order": "descending lexicographic",
            "zero_terms": "omitted",
        },
    )
    interval_schema_uri = schemas.register(
        name="jacobian.rational-interval",
        version="1",
        schema=model_schema(RationalInterval),
    )
    polynomial_schema_uri = schemas.register(
        name="jacobian.univariate-rational-polynomial-interval-source",
        version="1",
        schema=model_schema(UnivariateRationalPolynomial),
    )
    enclosure_schema_uri = schemas.register(
        name="jacobian.polynomial-interval-enclosure",
        version="1",
        schema=model_schema(PolynomialIntervalEnclosure),
    )
    claim_schema_uri = schemas.register(
        name="jacobian.polynomial-interval-enclosure-claim",
        version="1",
        schema=model_schema(PolynomialIntervalEnclosureClaim),
    )
    certificate_schema_uri = schemas.register(
        name="jacobian.polynomial-interval-enclosure-certificate",
        version="1",
        schema=model_schema(CertificateEnvelope),
    )
    checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact rational polynomial interval Bernstein enclosure checker",
                entrypoint="jacobian_checkers.polynomial_intervals:check_enclosure",
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="polynomial.interval_bernstein_enclosure_replay",
                format_version="1",
                claim_schema_uris=(claim_schema_uri,),
                semantics_uris=(semantics_uri,),
                candidate_schema_uris=(enclosure_schema_uri,),
                reason="bundled independent Bernstein-coefficient enclosure checker",
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    installation = PolynomialIntervalInstallation(
        semantics_uri=semantics_uri,
        polynomial_semantics_uri=polynomial_semantics_uri,
        interval_schema_uri=interval_schema_uri,
        polynomial_schema_uri=polynomial_schema_uri,
        enclosure_schema_uri=enclosure_schema_uri,
        claim_schema_uri=claim_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
        checker_id=checker_id,
    )
    resources = PolynomialIntervalResources(
        store=store,
        artifacts=artifacts,
        verification=verification,
        installation=installation,
    )
    verify_adapter = (
        PolynomialIntervalEnclosureVerifyAdapter(resources)
        if checker_id is not None
        else None
    )
    return (
        (PolynomialIntervalEncloseAdapter(resources), verify_adapter),
        installation,
    )


class PolynomialIntervalEncloseAdapter:
    """Compute one exact rational Bernstein-coefficient interval enclosure."""

    def __init__(self, resources: PolynomialIntervalResources) -> None:
        self.resources = resources
        checker_ids = (
            (resources.installation.checker_id,)
            if resources.installation.checker_id is not None
            else ()
        )
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.interval.enclose",
            version="1",
            title="Enclose a univariate polynomial on a rational interval",
            description=(
                "Compute an exact rational enclosure of the values of one "
                "univariate rational polynomial on one closed rational interval "
                "using the Bernstein-coefficient bound. The enclosure is a valid "
                "superset of the range, not the exact image. The result is "
                "UNVERIFIED; the advertised independent Bernstein-coefficient "
                "checker is the verification boundary."
            ),
            provider="jacobian.sympy",
            provider_runtime=known_provider_runtime(
                "jacobian.sympy",
                features=(
                    "univariate-polynomial",
                    "rational-interval",
                    "bernstein-coefficient-bound",
                ),
                checker_ids=checker_ids,
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(PolynomialIntervalEnclosureRequest),
            output_schema=model_schema(PolynomialIntervalEnclosureOutput),
            tags=(
                "polynomial",
                "univariate",
                "interval",
                "enclosure",
                "bernstein",
                "exact-computation",
            ),
            invocation_examples=(
                example(
                    "constant_zero_interval",
                    "Enclose the zero polynomial on [0,1].",
                    {
                        "polynomial": {"variable": "x", "polynomial": {"terms": []}},
                        "interval": {
                            "lo": {"num": "0", "den": "1"},
                            "hi": {"num": "1", "den": "1"},
                        },
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = _validate_request(
            PolynomialIntervalEnclosureRequest,
            request.input,
            code="INVALID_POLYNOMIAL_INTERVAL_ENCLOSURE_REQUEST",
            operation="interval enclosure",
        )
        started = time.monotonic()
        polynomial = validated.polynomial
        interval = validated.interval
        sp = _sympy.get()
        try:
            coefficients = _bernstein_coefficients(polynomial, interval)
        except (
            cast(type[BaseException], sp.PolynomialError),
            TypeError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise _interval_error(
                "POLYNOMIAL_INTERVAL_ENCLOSURE_FAILED",
                "enclosure_computation",
                "The exact Bernstein-coefficient enclosure computation failed.",
            ) from exc
        polynomial_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.polynomial_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=polynomial.model_dump(mode="json"),
            summary="exact univariate rational polynomial interval source",
        )
        enclosure = PolynomialIntervalEnclosure(
            polynomial_uri=polynomial_artifact.artifact_uri,
            interval=interval,
            degree=polynomial.degree,
            bernstein_coefficients=coefficients,
            lo=_rational(min(c.as_fraction() for c in coefficients)),
            hi=_rational(max(c.as_fraction() for c in coefficients)),
            backend_version=SYMPY_VERSION,
        )
        enclosure_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.enclosure_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=enclosure.model_dump(mode="json"),
            parents=(polynomial_artifact.artifact_uri,),
            summary="exact rational Bernstein-coefficient interval enclosure",
        )
        output = PolynomialIntervalEnclosureOutput(
            polynomial_uri=polynomial_artifact.artifact_uri,
            enclosure_uri=enclosure_artifact.artifact_uri,
            interval=interval,
            degree=enclosure.degree,
            bernstein_coefficients=enclosure.bernstein_coefficients,
            lo=enclosure.lo,
            hi=enclosure.hi,
            backend_version=SYMPY_VERSION,
        )
        checker_hint = (
            "invoke polynomial.interval.enclosure.verify with the authorized "
            "Bernstein-coefficient checker to obtain a VERIFIED record"
            if self.resources.installation.checker_id is not None
            else "no independent checker is authorized in this installation; the "
            "enclosure remains UNVERIFIED"
        )
        return _computed_result(
            descriptor=self.descriptor,
            request=request,
            started=started,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description=(
                    "one closed rational interval for one univariate polynomial"
                ),
                parameters={
                    "polynomial_uri": polynomial_artifact.artifact_uri,
                    "interval": interval.model_dump(mode="json"),
                    "degree": polynomial.degree,
                },
                artifact_uri=polynomial_artifact.artifact_uri,
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id="polynomial.relation.enclosure-of",
                    source_artifact_uris=(polynomial_artifact.artifact_uri,),
                    target_artifact_uris=(enclosure_artifact.artifact_uri,),
                ),
            ),
            artifact_uris=(
                polynomial_artifact.artifact_uri,
                enclosure_artifact.artifact_uri,
            ),
            completeness_basis=(
                "the Bernstein-coefficient bound covers the entire declared "
                "interval; the bound is a valid enclosure, not the exact range"
            ),
            assurance_basis=(
                "deterministic exact SymPy rational arithmetic over QQ produced "
                "the Bernstein coefficients; the computation did not authorize or "
                "invoke an independent checker; " + checker_hint
            ),
        )


class PolynomialIntervalEnclosureVerifyAdapter:
    """Verify one claimed Bernstein-coefficient enclosure independently."""

    def __init__(self, resources: PolynomialIntervalResources) -> None:
        self.resources = resources
        checker_id = resources.installation.checker_id
        assert checker_id is not None
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.interval.enclosure.verify",
            version="1",
            title="Verify a polynomial interval Bernstein enclosure",
            description=(
                "Independently replay the Bernstein coefficients of one univariate "
                "rational polynomial on one closed rational interval and confirm "
                "that the claimed enclosure is the valid Bernstein-coefficient "
                "bound. The checker uses pure rational arithmetic and does not "
                "import SymPy. Verification confirms enclosure validity, not "
                "equality with the exact polynomial range."
            ),
            provider="jacobian.exact-polynomial-interval-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.exact-polynomial-interval-checker",
                features=(
                    "polynomial-interval",
                    "bernstein-coefficient",
                    "exact-rational",
                ),
                checker_ids=(checker_id,),
            ),
            modes=(CapabilityMode.VERIFY,),
            input_schema=model_schema(PolynomialIntervalEnclosureVerifyRequest),
            output_schema=model_schema(PolynomialIntervalEnclosureVerifyOutput),
            tags=(
                "polynomial",
                "univariate",
                "interval",
                "enclosure",
                "bernstein",
                "verification",
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = _validate_request(
            PolynomialIntervalEnclosureVerifyRequest,
            request.input,
            code="INVALID_POLYNOMIAL_INTERVAL_ENCLOSURE_VERIFY_REQUEST",
            operation="interval enclosure verification",
        )
        installation = self.resources.installation
        checker_id = installation.checker_id
        assert checker_id is not None
        polynomial = validated.polynomial
        interval = validated.interval
        polynomial_artifact = self.resources.artifacts.put(
            schema_uri=installation.polynomial_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=polynomial.model_dump(mode="json"),
            summary="exact univariate rational polynomial interval source",
        )
        enclosure = PolynomialIntervalEnclosure(
            polynomial_uri=polynomial_artifact.artifact_uri,
            interval=interval,
            degree=polynomial.degree,
            bernstein_coefficients=validated.claimed_bernstein_coefficients,
            lo=validated.claimed_lo,
            hi=validated.claimed_hi,
            backend_version=SYMPY_VERSION,
        )
        enclosure_artifact = self.resources.artifacts.put(
            schema_uri=installation.enclosure_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=enclosure.model_dump(mode="json"),
            parents=(polynomial_artifact.artifact_uri,),
            summary="claimed exact rational Bernstein-coefficient interval enclosure",
        )
        claim = PolynomialIntervalEnclosureClaim(
            polynomial_uri=polynomial_artifact.artifact_uri,
            interval=interval,
        )
        claim_artifact = self.resources.artifacts.put(
            schema_uri=installation.claim_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=claim.model_dump(mode="json"),
            parents=(
                polynomial_artifact.artifact_uri,
                enclosure_artifact.artifact_uri,
            ),
            summary="polynomial interval Bernstein enclosure claim",
        )
        semantics = self.resources.store.get(installation.semantics_uri)
        replay = PolynomialIntervalEnclosureReplay(
            polynomial_uri=polynomial_artifact.artifact_uri,
            interval=interval,
            degree=polynomial.degree,
            bernstein_coefficients=validated.claimed_bernstein_coefficients,
            lo=validated.claimed_lo,
            hi=validated.claimed_hi,
        ).model_dump(mode="json")
        certificate = CertificateEnvelope(
            certificate_type="polynomial.interval_bernstein_enclosure_replay",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=claim_artifact.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=enclosure_artifact.object_digest,
                scope_digest=polynomial_artifact.object_digest,
            ),
            payload_digest=(
                "sha256:" + hashlib.sha256(canonicalize_json(replay)).hexdigest()
            ),
            payload=replay,
        )
        certificate_artifact = self.resources.artifacts.put(
            schema_uri=installation.certificate_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(
                claim_artifact.artifact_uri,
                enclosure_artifact.artifact_uri,
                polynomial_artifact.artifact_uri,
            ),
            summary="exact polynomial interval Bernstein enclosure replay certificate",
        )
        checked = self.resources.verification.verify_certificate(
            certificate_uri=certificate_artifact.artifact_uri,
            checker_id=checker_id,
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.conclusion in {Conclusion.TRUE, Conclusion.FALSE}
            and checked.assurance.verification is Verification.VERIFIED
            and checked.verification_record_uri is not None
        )
        conclusion = cast(
            Literal["TRUE", "FALSE", "UNKNOWN"],
            checked.conclusion.value,
        )
        record_uri = checked.verification_record_uri if verified else None
        output = PolynomialIntervalEnclosureVerifyOutput(
            polynomial_uri=polynomial_artifact.artifact_uri,
            enclosure_uri=enclosure_artifact.artifact_uri,
            claim_uri=claim_artifact.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            verification_record_uri=record_uri,
            checker_id=checker_id,
            interval=interval,
            degree=polynomial.degree,
            bernstein_coefficients=validated.claimed_bernstein_coefficients,
            lo=validated.claimed_lo,
            hi=validated.claimed_hi,
            enclosure_assurance="VERIFIED" if verified else "COMPUTED",
            conclusion=conclusion,
        )
        artifact_uris = [
            polynomial_artifact.artifact_uri,
            enclosure_artifact.artifact_uri,
            claim_artifact.artifact_uri,
            certificate_artifact.artifact_uri,
        ]
        if record_uri is not None:
            artifact_uris.append(record_uri)
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=checked.execution,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description=(
                    "one closed rational interval for one univariate polynomial"
                ),
                parameters={
                    "polynomial_uri": polynomial_artifact.artifact_uri,
                    "interval": interval.model_dump(mode="json"),
                    "degree": polynomial.degree,
                },
                artifact_uri=polynomial_artifact.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if checked.execution.status is ExecutionStatus.COMPLETED
                    else CapabilityCompletenessStatus.UNKNOWN
                ),
                basis=(
                    "the independent checker replayed every Bernstein coefficient "
                    "over the declared interval"
                    if verified
                    else (
                        "the adapter packaged the claimed enclosure, but the "
                        "checker did not accept the bound replay"
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else "checker execution did not establish complete coverage"
                    )
                ),
                assurance_level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else (
                        CapabilityAssuranceLevel.COMPUTED
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else CapabilityAssuranceLevel.HEURISTIC
                    )
                ),
                verification_record_uri=record_uri,
            ),
            relationships=(
                (
                    CapabilityRelationship(
                        relation_id="polynomial.relation.valid-bernstein-enclosure",
                        source_artifact_uris=(enclosure_artifact.artifact_uri,),
                        target_artifact_uris=(polynomial_artifact.artifact_uri,),
                        status=CapabilityRelationshipStatus.VERIFIED,
                        verification_record_uri=record_uri,
                    ),
                )
                if conclusion == "TRUE" and verified
                else ()
            ),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else (
                        CapabilityAssuranceLevel.COMPUTED
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else CapabilityAssuranceLevel.HEURISTIC
                    )
                ),
                basis=(
                    "accepted by the authorized independent Bernstein-coefficient "
                    "checker; the enclosure is a valid Bernstein bound, not the "
                    "exact polynomial range"
                    if verified
                    else (
                        "the claimed enclosure was packaged, but the independent "
                        "checker did not accept the bound replay"
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else "checker execution did not complete; no mathematical "
                        "conclusion follows"
                    )
                ),
                verification_record_uri=record_uri,
            ),
            artifact_uris=tuple(artifact_uris),
        )


def _validate_request[RequestModel: ContractModel](
    model: type[RequestModel],
    payload: object,
    *,
    code: str,
    operation: str,
) -> RequestModel:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise _interval_error(
            code,
            "request_validation",
            f"The complete polynomial {operation} request is invalid.",
        ) from exc


def _bernstein_coefficients(
    polynomial: UnivariateRationalPolynomial,
    interval: RationalInterval,
) -> tuple[CanonicalRational, ...]:
    """Compute exact Bernstein coefficients of ``polynomial`` on ``interval``.

    Substitutes ``x = a + (b - a) t`` to map the interval to ``[0, 1]``, then
    converts the resulting power-basis polynomial to the degree-``d`` Bernstein
    basis using the exact rational change-of-basis formula
    ``b_i = sum_{k=0}^{i} a_k * C(i, k) / C(d, k)``.
    """

    a = interval.lo.as_fraction()
    b = interval.hi.as_fraction()
    width = b - a
    degree = polynomial.degree
    sp = _sympy.get()
    generator: Symbol = sp.symbols(polynomial.variable)
    terms = {}
    for term in polynomial.polynomial.terms:
        coefficient = term.coefficient.as_fraction()
        terms[term.exponents] = sp.QQ(coefficient.numerator, coefficient.denominator)
    source = sp.Poly.from_dict(terms, generator, domain=sp.QQ)
    shifted = sp.Poly(
        sp.expand(source.as_expr().subs(generator, a + width * generator)),
        generator,
        domain=sp.QQ,
    )
    power_coefficients = {
        exponent: Fraction(int(coeff.p), int(coeff.q))
        for exponent, coeff in shifted.terms()
    }
    coefficients: list[CanonicalRational] = []
    for i in range(degree + 1):
        accumulator = Fraction(0)
        for k in range(i + 1):
            ak = power_coefficients.get((k,), Fraction(0))
            if ak == 0:
                continue
            accumulator += ak * Fraction(comb(i, k), comb(degree, k))
        coefficients.append(_rational(accumulator))
    return tuple(coefficients)


def _rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational(num=str(value.numerator), den=str(value.denominator))


def _computed_result(
    *,
    descriptor: CapabilityDescriptor,
    request: CapabilityRequest,
    started: float,
    output: dict[str, Any],
    scope: CapabilityScope,
    relationships: tuple[CapabilityRelationship, ...],
    artifact_uris: tuple[str, ...],
    completeness_basis: str,
    assurance_basis: str,
) -> CapabilityResult:
    return CapabilityResult(
        capability_id=descriptor.capability_id,
        capability_version=descriptor.version,
        mode=request.mode,
        execution=Execution(
            status=ExecutionStatus.COMPLETED,
            runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
        ),
        output=output,
        scope=scope,
        completeness=CapabilityCompleteness(
            status=CapabilityCompletenessStatus.COMPLETE,
            basis=(
                f"{completeness_basis}; no mathematical conclusion or "
                "independent verification is claimed"
            ),
            assurance_level=CapabilityAssuranceLevel.COMPUTED,
        ),
        relationships=relationships,
        assurance=CapabilityAssurance(
            level=CapabilityAssuranceLevel.COMPUTED,
            basis=assurance_basis,
        ),
        artifact_uris=artifact_uris,
    )


def _interval_error(
    code: str,
    stage: str,
    message: str,
) -> CapabilityInvocationError:
    return CapabilityInvocationError(
        CapabilityDiagnostic(
            code=code,
            stage=stage,
            message=message,
            hint=(
                "Use the advertised univariate QQ schema with reduced rationals, "
                "descending monomial order, a single variable, degree at most 64, "
                "and a non-degenerate rational interval lo < hi."
            ),
        )
    )


__all__ = (
    "PolynomialIntervalEncloseAdapter",
    "PolynomialIntervalEnclosureVerifyAdapter",
    "PolynomialIntervalInstallation",
    "PolynomialIntervalResources",
    "install_polynomial_interval_capabilities",
)
