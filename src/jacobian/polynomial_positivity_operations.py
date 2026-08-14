"""SymPy-backed exact rational univariate polynomial strict positivity.

This module installs two domain-atomic operations over one mathematical
outcome: an exact decision whether one univariate rational polynomial is
strictly positive on one closed rational interval, using Sturm's theorem.

``polynomial.interval.positivity.decide`` (EXPLORE) computes the Sturm
sequence with pinned SymPy, counts sign changes at the interval endpoints,
and determines whether p(x) > 0 for all x in [a,b]. It creates no verification
record.

``polynomial.interval.positivity.verify`` (VERIFY) packages the claimed
decision as a replay certificate and asks the operator-authorized independent
checker to recompute the Sturm sequence from scratch using pure
``fractions.Fraction`` arithmetic. The checker does not import SymPy.

The decision is exact: Sturm's theorem is a theorem, not a heuristic. A
``TRUE`` conclusion means p(x) > 0 for every x in [a,b]; a ``FALSE``
conclusion means p(x) <= 0 for at least one x in [a,b].
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.checker_authorization import authorize_checker_operation
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationRequest,
)
from jacobian.contracts.polynomial_intervals import (
    RationalInterval,
    UnivariateRationalPolynomial,
)
from jacobian.contracts.polynomial_positivity import (
    PolynomialIntervalPositivityClaim,
    PolynomialIntervalPositivityDecision,
    PolynomialIntervalPositivityOutput,
    PolynomialIntervalPositivityReplay,
    PolynomialIntervalPositivityRequest,
    PolynomialIntervalPositivityVerifyOutput,
    PolynomialIntervalPositivityVerifyRequest,
)
from jacobian.contracts.polynomials import (
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.contracts.results import (
    Conclusion,
    ExecutionStatus,
)
from jacobian.domains._examples import example
from jacobian.operation_catalog import OperationCatalog, OperationCatalogError
from jacobian.operation_errors import OperationInvocationError
from jacobian.operation_projection import OperationProjection
from jacobian.polynomials._support import (
    PolynomialOperationResult,
    _computed_result,
    _validate_request,
    _wire_rational,
)
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime
from jacobian.providers import LazyLoader
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService

if TYPE_CHECKING:
    from sympy import Poly, Symbol


class _SympyBackend(NamedTuple):
    """Heavy SymPy implementation symbols loaded on first operation invocation."""

    QQ: Any
    Poly: Any
    symbols: Any
    Rational: Any
    PolynomialError: type


def _load_sympy_backend() -> _SympyBackend:
    """Construct the pinned SymPy implementation bundle on first use."""
    from sympy import QQ, Poly, Rational, symbols
    from sympy.polys.polyerrors import PolynomialError

    return _SympyBackend(QQ, Poly, symbols, Rational, PolynomialError)


_sympy: LazyLoader[_SympyBackend] = LazyLoader(
    _load_sympy_backend, component_id="jacobian.sympy.polynomial-positivity"
)


@dataclass(frozen=True, slots=True)
class PolynomialPositivityInstallation:
    semantics_uri: str
    polynomial_semantics_uri: str
    polynomial_schema_uri: str
    decision_schema_uri: str
    claim_schema_uri: str
    certificate_schema_uri: str
    checker_id: str | None


@dataclass(frozen=True, slots=True)
class PolynomialPositivityResources:
    store: ArtifactRepository
    artifacts: ArtifactService
    verification: VerificationService
    installation: PolynomialPositivityInstallation


def register_polynomial_positivity_resources(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
) -> PolynomialPositivityInstallation:
    """Register passive positivity contracts without checker installation."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.univariate-rational-polynomial-positivity",
        version="1",
        definition={
            "description": (
                "exact Sturm-sequence decision on strict positivity of one "
                "univariate rational polynomial on one closed rational interval"
            ),
            "domain": "QQ",
            "decision_kind": "STRICT_POSITIVITY",
            "exactness": "EXACT_RATIONAL",
            "method": "STURM_SEQUENCE",
            "maximum_degree": 64,
            "maximum_terms": 1024,
            "interval": "closed rational [lo, hi] with lo < hi",
        },
    )
    polynomial_semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.univariate-rational-polynomial-positivity-source",
        version="1",
        definition={
            "description": (
                "univariate sparse polynomials over QQ serving as strict-"
                "positivity decision sources"
            ),
            "coefficient_field": "QQ",
            "maximum_degree": 64,
            "maximum_terms": 1024,
            "monomial_order": "descending lexicographic",
            "zero_terms": "omitted",
        },
    )
    return PolynomialPositivityInstallation(
        semantics_uri=semantics_uri,
        polynomial_semantics_uri=polynomial_semantics_uri,
        polynomial_schema_uri=schemas.register(
            name="jacobian.univariate-rational-polynomial-positivity-source",
            version="1",
            schema=model_schema(UnivariateRationalPolynomial),
        ),
        decision_schema_uri=schemas.register(
            name="jacobian.polynomial-interval-positivity-decision",
            version="1",
            schema=model_schema(PolynomialIntervalPositivityDecision),
        ),
        claim_schema_uri=schemas.register(
            name="jacobian.polynomial-interval-positivity-claim",
            version="1",
            schema=model_schema(PolynomialIntervalPositivityClaim),
        ),
        certificate_schema_uri=schemas.register(
            name="jacobian.polynomial-interval-positivity-certificate",
            version="1",
            schema=model_schema(CertificateEnvelope),
        ),
        checker_id=None,
    )


def bind_selected_polynomial_positivity_operation(
    operation_id: str,
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    catalog: OperationCatalog,
) -> (
    PolynomialIntervalPositivityDecideAdapter
    | PolynomialIntervalPositivityVerifyAdapter
    | None
):
    """Bind one positivity operation from passive resources and catalog authority."""

    if operation_id not in {
        "polynomial.interval.positivity.decide",
        "polynomial.interval.positivity.verify",
    }:
        return None
    installation = register_polynomial_positivity_resources(store, schemas)
    binding = catalog.checker_binding("polynomial.interval.positivity.verify")
    if binding is None:
        raise OperationCatalogError(
            "checker binding is missing; run `jacobian update`: "
            "polynomial.interval.positivity.verify"
        )
    checkers.require_catalog_binding(
        binding.checker_id,
        implementation_digest=binding.manifest_digest,
    )
    resources = PolynomialPositivityResources(
        store,
        artifacts,
        verification,
        replace(installation, checker_id=binding.checker_id),
    )
    if operation_id == "polynomial.interval.positivity.decide":
        return PolynomialIntervalPositivityDecideAdapter(resources)
    return PolynomialIntervalPositivityVerifyAdapter(resources)


def install_polynomial_positivity_operations(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[
    tuple[
        PolynomialIntervalPositivityDecideAdapter,
        PolynomialIntervalPositivityVerifyAdapter | None,
    ],
    PolynomialPositivityInstallation,
]:
    """Register exact univariate polynomial strict positivity contracts."""

    contracts = register_polynomial_positivity_resources(store, schemas)
    checker_id = authorize_checker_operation(
        checkers,
        CheckerOperation(
            name="exact rational polynomial interval strict positivity checker",
            entrypoint="jacobian_checkers.polynomial_positivity:check_positivity",
            evidence_kind=EvidenceKind.CERTIFICATE,
            format_id="polynomial.interval_sturm_positivity_replay",
            format_version="1",
            claim_schema_uris=(contracts.claim_schema_uri,),
            semantics_uris=(contracts.semantics_uri,),
            candidate_schema_uris=(contracts.decision_schema_uri,),
            reason="bundled independent Sturm-sequence positivity checker",
        ),
        authorize=authorize_checker,
    ).checker_id
    installation = replace(contracts, checker_id=checker_id)
    resources = PolynomialPositivityResources(
        store=store,
        artifacts=artifacts,
        verification=verification,
        installation=installation,
    )
    verify_adapter = (
        PolynomialIntervalPositivityVerifyAdapter(resources)
        if checker_id is not None
        else None
    )
    return (
        (PolynomialIntervalPositivityDecideAdapter(resources), verify_adapter),
        installation,
    )


class PolynomialIntervalPositivityDecideAdapter:
    """Decide exact strict positivity via Sturm's theorem."""

    def __init__(self, resources: PolynomialPositivityResources) -> None:
        self.resources = resources
        checker_ids = (
            (resources.installation.checker_id,)
            if resources.installation.checker_id is not None
            else ()
        )
        self._descriptor = OperationDescriptor(
            operation_id="polynomial.interval.positivity.decide",
            version="1",
            title="Decide strict positivity on a rational interval",
            description=(
                "Decide exactly whether one univariate rational polynomial is "
                "strictly positive (p(x) > 0) on one closed rational interval "
                "using Sturm's theorem. The decision is EXACT and "
                "DETERMINISTIC. The separate independent Sturm-sequence checker "
                "is the verification boundary. The input difference polynomial can "
                "represent a strict inequality such as one expression exceeding a "
                "bound."
            ),
            provider="jacobian.sympy",
            provider_runtime=known_provider_runtime(
                "jacobian.sympy",
                features=(
                    "univariate-polynomial",
                    "sturm-sequence",
                    "exact-positivity",
                ),
                checker_ids=checker_ids,
            ),
            input_schema=model_schema(PolynomialIntervalPositivityRequest),
            output_schema=model_schema(PolynomialIntervalPositivityOutput),
            tags=(
                "polynomial",
                "univariate",
                "interval",
                "positivity",
                "sturm",
                "exact-decision",
                "inequality",
                "exceeds-bound",
                "rational-derivative-bound",
            ),
            examples=(
                example(
                    "constant_one_positive",
                    "Decide positivity of 1 on [0,1].",
                    {
                        "polynomial": {
                            "variable": "x",
                            "polynomial": {
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "exponents": [0],
                                    }
                                ]
                            },
                        },
                        "interval": {
                            "lo": {"num": "0", "den": "1"},
                            "hi": {"num": "1", "den": "1"},
                        },
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> PolynomialIntervalPositivityRequest:
        return _validate_request(
            PolynomialIntervalPositivityRequest,
            request.input,
            code="INVALID_POLYNOMIAL_POSITIVITY_REQUEST",
            operation="positivity decision",
            error_factory=_positivity_error,
        )

    def invoke(
        self, validated: PolynomialIntervalPositivityRequest
    ) -> OperationProjection:
        started = time.monotonic()
        polynomial = validated.polynomial
        interval = validated.interval
        sp = _sympy.get()
        try:
            (
                sturm_polys,
                sign_changes_lo,
                sign_changes_hi,
                roots_in_open,
                endpoint_root,
                positive,
            ) = _sturm_positivity(polynomial, interval)
        except (
            cast(type[BaseException], sp.PolynomialError),
            TypeError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise _positivity_error(
                "POLYNOMIAL_POSITIVITY_DECISION_FAILED",
                "positivity_computation",
                "The exact Sturm-sequence positivity decision failed.",
            ) from exc
        polynomial_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.polynomial_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=polynomial.model_dump(mode="json"),
            summary="exact univariate rational polynomial positivity source",
        )
        decision = PolynomialIntervalPositivityDecision(
            polynomial_uri=polynomial_artifact.artifact_uri,
            interval=interval,
            degree=polynomial.degree,
            sturm_sequence=sturm_polys,
            sign_changes_at_lo=sign_changes_lo,
            sign_changes_at_hi=sign_changes_hi,
            roots_in_open_interval=roots_in_open,
            endpoint_root=endpoint_root,
            positive=positive,
            backend_version=SYMPY_VERSION,
        )
        decision_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.decision_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=decision.model_dump(mode="json"),
            parents=(polynomial_artifact.artifact_uri,),
            summary="exact Sturm-sequence strict positivity decision",
        )
        output = PolynomialIntervalPositivityOutput(
            polynomial_uri=polynomial_artifact.artifact_uri,
            decision_uri=decision_artifact.artifact_uri,
            interval=interval,
            degree=polynomial.degree,
            sign_changes_at_lo=sign_changes_lo,
            sign_changes_at_hi=sign_changes_hi,
            roots_in_open_interval=roots_in_open,
            endpoint_root=endpoint_root,
            positive=positive,
        )
        return _computed_result(
            descriptor=self.descriptor,
            started=started,
            output=output,
            artifact_uris=(
                polynomial_artifact.artifact_uri,
                decision_artifact.artifact_uri,
            ),
        )


class PolynomialIntervalPositivityVerifyAdapter:
    """Verify one claimed Sturm-sequence positivity decision independently."""

    def __init__(self, resources: PolynomialPositivityResources) -> None:
        self.resources = resources
        checker_id = resources.installation.checker_id
        if checker_id is None:
            raise RuntimeError(
                "polynomial positivity verify adapter requires an authorized checker"
            )
        self._descriptor = OperationDescriptor(
            operation_id="polynomial.interval.positivity.verify",
            version="1",
            title="Verify a polynomial interval positivity decision",
            description=(
                "Independently recompute the Sturm sequence of one univariate "
                "rational polynomial on one closed rational interval and confirm "
                "the claimed strict positivity decision. The checker uses pure "
                "rational arithmetic and does not import SymPy. The input difference "
                "polynomial can represent a strict inequality such as one expression "
                "exceeding a bound."
            ),
            provider="jacobian.exact-polynomial-positivity-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.exact-polynomial-positivity-checker",
                features=(
                    "polynomial-interval",
                    "sturm-sequence",
                    "exact-rational",
                ),
                checker_ids=(checker_id,),
            ),
            input_schema=model_schema(PolynomialIntervalPositivityVerifyRequest),
            output_schema=model_schema(PolynomialIntervalPositivityVerifyOutput),
            tags=(
                "polynomial",
                "univariate",
                "interval",
                "positivity",
                "sturm",
                "verification",
                "verify",
                "inequality",
                "exceeds-bound",
                "rational-derivative-bound",
            ),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(
        self, request: OperationRequest
    ) -> PolynomialIntervalPositivityVerifyRequest:
        return _validate_request(
            PolynomialIntervalPositivityVerifyRequest,
            request.input,
            code="INVALID_POLYNOMIAL_POSITIVITY_VERIFY_REQUEST",
            operation="positivity verification",
            error_factory=_positivity_error,
        )

    def invoke(
        self, validated: PolynomialIntervalPositivityVerifyRequest
    ) -> OperationProjection:
        installation = self.resources.installation
        checker_id = installation.checker_id
        if checker_id is None:
            raise _positivity_error(
                "POLYNOMIAL_POSITIVITY_CHECKER_UNAVAILABLE",
                "positivity_verification",
                "The independent positivity checker is not installed in this runtime.",
            )
        polynomial = validated.polynomial
        interval = validated.interval
        polynomial_artifact = self.resources.artifacts.put(
            schema_uri=installation.polynomial_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=polynomial.model_dump(mode="json"),
            summary="exact univariate rational polynomial positivity source",
        )
        sp = _sympy.get()
        try:
            sturm_polys, _, _, _, _, _ = _sturm_positivity(polynomial, interval)
        except (
            cast(type[BaseException], sp.PolynomialError),
            TypeError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise _positivity_error(
                "POLYNOMIAL_POSITIVITY_VERIFY_FAILED",
                "positivity_verification",
                "The Sturm sequence computation for verification packaging failed.",
            ) from exc
        decision = PolynomialIntervalPositivityDecision(
            polynomial_uri=polynomial_artifact.artifact_uri,
            interval=interval,
            degree=polynomial.degree,
            sturm_sequence=sturm_polys,
            sign_changes_at_lo=validated.claimed_sign_changes_at_lo,
            sign_changes_at_hi=validated.claimed_sign_changes_at_hi,
            roots_in_open_interval=validated.claimed_roots_in_open_interval,
            endpoint_root=validated.claimed_endpoint_root,
            positive=validated.claimed_positive,
            backend_version=SYMPY_VERSION,
        )
        decision_artifact = self.resources.artifacts.put(
            schema_uri=installation.decision_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=decision.model_dump(mode="json"),
            parents=(polynomial_artifact.artifact_uri,),
            summary="claimed exact Sturm-sequence positivity decision",
        )
        claim = PolynomialIntervalPositivityClaim(
            polynomial_uri=polynomial_artifact.artifact_uri,
            interval=interval,
            positive=validated.claimed_positive,
        )
        claim_artifact = self.resources.artifacts.put(
            schema_uri=installation.claim_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=claim.model_dump(mode="json"),
            parents=(
                polynomial_artifact.artifact_uri,
                decision_artifact.artifact_uri,
            ),
            summary="polynomial interval strict positivity claim",
        )
        semantics = self.resources.store.get(installation.semantics_uri)
        replay = PolynomialIntervalPositivityReplay(
            polynomial_uri=polynomial_artifact.artifact_uri,
            interval=interval,
            degree=polynomial.degree,
            sturm_sequence_length=max(1, polynomial.degree + 1),
            sign_changes_at_lo=validated.claimed_sign_changes_at_lo,
            sign_changes_at_hi=validated.claimed_sign_changes_at_hi,
            roots_in_open_interval=validated.claimed_roots_in_open_interval,
            endpoint_root=validated.claimed_endpoint_root,
            positive=validated.claimed_positive,
        ).model_dump(mode="json")
        certificate = CertificateEnvelope(
            certificate_type="polynomial.interval_sturm_positivity_replay",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=claim_artifact.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=decision_artifact.object_digest,
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
                decision_artifact.artifact_uri,
                polynomial_artifact.artifact_uri,
            ),
            summary="exact Sturm-sequence positivity replay certificate",
        )
        checked = self.resources.verification.verify_certificate(
            certificate_uri=certificate_artifact.artifact_uri,
            checker_id=checker_id,
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.conclusion in {Conclusion.TRUE, Conclusion.FALSE}
            and checked.verification_record_uri is not None
        )
        conclusion: Literal["TRUE", "FALSE", "UNKNOWN"] = (
            "TRUE"
            if verified and checked.conclusion is Conclusion.TRUE
            else (
                "FALSE"
                if verified and checked.conclusion is Conclusion.FALSE
                else "UNKNOWN"
            )
        )
        record_uri = checked.verification_record_uri if verified else None
        output = PolynomialIntervalPositivityVerifyOutput(
            polynomial_uri=polynomial_artifact.artifact_uri,
            decision_uri=decision_artifact.artifact_uri,
            claim_uri=claim_artifact.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            verification_record_uri=record_uri,
            checker_id=checker_id,
            interval=interval,
            degree=polynomial.degree,
            positive=validated.claimed_positive,
            sign_changes_at_lo=validated.claimed_sign_changes_at_lo,
            sign_changes_at_hi=validated.claimed_sign_changes_at_hi,
            roots_in_open_interval=validated.claimed_roots_in_open_interval,
            endpoint_root=validated.claimed_endpoint_root,
            conclusion=conclusion,
        )
        artifact_uris = [
            polynomial_artifact.artifact_uri,
            decision_artifact.artifact_uri,
            claim_artifact.artifact_uri,
            certificate_artifact.artifact_uri,
        ]
        if record_uri is not None:
            artifact_uris.append(record_uri)
        return PolynomialOperationResult(
            execution=checked.execution,
            value=output,
            verification_record_uri=(record_uri if verified else None),
            artifact_uris=tuple(artifact_uris),
        ).project(self.descriptor)


def _sturm_positivity(
    polynomial: UnivariateRationalPolynomial,
    interval: RationalInterval,
) -> tuple[
    tuple[SparseRationalPolynomial, ...],
    int,
    int,
    int,
    bool,
    bool,
]:
    """Compute the Sturm sequence and decide strict positivity.

    Returns (sturm_polys, sign_changes_at_lo, sign_changes_at_hi,
    roots_in_open_interval, endpoint_root, positive).

    p > 0 on [a,b] iff p(a) > 0 and there are no roots of p in [a,b].
    Roots in (a,b] = V(a) - V(b) by Sturm's theorem.
    Root at a = (p(a) == 0).
    """

    a = interval.lo.as_fraction()
    b = interval.hi.as_fraction()
    sp = _sympy.get()
    generator: Symbol = sp.symbols(polynomial.variable)
    terms = {}
    for term in polynomial.polynomial.terms:
        coefficient = term.coefficient.as_fraction()
        terms[term.exponents] = sp.QQ(coefficient.numerator, coefficient.denominator)
    source = sp.Poly.from_dict(terms, generator, domain=sp.QQ)
    degree = polynomial.degree
    if degree == 0:
        constant = terms.get((0,), sp.QQ(0))
        p_at_a = Fraction(int(constant.p), int(constant.q))
        positive = p_at_a > 0
        sturm_poly = _wire_univariate_poly(source)
        return (
            (sturm_poly,),
            0,
            0,
            0,
            p_at_a == 0,
            positive,
        )
    sturm_sequence = source.sturm()
    sturm_polys = tuple(_wire_univariate_poly(poly) for poly in sturm_sequence)
    sign_changes_lo = _count_sign_changes(sturm_sequence, a, generator)
    sign_changes_hi = _count_sign_changes(sturm_sequence, b, generator)
    roots_in_open = sign_changes_lo - sign_changes_hi
    p_at_a = Fraction(*source.eval(a).as_numer_denom())
    endpoint_root = p_at_a == 0
    positive = p_at_a > 0 and roots_in_open == 0 and not endpoint_root
    return (
        sturm_polys,
        sign_changes_lo,
        sign_changes_hi,
        roots_in_open,
        endpoint_root,
        positive,
    )


def _count_sign_changes(
    sequence: tuple[Poly, ...],
    point: Fraction,
    generator: Symbol,
) -> int:
    signs: list[int] = []
    for poly in sequence:
        value = poly.eval(point)
        rational = Fraction(int(value.p), int(value.q))
        if rational > 0:
            signs.append(1)
        elif rational < 0:
            signs.append(-1)
    changes = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            changes += 1
    return changes


def _wire_univariate_poly(poly: Poly) -> SparseRationalPolynomial:
    return SparseRationalPolynomial(
        terms=tuple(
            RationalPolynomialTerm(
                coefficient=_wire_rational(coefficient),
                exponents=(exponent,),
            )
            for (exponent,), coefficient in poly.terms()
            if coefficient != 0
        )
    )


def _positivity_error(
    code: str,
    stage: str,
    message: str,
) -> OperationInvocationError:
    return OperationInvocationError(
        OperationDiagnostic(
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
    "PolynomialIntervalPositivityDecideAdapter",
    "PolynomialIntervalPositivityVerifyAdapter",
    "PolynomialPositivityInstallation",
    "PolynomialPositivityResources",
    "install_polynomial_positivity_operations",
)
